"""Local-only browser demo. No framework or cloud calls are needed for inference."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, UnidentifiedImageError

from inference import CHECKPOINT, Segmenter

STATIC = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000


def create_server(host, port, segmenter):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def json_reply(self, code, value):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/api/status":
                self.json_reply(200, {"checkpoint": segmenter.checkpoint.is_file(),
                                      "loaded": segmenter.model is not None,
                                      "device": "CPU", "model": "PSPNet R101 · PPM 1237"})
                return
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if self.path not in files:
                self.send_error(404)
                return
            name, content_type = files[self.path]
            data = (STATIC / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/api/predict":
                self.json_reply(404, {"error": "Không có API này."})
                return
            # Same-origin browser only. There is no cross-origin API access.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json_reply(403, {"error": "Chỉ nhận yêu cầu từ giao diện local."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                base_size = int(self.headers.get("X-Base-Size", "512"))
                if length < 1 or length > MAX_UPLOAD or base_size not in (384, 512, 768):
                    raise ValueError("Ảnh tối đa 20 MB; kích thước suy luận phải là 384, 512 hoặc 768.")
                payload = self.rfile.read(length)
                image = Image.open(BytesIO(payload))
                if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Ảnh tối đa 16 megapixel. Hãy giảm kích thước ảnh trước.")
                image.load()
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                self.json_reply(400, {"error": str(error)})
                return
            if not inference_lock.acquire(blocking=False):
                self.json_reply(409, {"error": "CPU đang xử lý một ảnh khác. Vui lòng chờ."})
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def emit(event):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                try:
                    segmenter.predict(image, emit, base_size=base_size)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    try:
                        emit({"type": "error", "error": str(error)})
                    except OSError:
                        pass
            finally:
                inference_lock.release()

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    server = create_server("127.0.0.1", args.port, Segmenter(args.checkpoint, args.threads))
    print(f"PSPNet CPU demo: http://127.0.0.1:{args.port}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
