"use strict";
const $ = id => document.getElementById(id);
let selectedFile = null, objectUrl = null, busy = false, ready = false, result = null;
let startTime = 0, timer = null, currentLayer = 0;

function showError(message) { $("errorBox").textContent = message; $("errorBox").hidden = false; }
function setImage(id, source) { $(id).src = source; $(id).hidden = false; }
function updateButton() { $("runButton").disabled = !selectedFile || busy || !ready; }
function progress(value) {
  const bounded = Math.max(0, Math.min(100, Math.round(value)));
  $("percentage").innerHTML = `${bounded}<span>%</span>`;
  $("progressBar").style.width = `${bounded}%`;
}
function reset() {
  result = null; progress(0);
  ["resultImage", "compareOriginal", "scanBox", "activationImage", "resultsPanel", "comparisonPanel"].forEach(id => $(id).hidden = true);
  $("resultImage").style.clipPath = "none";
  $("resultEmpty").hidden = false; $("activationEmpty").hidden = false;
  $("maskButton").disabled = $("overlayButton").disabled = true;
  $("overlayButton").classList.add("active"); $("maskButton").classList.remove("active");
  $("activationTitle").textContent = "Chưa bắt đầu";
  $("activationShape").textContent = "ResNet-101 → PPM → classifier";
  $("stageText").textContent = "Sẵn sàng quan sát";
  $("stageDetail").textContent = "Model được nạp một lần vào RAM.";
  $("resultCaption").textContent = "Chưa có kết quả · không sử dụng GPU";
  $("elapsed").textContent = "SẴN SÀNG";
  document.querySelectorAll("[data-layer]").forEach(el => el.className = "");
}
function chooseFile(file) {
  if (busy || !file) return;
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) return showError("Hãy chọn ảnh JPG, PNG, WebP hoặc BMP.");
  if (file.size > 20 * 1024 * 1024) return showError("Ảnh vượt quá 20 MB. Hãy giảm kích thước trước.");
  selectedFile = file; $("errorBox").hidden = true; reset();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  setImage("inputImage", objectUrl); $("inputEmpty").hidden = true;
  $("fileName").textContent = file.name;
  const probe = new Image(); probe.onload = () => { $("imageInfo").textContent = `${probe.width} × ${probe.height} px · ${(file.size / 1024 / 1024).toFixed(1)} MB`; }; probe.src = objectUrl;
  updateButton();
}
$("fileInput").addEventListener("change", event => chooseFile(event.target.files[0]));
const dropZone = $("dropZone");
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); if (!busy) dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));

function positionBox() {
  const stage = $("resultStage"), img = $("resultImage"), box = $("scanBox");
  if (!box.dataset.coords || !img.naturalWidth) return;
  const coords = JSON.parse(box.dataset.coords);
  const scale = Math.min(stage.clientWidth / img.naturalWidth, stage.clientHeight / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  Object.assign(box.style, { left: `${(stage.clientWidth - w) / 2 + coords[0] * w}px`,
    top: `${(stage.clientHeight - h) / 2 + coords[1] * h}px`,
    width: `${(coords[2] - coords[0]) * w}px`, height: `${(coords[3] - coords[1]) * h}px` });
}
$("resultImage").addEventListener("load", positionBox); window.addEventListener("resize", positionBox);
function renderEvent(event) {
  if (event.type === "error") throw new Error(event.error);
  if (event.progress !== undefined) progress(event.progress);
  if (event.stage) $("stageText").textContent = event.stage;
  if (event.detail) $("stageDetail").textContent = event.detail;
  if (event.type === "input") {
    setImage("inputImage", event.image); setImage("compareOriginal", event.image); $("compareOriginal").hidden = true;
    $("imageInfo").textContent = `${event.width} × ${event.height} px${event.resized ? " · đã thu nhỏ để giới hạn RAM" : ""}`;
  } else if (event.type === "tile_start") {
    currentLayer = 0;
    if (event.tile === 1) setImage("resultImage", event.canvas);
    $("resultEmpty").hidden = true; $("scanBox").hidden = false;
    $("scanBox").dataset.coords = JSON.stringify(event.box); positionBox();
    document.querySelectorAll("[data-layer]").forEach(el => el.className = "");
    $("tileText").textContent = `CỬA SỔ ${event.tile} / ${event.total} · 473 × 473`;
    $("stageDetail").textContent = "Trích xuất đặc trưng → gom ngữ cảnh → dự đoán nhãn.";
    $("resultCaption").textContent = "Đang suy luận · vùng chưa xử lý giữ ảnh gốc";
  } else if (event.type === "activation") {
    setImage("activationImage", event.activation); $("activationEmpty").hidden = true;
    $("activationTitle").textContent = event.stage;
    $("activationShape").textContent = `Tensor ${event.shape.join(" × ")}`;
    document.querySelectorAll("[data-layer]").forEach((el, index) => { el.className = index < currentLayer ? "done" : index === currentLayer ? "active" : ""; });
    currentLayer++;
  } else if (event.type === "tile_done") {
    setImage("resultImage", event.preview);
    $("timingText").textContent = `Cửa sổ vừa xong: ${event.tile_seconds.toFixed(2)} giây`;
    $("resultCaption").textContent = `Kết quả tạm · đã ghép ${event.tile}/${event.total} cửa sổ`;
  } else if (event.type === "complete") {
    result = event; $("scanBox").hidden = true;
    setImage("resultImage", event.overlay); $("compareOriginal").hidden = false;
    $("resultCaption").textContent = `Hoàn thành · ${event.seconds.toFixed(2)} giây · ${event.tiles} cửa sổ`;
    $("stageDetail").textContent = "Đã ghép xác suất, khôi phục kích thước và chọn nhãn từng pixel.";
    $("timingText").textContent = `CPU · ${event.base_size} px · ${event.seconds.toFixed(2)} giây`;
    $("modelStatus").textContent = "MODEL ĐÃ NẠP · CPU";
    $("maskButton").disabled = $("overlayButton").disabled = false;
    $("resultsPanel").hidden = $("comparisonPanel").hidden = false;
    $("compareSlider").value = 100; $("compareValue").textContent = "100%";
    $("legend").replaceChildren();
    for (const item of event.legend) {
      const div = document.createElement("div"); div.className = "legend-item";
      const swatch = document.createElement("span"); swatch.className = "swatch"; swatch.style.background = item.color;
      const label = document.createElement("span"); label.className = "legend-name"; label.textContent = item.name;
      const percent = document.createElement("span"); percent.className = "legend-percent"; percent.textContent = `${item.percent.toFixed(1)}%`;
      div.append(swatch, label, percent); $("legend").append(div);
    }
    $("downloadOverlay").href = event.overlay; $("downloadMask").href = event.mask; $("downloadLabels").href = event.labels;
  }
}
$("compareSlider").addEventListener("input", event => {
  const value = Number(event.target.value);
  $("resultImage").style.clipPath = `inset(0 ${100 - value}% 0 0)`; $("compareValue").textContent = `${value}%`;
});
for (const [button, kind] of [["overlayButton", "overlay"], ["maskButton", "mask"]]) {
  $(button).addEventListener("click", () => {
    if (!result) return;
    setImage("resultImage", result[kind]);
    $("overlayButton").classList.toggle("active", kind === "overlay"); $("maskButton").classList.toggle("active", kind === "mask");
  });
}
$("runButton").addEventListener("click", async () => {
  if (busy || !selectedFile) return;
  reset(); busy = true; updateButton(); $("errorBox").hidden = true;
  $("fileInput").disabled = $("resolution").disabled = true; $("liveDot").classList.add("running");
  $("stageText").textContent = "Đang chuẩn bị suy luận"; startTime = performance.now();
  timer = setInterval(() => { $("elapsed").textContent = `${((performance.now() - startTime) / 1000).toFixed(1)} GIÂY`; }, 100);
  try {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Base-Size": $("resolution").value }, body: selectedFile });
    if (!response.ok) { const payload = await response.json(); throw new Error(payload.error || "Không gửi được ảnh."); }
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
        if (line.trim()) { renderEvent(JSON.parse(line)); await new Promise(resolve => requestAnimationFrame(resolve)); }
      }
      if (done) break;
    }
    if (!result) throw new Error("Kết nối kết thúc trước khi có kết quả. Xem cửa sổ terminal để biết chi tiết.");
  } catch (error) {
    showError(error.message); $("stageText").textContent = "Chưa hoàn thành";
    $("stageDetail").textContent = "Kiểm tra thông báo lỗi và thử lại.";
    $("resultCaption").textContent = "Xử lý chưa hoàn thành · kết quả đang thấy có thể chỉ là tạm";
  } finally {
    busy = false; clearInterval(timer); $("liveDot").classList.remove("running"); $("scanBox").hidden = true;
    $("fileInput").disabled = $("resolution").disabled = false; updateButton();
  }
});
fetch("/api/status").then(response => response.json()).then(status => {
  ready = status.checkpoint;
  $("modelStatus").textContent = ready ? "CHECKPOINT SẴN SÀNG · CPU" : "CHƯA CÓ CHECKPOINT";
  $("modelStatus").classList.toggle("missing", !ready);
  if (!ready) showError("Chưa tìm thấy checkpoints/b7_seed17_last.pt. Chạy DOWNLOAD_MODEL.cmd trước.");
  updateButton();
}).catch(() => { showError("Không kết nối được máy chủ local. Hãy mở giao diện từ RUN_DEMO.cmd."); });
