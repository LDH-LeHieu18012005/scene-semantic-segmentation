"""Check bytes against the exact checkpoint digest saved in the original notebook."""
import hashlib
from pathlib import Path
import sys

EXPECTED_SHA256 = "b7f060e54ec271ea41049daa18f82d607b2581b93d642884971d192650ac89cd"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "checkpoints" / "b7_seed17_last.pt"
with path.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
if digest != EXPECTED_SHA256:
    raise SystemExit(f"SHA256 mismatch: {digest}. Expected {EXPECTED_SHA256}")
print(f"PASS checkpoint SHA256 ({path.stat().st_size:,} bytes): {digest}")
