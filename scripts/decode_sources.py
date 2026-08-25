#!/usr/bin/env python3
import base64
from pathlib import Path

def load_b64(path):
    return base64.b64decode("".join(Path(path).read_text().split()))

MAP = {
    "scripts/b64/src__models__predict_anemia.py.b64": "src/models/predict_anemia.py",
    "scripts/b64/src__models__predict_preeclampsia.py.b64": "src/models/predict_preeclampsia.py",
    "scripts/b64/src__models__predict_gdm.py.b64": "src/models/predict_gdm.py",
    "scripts/b64/src__models__predict_thyroid.py.b64": "src/models/predict_thyroid.py",
    "scripts/b64/src__integration__schemas.py.b64": "src/integration/schemas.py",
    "scripts/b64/src__integration__aggregator.py.b64": "src/integration/aggregator.py",
}
for src, dest in MAP.items():
    if not Path(src).exists():
        print("missing", src); continue
    data = load_b64(src)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(data)
    print("wrote", dest, len(data))

parts = sorted(Path("scripts/b64").glob("frontend_part*.b64"), key=lambda p: int(p.stem.replace("frontend_part", "")))
if parts:
    blob = "".join("".join(p.read_text().split()) for p in parts)
    data = base64.b64decode(blob)
    Path("frontend").mkdir(exist_ok=True)
    Path("frontend/app.py").write_bytes(data)
    print("wrote frontend/app.py", len(data))
print("Done decoding sources.")
