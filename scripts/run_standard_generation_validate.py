"""One controlled Standard FLUX generation against the local API."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
img = ROOT / "data" / "uploads" / "val_cotton.png"
if not img.exists():
    imgs = list((ROOT / "data" / "uploads").glob("*.png"))
    img = imgs[0]

boundary = "----fluxboundary"
parts: list[bytes] = []


def add_field(name: str, value: bytes | str, filename: str | None = None, ctype: str = "application/octet-stream"):
    parts.append(f"--{boundary}\r\n".encode())
    if filename:
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        parts.append(value if isinstance(value, bytes) else value.encode())
        parts.append(b"\r\n")
    else:
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")


add_field("fabric_image", img.read_bytes(), img.name, "image/png")
fields = {
    "garment_type": "dress",
    "fit": "slim",
    "style": "casual",
    "gender": "women",
    "season": "summer",
    "occasion": "casual",
    "fabric": "cotton",
    "material": "cotton",
    "texture": "smooth",
    "color": "match_fabric",
    "sleeve": "short",
    "neckline": "round",
    "generation_mode": "standard",
}
for k, v in fields.items():
    add_field(k, v)
parts.append(f"--{boundary}--\r\n".encode())
body = b"".join(parts)

req = urllib.request.Request("http://127.0.0.1:8000/api/v1/generate", data=body, method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
with urllib.request.urlopen(req, timeout=120) as resp:
    init = json.loads(resp.read().decode())
print("job", init)
job_id = init["job_id"]
t0 = time.time()
while True:
    with urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/status/{job_id}", timeout=60) as resp:
        st = json.loads(resp.read().decode())
    elapsed = int(time.time() - t0)
    print(
        f"[{elapsed}s] status={st.get('status')} stage={st.get('stage')} "
        f"progress={st.get('progress')} step={st.get('current_step')} "
        f"err_type={st.get('error_type')} failed_stage={st.get('failed_stage')}"
    )
    if st.get("status") in ("completed", "failed"):
        out = ROOT / "experiments" / "generation_results" / "day18_standard_validation_status.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(json.dumps(st, indent=2)[:3000])
        break
    time.sleep(15)
