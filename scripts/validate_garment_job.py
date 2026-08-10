"""Submit a fabric→garment job and poll until done."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----BoundaryFabricVisionValidate"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    for name, (filename, content, content_type) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += content
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--garment-type", default="shirt")
    parser.add_argument("--fit", default="slim")
    parser.add_argument("--style", default="casual")
    parser.add_argument("--gender", default="women")
    parser.add_argument("--season", default="summer")
    parser.add_argument("--occasion", default="casual")
    parser.add_argument("--fabric", default="silk")
    parser.add_argument("--material", default="silk")
    parser.add_argument("--texture", default="smooth")
    parser.add_argument("--color", default="navy blue")
    parser.add_argument("--sleeve", default="puff sleeve")
    parser.add_argument("--neckline", default="sweetheart neck")
    parser.add_argument("--generation-mode", default="standard")
    parser.add_argument("--label", default="job")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    img_path = Path(args.image)
    raw = img_path.read_bytes()
    ctype = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    fields = {
        "garment_type": args.garment_type,
        "fit": args.fit,
        "style": args.style,
        "gender": args.gender,
        "season": args.season,
        "occasion": args.occasion,
        "fabric": args.fabric,
        "material": args.material,
        "texture": args.texture,
        "color": args.color,
        "sleeve": args.sleeve,
        "neckline": args.neckline,
        "generation_mode": args.generation_mode,
    }
    body, content_type = _multipart(fields, {"fabric_image": (img_path.name, raw, ctype)})
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/generate",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    created = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
    job_id = created["job_id"]
    print(f"created {args.label} job_id={job_id}", flush=True)

    out = Path("outputs") / f"val_{args.label}.json"
    started = time.time()
    while True:
        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:8000/api/v1/status/{job_id}", timeout=60
            ).read().decode()
        )
        elapsed = int(time.time() - started)
        print(
            f"{elapsed}s {status.get('status')} {status.get('progress')} {status.get('current_step')}",
            flush=True,
        )
        if status.get("status") in {"completed", "failed"}:
            out.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status, indent=2), flush=True)
            return 0 if status.get("status") == "completed" else 1
        if elapsed > args.timeout:
            print("TIMEOUT", flush=True)
            out.write_text(json.dumps(status, indent=2), encoding="utf-8")
            return 2
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
