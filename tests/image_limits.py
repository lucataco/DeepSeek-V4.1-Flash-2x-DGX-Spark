"""Image limits on the DS-V4.1-Flash server: the per-request count cap, per-image token cost, body cap, cache reuse.

    python3 tests/image_limits.py [http://HEAD:8888] [model]

Above 8 images needs the engine patch in docker/patches/ (TF_DS_VISION_MAX_IMAGES), and 64 large images need
TF_MAX_BODY_MIB as well: their bodies total 78 MB, more than the stock 32 MiB.
"""
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4.1-flash"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def png(w, h, seed):
    rows = b"".join(b"\x00" + bytes(c for x in range(w) for c in ((x * 7 + y * 3 + seed * 41) % 256,
                                                                   (x + seed * 13) % 256, (y + seed * 29) % 256))
                    for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
          chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}


def chat(parts_or_msgs, max_tokens=4):
    msgs = parts_or_msgs if parts_or_msgs and "role" in parts_or_msgs[0] else \
        [{"role": "user", "content": parts_or_msgs + [{"type": "text", "text": "How many images? Reply with a number."}]}]
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "messages": msgs, **OFF}).encode()
    t = time.time()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", body,
                       {"Content-Type": "application/json"}), timeout=900).read())
        u = r["usage"]
        return 200, u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens"), \
            r["choices"][0]["message"]["content"], time.time() - t, len(body)
    except urllib.error.HTTPError as e:
        return e.code, None, None, json.loads(e.read() or b"{}").get("error", {}).get("message"), \
            time.time() - t, len(body)


base = chat([])[1]                                    # text only
small = [png(32, 32, i) for i in range(70)]           # below min_pixels: scaled up
large = [png(1600, 1200, 100 + i) for i in range(64)]  # above the 1024-token cap: scaled down
one_small = chat(small[:1])[1] - base
one_large = chat(large[:1])[1] - base
print(f"text-only prompt {base} tokens; one 32x32 image +{one_small}; one 1600x1200 image +{one_large}")

for n in (8, 9, 64, 65):
    code, p, _, msg, dt, size = chat(small[:n])
    extra = f"prompt {p} = text + {n} x {(p - base) / n:.1f}" if p else msg
    print(f"{n} small images: HTTP {code} ({dt:.1f}s, body {size/1e6:.1f} MB): {extra}")

# determinism: an image costs the same alone and among 63 others, so the sum adds up exactly
p64 = chat(small[:64])[1]
print(f"additivity: 64 small = text + 64 x {one_small}? {p64 == base + 64 * one_small} ({p64} vs {base + 64 * one_small})")

code, p, _, msg, dt, size = chat(large)
print(f"64 large images: HTTP {code} ({dt:.1f}s, body {size/1e6:.1f} MB): "
      + (f"prompt {p} = text + 64 x {(p - base) / 64:.1f}; reply {msg!r}" if p else msg))

# cache: same conversation, then the same plus one more image turn
turn1 = [{"role": "user", "content": large[:4] + [{"type": "text", "text": "Describe these briefly."}]}]
c1 = chat(turn1, 16)
turn2 = turn1 + [{"role": "assistant", "content": c1[3]},
                 {"role": "user", "content": large[4:5] + [{"type": "text", "text": "And this one?"}]}]
c2 = chat(turn2, 16)
print(f"cache: turn 1 prompt {c1[1]} (cached {c1[2]}); turn 2 adds an image: prompt {c2[1]}, cached {c2[2]}")
