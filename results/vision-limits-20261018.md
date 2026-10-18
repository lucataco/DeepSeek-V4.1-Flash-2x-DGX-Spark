# Image limits

Measured 2026-10-18 with [`tests/image_limits.py`](../tests/image_limits.py) on the setup in
[RESULTS.md](RESULTS.md), thinking off, `max_tokens: 4`, synthetic PNGs, `TF_DS_VISION_MAX_IMAGES=64` and
`TF_MAX_BODY_MIB=128`.

| request | body | result | prompt tokens |
|---|---:|---|---:|
| text only | 0.0 MB | 200 | 13 |
| 1 × 32×32 | 0.0 MB | 200 | 198 (+185) |
| 1 × 1600×1200 | 0.2 MB | 200 | 1,015 (+1,002) |
| 8 × 32×32 | 0.0 MB | 200 | 1,493 = 13 + 8 × 185 |
| 64 × 32×32 | 0.3 MB | 200, 26.0 s | 11,853 = 13 + 64 × 185 |
| 65 × 32×32 | 0.3 MB | **400** `image input: a request may carry at most 64 images` | — |
| 64 × 1600×1200 | **78.1 MB** | 200, 275.7 s | 64,141 = 13 + 64 × 1,002 |

Per-image cost adds exactly and does not depend on the other images in the request: image 0 costs 185 tokens alone and
185 inside a 64-image request. Size alone decides it — below `min_pixels` (295,936) an image is scaled up, at or above
`max_image_tokens` (1,024) it is scaled down; both come from the model's `config.json`. These two shapes are the only
ones measured; the count after resizing to the patch grid depends on aspect ratio, so read 185 and 1,002 as "a 32x32
image" and "a 1600x1200 image", not as universal values. One image is capped at 20 MB (`DsVision.max_bytes`).

With the stock 32 MiB body cap, 64 images may average only 0.5 MB each, and a 78 MB body is refused before it reaches
the engine; `TF_MAX_BODY_MIB` raises the cap.

Separately recorded during the same run, not by the script (which reads only `usage`): 275.7 s end to end from a laptop
over WiFi, a server time to first token of 174 s (63 s of prefill plus about 110 s in rank 0's vision tower, roughly
1.7 s per image), and the rest dominated by the 78 MB upload. Prefix caching still works across image turns: turn 1
rendered 4,016 prompt tokens (2,048 cached), and a second turn adding one image rendered 5,042 with the same 2,048
cached.

On 64 synthetic images the reply was "I observed forty images". Counting 64 abstract PNGs is not something to rely on;
the token arithmetic above is the reproducible part.
