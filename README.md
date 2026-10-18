# DeepSeek-V4.1-Flash-1017 on two DGX Sparks

Serve [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on two NVIDIA DGX Spark /
GB10 nodes (we used ASUS Ascent GX10s) joined by one ConnectX-7 cable: tensor parallel over RoCE with
[TensorFold](https://github.com/bertholomus/TensorFold)'s DeepSeek-V4.1 engine, an EXL3 2.9 bpw pack of the 1017
weights, 1M-token context, 4 concurrent sessions, DSpark speculative decoding, tool calls and image input, behind an
OpenAI-compatible API.

This repo holds the scripts, the one fix the 1017 release needs, and the speed and quality measurements, so you can
reproduce them. It contains no weights.

| Decode, 1 stream (code / prose / structured) | 97 / 60 / 144 tok/s |
|---|---|
| Decode, 4 streams, aggregate | 118 tok/s |
| Decode after a 128K-token prompt | 71 tok/s |
| Prefill (8K / 32K / 128K) | ~1,650–2,000 / ~1,300 / ~1,300 tok/s |
| Quality vs DeepSeek's own API (seeded MMLU + GSM8K, thinking off and on) | same within noise; same answer on 96–98% of questions |

Full numbers, versions and method: [results/RESULTS.md](results/RESULTS.md). How we got here, including what did not
work: [docs/JOURNEY.md](docs/JOURNEY.md).

## What you need

- Two DGX Spark / GB10 nodes (128 GB unified memory each), DGX OS with Docker and the NVIDIA container toolkit,
  driver 580.x (measured on 580.178.04).
- One ConnectX-7 cable between them with an IPv4 on each end. Its two RoCE devices ("rails") are both used.
- Passwordless ssh from the head (rank 0) to the worker over that link, and the same absolute paths on both nodes.
- ~540 GB free on each node: 211 GB pack, 203 GB Engram tables, ~99 GB weight cache written on the first start, plus a
  ~25 GB image. (Check with `scripts/preflight.sh`.)
- The Hugging Face CLI (`hf`) on the head.
- Optional, for the quality comparison: an [OpenRouter](https://openrouter.ai) API key (~$0.25 per run).

## Quick start

All commands run on the head node from a checkout of this repo.

```bash
git clone https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark && cd DeepSeek-V4.1-Flash-2x-DGX-Spark
cp .env.example .env && $EDITOR .env      # worker ssh target, link IPs, RoCE devices + GID, paths

scripts/preflight.sh                      # read-only checks of both nodes; fix any FAIL
scripts/build-image.sh                    # TensorFold image (pinned commit), copied to the worker over CX7
scripts/download-weights.sh               # pack + 1017 Engram shards on both nodes, checksummed (~400 GB)
scripts/install-chat-template.sh          # the chat template 1017 does not ship (see below)
scripts/preflight.sh                      # everything should PASS now

scripts/start.sh                          # first start writes ~99 GB of weight cache per node; later ~2 min
python3 tests/smoke.py                    # arithmetic, recall, code, thinking, tool call
```

`scripts/start.sh status | logs [worker] | restart`, `scripts/stop.sh`.

### The one fix 1017 needs

DeepSeek released 1017 without a Jinja chat template (only `encoding/encoding.py`), and the engine renders every
prompt through one: without it the server loads fully and then exits with `KeyError: 'chat_template'`. 1017's encoder,
tokenizer and golden encodings are byte-identical to the original V4.1's, and DeepSeek's official V4.1 template
reproduces all five of 1017's golden encodings byte for byte. One more detail matters for images: TensorFold puts
DeepSeek's image placeholder token into the message text before rendering, and the official template rejects that
with `Message content contains image special token`. `install-chat-template.sh` fetches the official template at a
pinned, hash-checked revision and removes those three guard lines (TensorFold itself still rejects placeholders typed
by users). `python3 tests/check_chat_template.py` re-runs both checks: the five golden encodings, and the image case
through TensorFold's placeholder path.

## Using it

```bash
curl http://HEAD:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "max_tokens": 512,
  "messages": [{"role": "user", "content": "Write a haiku about RoCE."}],
  "reasoning_effort": "none"}'
```

- **Thinking:** `reasoning_effort` = `none` (off), `minimal`, `low`, `medium`, `high` (the default when on), `xhigh`, `max`;
  or `"chat_template_kwargs": {"enable_thinking": false}`. From `low` to `max` the reasoning budget roughly quadruples.
- **Tools:** the standard OpenAI `tools` field; calls come back as `tool_calls` with `finish_reason: "tool_calls"`.
- **Images:** OpenAI-style `image_url` content parts; 8 per request and a 32 MiB body by default, both
  configurable — see [Images](#images).
- **Pi:** [`clients/pi/models.json`](clients/pi/models.json) is a provider entry for the [Pi](https://pi.dev) coding
  agent, mapping Pi's thinking levels onto the server's (`pi --model deepseek-v4.1-flash:off` etc.).
- The server is open to your network with no API key; see [TROUBLESHOOTING](docs/TROUBLESHOOTING.md#serving).

### Images

The engine hardcodes 8 images per request and a 32 MiB request body. `docker/patches/` makes both configurable and
leaves the defaults alone:

```bash
EXTRA_ENV="-e TF_DS_VISION_MAX_IMAGES=64 -e TF_MAX_BODY_MIB=128"    # in .env
```

The body cap has to rise with the image count: 64 synthetic 1600x1200 PNGs are 78 MB, photos that size are larger,
JPEGs smaller. Vision tokens depend only on each image's own size, not on how many images the request carries: a
32x32 image is scaled up to `min_pixels` (295,936) and costs 185 tokens, a 1600x1200 one is scaled down to the
`max_image_tokens` (1,024) cap and costs 1,002, and 64 of those are 64,141 prompt tokens. One image is capped at
20 MB. Numbers and script:
[results/vision-limits-20261018.md](results/vision-limits-20261018.md).

## Reproducing the measurements

```bash
bench/run-bench.sh                        # speed: the engine author's kit_bench.py (fetched, hash-checked)
pip install huggingface_hub pandas pyarrow jinja2
python3 eval/prepare-data.py              # MMLU + GSM8K test splits -> eval/data/
export OPENROUTER_API_KEY=...
eval/run-eval.sh                          # local vs OpenRouter (DeepSeek, DeepInfra), paired comparison
```

New runs land in `results/bench/run-<date>/` and `results/eval/run-<date>/`.

## Layout

```
.env.example                 settings for every script (copy to .env)
docker/Dockerfile            TensorFold DeepSeek-V4.1 TP2 engine on nvcr.io/nvidia/pytorch:26.07-py3, pinned commit
scripts/preflight.sh         read-only checks of both nodes
scripts/build-image.sh       build on the head, copy to the worker
scripts/download-weights.sh  pinned pack + 1017 Engram shards, SHA-256 checked, mirrored over CX7
scripts/install-chat-template.sh  the official V4.1 template into the pack, both nodes
scripts/start.sh, stop.sh    start / stop / status / logs for both ranks
tests/smoke.py               end-to-end correctness smoke
tests/check_chat_template.py template vs 1017's golden encodings
bench/run-bench.sh           speed suite
eval/                        quality_eval.py (modified from TensorFold), compare.py, prepare-data.py, run-eval.sh
results/                     our raw runs and RESULTS.md
clients/pi/models.json       Pi provider config
docs/JOURNEY.md              the path and the dead ends
docs/TROUBLESHOOTING.md      every problem we hit, and open questions
```

## Credits

The engine, its benchmark client and the eval script are [bertholomus](https://github.com/bertholomus)'s
TensorFold work (built on [ashhart/TensorFold](https://github.com/ashhart/TensorFold)). The 1017 EXL3 pack is by
[LibertAIDAI](https://huggingface.co/LibertAIDAI), in the layout [Mia-AiLab](https://huggingface.co/Mia-AiLab)
established; EXL3 is [turboderp](https://github.com/turboderp-org/exllamav3)'s. [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41)'s
recipe and findings shaped how we approached GB10. The model, its encoder and the chat template are
[DeepSeek](https://huggingface.co/deepseek-ai)'s (MIT).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party terms.
