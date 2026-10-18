#!/usr/bin/env python3
"""200K-token prefill + decode measurement against the local ds41 server.

Cold run: corpus built from a random seed that is NEVER sent to /v1/tokenize,
so no prefix-cache entry can match it (tokenizing the exact text appears to warm
the KV window -- the first attempt measured a 97-block cache hit instead).
Warm run: resend the same prompt to show 2048-block prefix-cache reuse at 200K.
"""
import json, random, time, requests

BASE = "http://192.168.0.112:8888/v1"
TARGET = 200_000
MAXTOK = 256
LINE = "Record %06d: pipeline measured %d.4 ms on shard %d, retry %d, checksum %08x."


def tok(text):
    return requests.post(BASE + "/tokenize", json={"prompt": text}, timeout=300).json()["count"]


def corpus(n, seed):
    rnd = random.Random(seed)
    return "\n".join(LINE % (i, rnd.randrange(100, 999), rnd.randrange(32),
                             rnd.randrange(5), rnd.getrandbits(32)) for i in range(n))


def health():
    return requests.get(BASE + "/health", timeout=30).json()


def run(label, text):
    h0 = health()
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user",
                      "content": text + "\n\nSummarize the data above in about 200 words."}],
        "max_tokens": MAXTOK,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    r = requests.post(BASE + "/chat/completions", json=payload, stream=True, timeout=1800)
    if r.status_code != 200:
        print("%s HTTP %d %s" % (label, r.status_code, r.text[:300]))
        return None
    ttft = first = last = None
    deltas = 0
    usage = {}
    finish = None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        d = json.loads(data)
        ch = d.get("choices") or []
        if ch:
            delta = ch[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if ttft is None:
                    ttft, first = now - t0, now
                last = now
                deltas += 1
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
        if d.get("usage"):
            usage = d["usage"]
    total = time.time() - t0
    h1 = health()

    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    dec = (last - first) if (first and last) else 0
    dpre = h1["prefill_seconds_total"] - h0["prefill_seconds_total"]
    ddec = h1["decode_seconds_total"] - h0["decode_seconds_total"]
    fresh = pt - cached

    print("\n== %s ==" % label)
    print("prompt_tokens        %d  (cached %d, freshly prefilled %d)" % (pt, cached, fresh))
    print("completion_tokens    %d  finish=%s" % (ct, finish))
    print("TTFT                 %.2f s   -> %.0f tok/s over prompt" % (ttft, pt / ttft if ttft else 0))
    print("decode window        %.2f s   -> %.1f tok/s  (stream deltas %d)" % (dec, ct / dec if dec else 0, deltas))
    print("end-to-end           %.2f s" % total)
    print("server prefill       %.2f s   -> %.0f tok/s (fresh only: %.0f)" % (
        dpre, pt / dpre if dpre else 0, fresh / dpre if dpre else 0))
    print("server decode        %.2f s   -> %.1f tok/s" % (ddec, ct / ddec if ddec else 0))
    return dict(label=label, pt=pt, cached=cached, fresh=fresh, ct=ct, ttft=ttft,
                decode_s=dec, total=total, dpre=dpre, ddec=ddec, finish=finish)


if __name__ == "__main__":
    sample = corpus(300, 1)
    per = tok(sample) / 300.0
    n = int(TARGET / per)
    text = corpus(n, int(time.time()) % 10**9)      # never tokenized
    print("corpus: %d lines, %d chars, ~%.1f tok/line -> expect ~%d tokens" % (
        n, len(text), per, int(n * per)))
    run("cold (200K prompt, unique corpus)", text)
    run("warm (same prompt, prefix cache)", text)
