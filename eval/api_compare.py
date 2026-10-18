#!/usr/bin/env python3
"""Paired eval: local ds41 (DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw, TensorFold TP2)
vs official DeepSeek API, on the tinyBenchmarks 100-item subsets, at MATCHED
reasoning effort and with a byte-identical prompt.

Prompts come from each dataset's `input_formatted` column, i.e. the canonical
lm-eval few-shot prompt ending in "Answer:" -- so both endpoints see the same
string and the result is comparable to published tinyBenchmarks numbers.

Usage:
  ./compare.py --effort high                 # both tasks, 100 items each
  ./compare.py --task gsm8k --limit 10       # smoke
  ./compare.py --effort low --task mmlu

Output: raw-<tag>.jsonl, summary-<tag>.md, paired-<tag>.csv
"""
import argparse, json, math, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = dict(url="http://192.168.0.112:8888/v1/chat/completions", model="deepseek-v4.1-flash", key=None)
OFFICIAL = dict(url="https://api.deepseek.com/v1/chat/completions", model="deepseek-flash",
                key=os.environ.get("DEEPSEEK_API_KEY"))

TASKS = {
    "gsm8k": dict(parquet="tinyGSM8k-test.parquet", max_tokens=2048),
    "mmlu":  dict(parquet="tinyMMLU-test.parquet",  max_tokens=1024),
}


# ---------------------------------------------------------------- grading
NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def norm_num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return round(float(s), 6)
    except ValueError:
        return None


def gold_gsm8k(answer):
    m = re.search(r"####\s*([-+]?[\d,.]+)", answer)
    return norm_num(m.group(1)) if m else None


def pred_gsm8k(text):
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        v = norm_num(boxed[-1])
        if v is not None:
            return v
    nums = NUM.findall(text)
    return norm_num(nums[-1]) if nums else None


def gold_mmlu(idx):
    return "ABCD"[int(idx)]


def pred_mmlu(text):
    m = re.search(r"Answer\s*:\s*\(?([A-D])\)?\b", text)
    if m:
        return m.group(1)
    letters = re.findall(r"\b([A-D])\b", text)
    return letters[0] if letters else None


def grade(task, text, gold):
    if task == "gsm8k":
        p = pred_gsm8k(text)
        return p is not None and p == gold, p
    p = pred_mmlu(text)
    return p == gold, p


# ---------------------------------------------------------------- transport
def call(cfg, prompt, effort, max_tokens, retries=3):
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": effort,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = "Bearer " + cfg["key"]
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(cfg["url"], json=payload, headers=headers, timeout=900)
            dt = time.time() - t0
            if r.status_code != 200:
                last = "HTTP %d %s" % (r.status_code, r.text[:200])
            else:
                d = r.json()
                msg = d["choices"][0]["message"]
                u = d.get("usage") or {}
                return dict(
                    ok=True,
                    content=msg.get("content") or "",
                    reasoning=msg.get("reasoning_content") or "",
                    finish=d["choices"][0].get("finish_reason"),
                    latency=round(dt, 2),
                    prompt_tokens=u.get("prompt_tokens"),
                    completion_tokens=u.get("completion_tokens"),
                    reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                )
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(2 * (attempt + 1))
    return dict(ok=False, error=last, latency=None)


# ---------------------------------------------------------------- stats
def binom_two_sided(b, c):
    """Exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="high", choices=["none", "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--task", default="both", choices=["gsm8k", "mmlu", "both"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4, help="local PARALLEL=4")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if not OFFICIAL["key"]:
        sys.exit("DEEPSEEK_API_KEY not set")

    tasks = ["gsm8k", "mmlu"] if args.task == "both" else [args.task]
    tag = args.tag or "%s-%s" % (args.task, args.effort)
    jobs = []

    for task in tasks:
        spec = TASKS[task]
        df = pd.read_parquet(os.path.join(HERE, spec["parquet"])).head(args.limit)
        for i, row in df.iterrows():
            gold = gold_gsm8k(row["answer"]) if task == "gsm8k" else gold_mmlu(row["answer"])
            jobs.append(dict(task=task, idx=int(i), prompt=row["input_formatted"],
                             gold=gold, max_tokens=spec["max_tokens"]))

    print("%d items x 2 endpoints, effort=%s, concurrency=%d" % (len(jobs), args.effort, args.concurrency))
    results = {}

    def run(job):
        out = {}
        for name, cfg in (("local", LOCAL), ("official", OFFICIAL)):
            r = call(cfg, job["prompt"], args.effort, job["max_tokens"])
            if r.get("ok"):
                good, pred = grade(job["task"], r["content"], job["gold"])
                r["correct"], r["pred"] = good, pred
            out[name] = r
        return job, out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for n, (job, out) in enumerate(ex.map(run, jobs), 1):
            key = "%s:%d" % (job["task"], job["idx"])
            results[key] = dict(job=job, **out)
            l, o = out["local"], out["official"]
            print("%3d/%d %-10s local=%s official=%s  (rt %s/%s)" % (
                n, len(jobs), key,
                "OK" if l.get("correct") else ("ERR" if not l.get("ok") else "x"),
                "OK" if o.get("correct") else ("ERR" if not o.get("ok") else "x"),
                l.get("reasoning_tokens"), o.get("reasoning_tokens")), flush=True)
    wall = time.time() - t0

    with open(os.path.join(HERE, "raw-%s.jsonl" % tag), "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"key": k, **v}, ensure_ascii=False) + "\n")

    # ------------------------------------------------ summary
    lines = ["# local ds41 vs official DeepSeek API — effort `%s`" % args.effort, "",
             "wall %.1f min, %d items, concurrency %d" % (wall / 60, len(jobs), args.concurrency), ""]
    lines.append("| task | n | local acc | official acc | delta | local rt/item | official rt/item | local s/item | official s/item |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    csv_rows = []
    for task in tasks:
        sel = [v for k, v in results.items() if v["job"]["task"] == task]
        ok = [v for v in sel if v["local"].get("ok") and v["official"].get("ok")]
        n = len(ok)
        lc = sum(v["local"]["correct"] for v in ok)
        oc = sum(v["official"]["correct"] for v in ok)
        b = sum(1 for v in ok if v["local"]["correct"] and not v["official"]["correct"])
        c = sum(1 for v in ok if not v["local"]["correct"] and v["official"]["correct"])
        lrt = [v["local"].get("reasoning_tokens") or 0 for v in ok]
        ort = [v["official"].get("reasoning_tokens") or 0 for v in ok]
        lsec = [v["local"].get("latency") or 0 for v in ok]
        osec = [v["official"].get("latency") or 0 for v in ok]
        lci, oci = wilson(lc, n), wilson(oc, n)
        lines.append("| %s | %d | %.1f%% (%.0f–%.0f) | %.1f%% (%.0f–%.0f) | %+.1f pt | %.0f | %.0f | %.1f | %.1f |" % (
            task, n, 100 * lc / n, 100 * lci[0], 100 * lci[1], 100 * oc / n, 100 * oci[0], 100 * oci[1],
            100 * (lc - oc) / n,
            sum(lrt) / n, sum(ort) / n, sum(lsec) / n, sum(osec) / n))
        lines.append("")
        lines.append("%s paired: local-only-correct b=%d, official-only-correct c=%d, McNemar p=%.4f" % (
            task, b, c, binom_two_sided(b, c)))
        lines.append("")
        for v in ok:
            csv_rows.append(dict(task=task, idx=v["job"]["idx"], gold=v["job"]["gold"],
                                 local_pred=v["local"]["pred"], official_pred=v["official"]["pred"],
                                 local_correct=int(v["local"]["correct"]),
                                 official_correct=int(v["official"]["correct"]),
                                 local_reasoning=v["local"].get("reasoning_tokens"),
                                 official_reasoning=v["official"].get("reasoning_tokens"),
                                 local_latency=v["local"].get("latency"),
                                 official_latency=v["official"].get("latency")))
        bad = [v for v in sel if not (v["local"].get("ok") and v["official"].get("ok"))]
        if bad:
            lines.append("errors: %d" % len(bad))
            for v in bad[:5]:
                lines.append("- %s local=%s official=%s" % (
                    v["job"]["idx"], v["local"].get("error", "ok"), v["official"].get("error", "ok")))
            lines.append("")

    pd.DataFrame(csv_rows).to_csv(os.path.join(HERE, "paired-%s.csv" % tag), index=False)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "summary-%s.md" % tag), "w") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
