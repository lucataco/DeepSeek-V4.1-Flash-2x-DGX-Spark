# local ds41 vs official DeepSeek API — effort `high`

wall 11.5 min, 200 items, concurrency 4

| task | n | local acc | official acc | delta | local rt/item | official rt/item | local s/item | official s/item |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 100 | 97.0% (92–99) | 97.0% (92–99) | +0.0 pt | 161 | 161 | 9.1 | 3.5 |

gsm8k paired: local-only-correct b=1, official-only-correct c=1, McNemar p=1.0000

| mmlu | 100 | 90.0% (83–94) | 90.0% (83–94) | +0.0 pt | 240 | 274 | 10.6 | 4.1 |

mmlu paired: local-only-correct b=1, official-only-correct c=1, McNemar p=1.0000

