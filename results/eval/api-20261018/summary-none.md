# local ds41 vs official DeepSeek API — effort `none`

wall 7.3 min, 200 items, concurrency 4

| task | n | local acc | official acc | delta | local rt/item | official rt/item | local s/item | official s/item |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 100 | 98.0% (93–99) | 97.0% (92–99) | +1.0 pt | 0 | 0 | 6.4 | 2.7 |

gsm8k paired: local-only-correct b=1, official-only-correct c=0, McNemar p=1.0000

| mmlu | 100 | 88.0% (80–93) | 88.0% (80–93) | +0.0 pt | 0 | 0 | 4.9 | 3.1 |

mmlu paired: local-only-correct b=4, official-only-correct c=4, McNemar p=1.0000

