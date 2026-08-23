| Condition | n | Returned | Coverage | Valid | Tok | LF | UF |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemini-3.5-flash zero-shot | 20 | 20 | 100.0 | 95.0 | 70.0 | **45.53** | 64.23 |
| gemini-3.5-flash 3-shot | 20 | 20 | 100.0 | 95.0 | 85.0 | **55.37** | 74.38 |
| gemini-3.5-flash-lite zero-shot | 100 | 100 | 100.0 | 86.0 | 44.0 | **24.74** | 42.46 |
| gemini-3.5-flash-lite 3-shot | 100 | 100 | 100.0 | 86.0 | 71.0 | **37.76** | 54.34 |
| gemini-3.6-flash zero-shot | 30 | 22 | 73.33 | 100.0 | 95.45 | **54.67** | 79.58 |

Labeled F1 restricted to answers that reproduced the given tokens (separating parsing ability from instruction-following):

| Condition | Token-faithful | LF (all) | LF (faithful only) |
|---|---:|---:|---:|
| gemini-3.5-flash zero-shot | 14/20 | 45.53 | **59.65** |
| gemini-3.5-flash 3-shot | 17/20 | 55.37 | **64.55** |
| gemini-3.5-flash-lite zero-shot | 44/100 | 24.74 | **46.41** |
| gemini-3.5-flash-lite 3-shot | 71/100 | 37.76 | **51.35** |
| gemini-3.6-flash zero-shot | 21/22 | 54.67 | **55.83** |

Labeled P/R and macro LF:

| Condition | LP | LR | Macro LF |
|---|---:|---:|---:|
| gemini-3.5-flash zero-shot | 44.8 | 46.28 | 42.08 |
| gemini-3.5-flash 3-shot | 55.37 | 55.37 | 58.43 |
| gemini-3.5-flash-lite zero-shot | 26.53 | 23.18 | 28.31 |
| gemini-3.5-flash-lite 3-shot | 42.86 | 33.74 | 40.35 |
| gemini-3.6-flash zero-shot | 51.97 | 57.66 | 51.55 |

> Coverage below 100% (API quota, not model failure) in: llm_zeroshot.json (73.33%). State this in the report.
