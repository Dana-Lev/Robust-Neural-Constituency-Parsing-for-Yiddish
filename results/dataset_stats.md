| Split | Sentences | Tokens | Mean len | Max len | Mean depth | Malformed |
|---|---:|---:|---:|---:|---:|---:|
| train.txt | 15,394 | 171,065 | 11.1 | 165 | 6.8 | 0 |
| dev.txt | 855 | 9,323 | 10.9 | 71 | 6.7 | 0 |
| test.txt | 856 | 9,619 | 11.2 | 59 | 6.8 | 0 |
| **total** | **17,105** | **190,007** | | | | |

Distinct constituent labels: 226 after stripping co-indexation (402 raw). The scorer ignores co-indexation, so the first number is the one to quote.
Top 15 labels:
  NP-SBJ            19,565
  TOP               17,105
  NP                14,749
  IP-MAT            14,164
  PP                13,590
  NP-ACC             7,541
  IP-SUB             6,402
  ADVP               5,945
  CONJP              3,014
  NP-DTV             2,581
  WNP                1,847
  CP-ADV             1,634
  IP-INF             1,550
  NP-RFL             1,486
  IP-MAT-SPE         1,480
