# StableToken MPS Benchmarks

This directory records the local MPS benchmark runs produced by `benchmark_mps.py`.

Environment:
- `torch 2.7.1`
- Apple MPS backend
- `PYTORCH_ENABLE_MPS_FALLBACK=1`

Summary:

| Run | Tokenization median | Reconstruction median | Notes |
| --- | ---: | ---: | --- |
| `cpu_5s.json` | 0.226 s | 3.173 s | CPU baseline |
| `mps_5s_no_autocast.json` | 0.095 s | 1.501 s | Best stable MPS 5 s run |
| `mps_5s_flow5.json` | 0.090 s | 0.888 s | Fastest decode setting tested |
| `mps_30s.json` | 0.771 s | 22.115 s | Longer sequence, no autocast |
| `mps_30s_autocast.json` | 0.772 s | 20.669 s | Slightly better decode than no autocast |

Notes:
- The MPS path works, but first-load overhead is high, so the `load_s` number is not a good steady-state metric.
- In these runs, decoder autocast did not improve the short 5 s benchmark.
- If you want the fastest local decode path, `--flow_steps 5` was the best setting tested here.
