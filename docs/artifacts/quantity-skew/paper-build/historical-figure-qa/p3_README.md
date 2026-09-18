# Historical P3 figure readability update

Generated on gpu003 from the frozen P3 completion report, context `b729ad24b6b0`. All nine arms, all 50 rounds, and all four endpoints are preserved. The 1,800 plotted mean/sample-SD pairs were independently checked against seeds 42–44. Accuracy stays on the original 0–1 axis. Payload is divided by 1e9 for the explicitly labeled billions axis; it is simulated fp32 float count, not measured throughput. The detached legend has its own reserved area below the axes. Original figures are unchanged.

Reproduce with `scripts/plot_historical_paper_readable.py` on gpu003 using the command in its docstring. The JSON sidecar contains all exact plotted arrays, and `p3_provenance.json` binds inputs, script, and outputs by SHA-256. `p3_actual_width_preview.png` renders the figure at 6.5 inches × 100 dpi for paper-width QA.
