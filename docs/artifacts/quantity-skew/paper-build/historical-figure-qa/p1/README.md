# Historical P1 figure readability replacement

Generated on gpu003 by `scripts/plot_historical_p1_paper.py` from the exact five
rows in `docs/artifacts/p1-adaptive-rank/federated_lora_summary.csv` and its
original manifest. The source CSV and manifest hashes are in
`p1-provenance.json`; `.data.json` preserves every plotted endpoint.

The new figure shows final accuracy at round 5, seed 42. It is the original
controller, not the revised controller or the new SST-2 campaign. Fixed rank 32
exceeds all declared capacity ceilings. One seed supplies no uncertainty
interval. All five final differences and recorded rank assignments are checked
by the plot script on gpu003. No experiment was rerun.

The old five-panel round-curve PNG remains in the historical artifact folder.
Intermediate round-accuracy arrays are not in the retained CSV, so the new
figure does not infer or reconstruct them from pixels. The paper caption now
explicitly identifies the plotted final endpoints.
