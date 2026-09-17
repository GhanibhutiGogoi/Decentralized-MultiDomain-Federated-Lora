# Replication chosen before the masked screen finishes

Replicate the proposed adaptive/domain partial-gradient variant and its fixed/
domain counterpart at seeds43/44, with the uniform16/sample driver control.
These are chosen because they answer the original adaptive-rank + domain-weight
question, not by selecting the best validation endpoint. Seed42 screening is
still running when this plan is written. Also train ordinary pooled LoRA at
seeds43/44 on the identical45k/5k split, giving a genuine conventional reference.

Use the unchanged screening code/configuration,30epochs, and final checkpoint.
Each seed runs on a separate remaining V100 of gpu003; no existing processes
are stopped. Do not open the official test set. Report sample SD and paired
accuracy differences; no noninferiority/equivalence margin has been specified.
Even good validation results cannot satisfy the original whole-adapter memory
cap because this variant retains fullrank16 model/Adam state.
