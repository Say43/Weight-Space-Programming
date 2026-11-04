# Delta files

Each file is a dict `parameter name → delta`. It contains no model weights and works only together with the original model.

**These files are modifications of Qwen2.5-0.5B-Instruct** (Qwen Team, Alibaba Cloud, Apache License 2.0, commit `7ae557604adf67be50417f59c2c2f167def9a775`). They were created by this project in 2026 and do not come from the Qwen team. Anyone who distributes a model changed with them must, under section 4 of the Apache License 2.0, include the license and mark the change.

| file | s | role | SHA-256 |
| --- | --- | --- | --- |
| `e1.safetensors` | 0.4 | primary candidate | `d26c5d85721e762f25007536c6e721d3f1751f93f6d36f4f227aeb1d0c5f71c1` |
| `e2.safetensors` | 0.5 | stronger | `31f19cc06896103b40e62a928366ae0d16c64ab22f5ce65b9127b43207831e1f` |
| `e3.safetensors` | 0.3 | gentler | `2b0576b9e4eef7a7749b92981126ded6c6550f5424b2133bf23aa82e3188be7a` |

Each file changes 10 matrices (`k_proj` of layers 9 to 18) with 1 146 880 weights, which is 0.232 % of the model. The embeddings and the LM head stay unchanged. The formula is in the [report](../report/report.md).

The baseline deltas `b1` and `b2` are not included because they take 330 MB each. `python -m baselines.run_b1` regenerates them.

The deltas are valid for the commit above only. The canonical weights are base + delta in float32 on the CPU; re-applying gives bit-identical weights (`python -m edits.apply_edit edits/e1.safetensors`).
