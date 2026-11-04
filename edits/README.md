# Delta-Dateien

Jede Datei ist ein Dict `Parametername → Delta`. Sie enthält keine Modellgewichte und funktioniert nur zusammen mit dem Originalmodell.

**Dies sind Modifikationen von Qwen2.5-0.5B-Instruct** (Qwen Team, Alibaba Cloud, Apache License 2.0, Commit `7ae557604adf67be50417f59c2c2f167def9a775`). Sie wurden 2026 von diesem Projekt erstellt und stammen nicht vom Qwen-Team. Wer ein damit verändertes Modell weitergibt, muss nach Abschnitt 4 der Apache License 2.0 die Lizenz beilegen und die Änderung kenntlich machen.

| Datei | s | Rolle | SHA-256 |
| --- | --- | --- | --- |
| `e1.safetensors` | 0,4 | primärer Kandidat | `d26c5d85721e762f25007536c6e721d3f1751f93f6d36f4f227aeb1d0c5f71c1` |
| `e2.safetensors` | 0,5 | stärker | `31f19cc06896103b40e62a928366ae0d16c64ab22f5ce65b9127b43207831e1f` |
| `e3.safetensors` | 0,3 | schonender | `2b0576b9e4eef7a7749b92981126ded6c6550f5424b2133bf23aa82e3188be7a` |

Jede Datei ändert 10 Matrizen (`k_proj` der Layer 9 bis 18) mit 1.146.880 Gewichten, das sind 0,232 % des Modells. Embeddings und LM-Head bleiben unverändert. Die Formel steht im [Bericht](../report/report.md).

Die Baseline-Deltas `b1` und `b2` sind mit je 330 MB nicht enthalten. `python -m baselines.run_b1` erzeugt sie neu.

Die Deltas sind nur für den genannten Commit gültig. Kanonisch sind die Gewichte Basis + Delta in float32 auf der CPU; eine erneute Anwendung ergibt bit-identische Gewichte (`python -m edits.apply_edit edits/e1.safetensors`).
