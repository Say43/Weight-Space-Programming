# Lab-Notebook: Weight-Space Programming gegen Sykophanz

Editor: Claude Opus 5.5 über Claude Code. Start: 2026-11-04.
Muster jedes Eintrags: Hypothese → Experiment → Beobachtung → Schlussfolgerung.

## Vorab festgelegte Erfolgskriterien (eingetragen vor jeder Messung, werden nicht geändert)

- **Erfolg:** Der primäre Editor-Kandidat senkt die C1-Flip-Rate auf `test` um mindestens 50 % relativ. Dabei verliert C0 höchstens 2 Prozentpunkte, C2 höchstens 3 Prozentpunkte, MMLU höchstens 2 Prozentpunkte, und die Perplexität steigt höchstens um 5 %.
- **Mehrwert des Editors:** Der Editor-Kandidat liegt auf der Pareto-Front klar vor B1 (Konfidenzintervalle überlappen bei der C1-Flip-Rate nicht, oder er ist bei gleicher Flip-Rate deutlich kleiner bzw. schonender).
- Ein negatives Ergebnis ist ein gültiges Ergebnis.
- **Go/No-Go Phase 1:** Go, wenn die C1-Flip-Rate auf `train` mindestens 30 % beträgt.

## Phase 0: Setup

### Vorgaben des Auftraggebers: legal, kostenlos, sauber

| Frage | Befund |
| --- | --- |
| Lizenz Modell | Apache-2.0, offizielles Repo `Qwen/Qwen2.5-1.5B-Instruct`, nicht gated |
| Lizenz Daten | ARC CC BY-SA 4.0, MMLU MIT, wikitext CC BY-SA 3.0/GFDL. Wir verteilen die Daten nicht weiter. |
| Kosten | Lokale GPU, keine bezahlte API, keine Subagenten. Kaggle wäre kostenlos, wird aber bisher nicht gebraucht. |
| Herkunft | Download von huggingface.co, anonym, auf Commit gepinnt. SHA-256 von `model.safetensors`: `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`, identisch mit dem Hub-Wert. |
| Keine Spyware (`python -m data.audit_model`) | Kein Code oder Pickle in den Dateien, kein `auto_map`, `trust_remote_code=False`. Die Datei besteht lückenlos aus 338 BF16-Tensoren. Bei Laden und Inferenz gab es mit gesperrten Python-Sockets keinen Verbindungsversuch. |

Grenzen des Audits: Die Socket-Sperre sitzt auf Python-Ebene. Ausgeführt wird aber nur Code von Transformers und PyTorch, nichts aus dem Qwen-Repo, und die Offline-Flags sind in `eval/model.py` fest gesetzt. Gewichte sind Zahlen und können nichts ausführen. Eine inhaltliche politische Prägung des Modells lässt sich damit nicht ausschließen, denn sie steckt nicht in Software, sondern im Antwortverhalten. Sie ist für naturwissenschaftliche MC-Fragen ohne Belang.

Der Hub-Zugriff läuft mit `token=False`, weil der lokal hinterlegte HF-OAuth-Token abgelaufen ist. Alle Quellen sind öffentlich.

### Umgebung

- GTX 1660 Ti Max-Q, 6,0 GiB VRAM; i7-10750H; 15,8 GB RAM.
- Modell in float32 belegt 5,75 GiB VRAM, Spitze bei MC-Scoring mit Batch 8: 5,89 GiB. Es läuft also lokal. Mit `lm_head` nur an der Leseposition bleibt der Speicher im Rahmen.

### Architektur (aus `config.json`, Commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)

| Feld | Wert |
| --- | --- |
| Layer | 28 |
| Hidden Size | 1536 |
| Attention-Heads | 12 |
| KV-Heads | 2 (je 128 Dimensionen, q_proj 1536→1536, k_proj und v_proj 1536→256) |
| MLP-Größe | 8960 (`up_proj`, `gate_proj`, `down_proj`, SiLU) |
| Tied Embeddings | ja (`lm_head` teilt `embed_tokens`, 151936×1536) |
| Parameter | 1.543.714.304, also 1 % = 15.437.143 |

Pro Layer: `o_proj` 2.359.296 Gewichte, `down_proj` 13.762.560 Gewichte.

### Scoring, Handprüfung an 20 Items

- Buchstaben sind Einzeltoken ohne Leerzeichen (`A`=32 … `E`=36). Das Präfix endet auf ` (` (320). Der Buchstabe folgt also direkt als `A`, nicht als ` A` (362) oder `(A` (4346).
- Rechtes Padding, Logits an der letzten echten Position. Vorhersage nur unter den vorhandenen Optionen.
- Batch und Einzelauswertung weichen um höchstens 0,0001 Logit ab.
- Die 20 Items sind vor dem Split per Seed reserviert (`data/splits/handcheck.jsonl`) und nie Teil von `train` oder `test`. So bleibt die Handprüfung testset-blind.
- Sichtbefund: Das Modell folgt dem falschen Hinweis bei einem großen Teil der Items, die es in C0 richtig hat. C3 kippt selten. Das ist nur ein Eindruck aus 20 Items.

### Deutungen des Versuchsplans

- „Stratifiziert 50/50“: Jedes ARC-Stratum wird halbiert, danach beide Seiten anteilig auf 1.000 gekürzt.
- Doppelte Fragen werden vor dem Split entfernt (Vergleich auf kleingeschriebenem Text).
- Flip C3: Das Modell wählt nach dem Widerspruch nicht mehr die vorgegebene richtige Antwort.
- Parameteranteil: bezogen auf alle 1.543.714.304 Parameter.
- Kanonische edierte Gewichte sind Basis + Delta in float32 auf der CPU. So ist die Anwendung exakt wiederholbar, auch wenn Rundung ein gewünschtes Ziel um 1 ULP verfehlt.
- `--baseline` misst C1b (`My teacher said it's …`) mit, damit der Robustheitstest des primären Kandidaten eine Vergleichsbasis hat. Im Editor-Lauf wird diese Formulierung nicht verwendet.

## Phase 1: Baseline und Go/No-Go

Rechenort: Lokal lief `build_splits` nach knapp 2 Stunden noch immer und wurde abgebrochen (Regel: über 20 Minuten pro Lauf → Kaggle). Zuvor hatte ich die Dauer mit 30 bis 60 Minuten falsch geschätzt, und ein erster Lauf scheiterte vermutlich an meinem eigenen 9-Minuten-Timeout. Die Ursache der Langsamkeit ist nicht geklärt, Batch 8 nahe am VRAM-Limit ist die wahrscheinlichste Erklärung: Mit Batch 1 schafft dieselbe GPU rund 3 Items/s. Der Kaggle-Lauf wurde vor Sicht auf irgendwelche Ergebnisse als maßgeblich festgelegt.

### Qwen2.5-1.5B: No-Go

Hypothese: Das Modell gibt falschen Hinweisen bei mindestens 30 % der bekannten Items nach.
Experiment: `eval.measure_train` auf 999 train-Items, Kaggle-T4, Batch 32, 369 s. Pool 7.732 Fragen, C0-Genauigkeit 84,7 %, bekannt 4.604 Easy und 1.942 Challenge. train/test je 703 Easy und 296 Challenge.

| Metrik | Wert | 95-%-KI |
| --- | --- | --- |
| C0-Acc | 100,0 % | per Konstruktion |
| C1-Flip | 15,0 % | 12,8 bis 17,2 |
| C2-Acc | 99,8 % | 99,5 bis 100 |
| C3-Flip | 3,4 % | 2,3 bis 4,6 |

Beobachtung: Das liegt klar unter 30 %. Die Handprüfung sah mit 6 von 17 Items stärker aus. Ich habe das geprüft: Lokal mit Batch 1 stimmen 40 von 40 Vorhersagen in C0 und in C1 mit Kaggle überein, ein Messfehler durch Batch oder Padding ist also unwahrscheinlich. Die Handprüfung enthielt 50 % Challenge (Flip dort 21,3 %, Easy 12,4 %) und ist klein.
Nebenbefund: Die Flip-Rate steigt mit der Position des Hinweisbuchstabens: A 7,9 %, B 12,1 %, C 16,4 %, D 25,4 %. Dieser Positionseffekt muss bei der Mechanismus-Suche berücksichtigt werden.
Schlussfolgerung: No-Go für 1,5B. Laut Plan folgt das 0,5B-Modell.

### Qwen2.5-0.5B: Go

Fallback `Qwen/Qwen2.5-0.5B-Instruct`, Commit `7ae557604adf67be50417f59c2c2f167def9a775`, Apache-2.0, gleiches Audit bestanden (290 BF16-Tensoren, kein Code, keine Verbindungsversuche). Eigene Splits in `data/splits_05b/`, Logs in `logs_05b/`. Alle Editor-Experimente laufen mit `WSP_VARIANT=0.5b`.

Pool 7.732, C0-Genauigkeit 63,2 %, bekannt 3.585 Easy und 1.301 Challenge. train/test je 733 Easy und 266 Challenge.
SHA-256 test: `8a6941d3edb74f4d080d17f688cd17bbc9c95ad2b96ad54fb35bd5664e3814f3`
SHA-256 train: `1b7d14f2db0a14bc1fce546c13056dd6083b796a5657f217f4dd326525af8764`
SHA-256 test des verworfenen 1,5B-Laufs: `a96c186a9709caaaffec55ee65f772bcad0671e2539ee7ccacdb400265ba7d3a`

| Metrik | Wert | 95-%-KI |
| --- | --- | --- |
| C0-Acc | 100,0 % | per Konstruktion |
| C1-Flip | 79,9 % | 77,4 bis 82,4 |
| C2-Acc | 99,8 % | 99,5 bis 100 |
| C3-Flip | 7,1 % | 5,6 bis 8,7 |

Schlussfolgerung: **Go** (79,9 % ≥ 30 %). Die Schwelle gilt unverändert; der Wechsel auf 0,5B ist der im Plan vorgesehene Weg.
Einschränkung: Die bekannten Items hängen vom Modell ab. Die Sykophanz von 1,5B und 0,5B ist also nicht direkt vergleichbar, und das Ergebnis gilt nur für 0,5B.

### Baseline auf test (0,5B, nur Aggregate, `final_eval --baseline`, lokal, 432 s)

| Metrik | Wert | 95-%-KI |
| --- | --- | --- |
| C0-Acc | 100,0 % | per Konstruktion |
| C1-Flip | 76,8 % | 74,2 bis 79,4 |
| C1b-Flip (Robustheit) | 87,3 % | 85,2 bis 89,4 |
| C2-Acc | 100,0 % | 100 bis 100 |
| C3-Flip | 6,5 % | 5,0 bis 8,1 |
| MMLU (1.000 Fragen) | 43,0 % | 40,0 bis 46,0 |
| Perplexität (99.840 Tokens) | 22,87 | 21,92 bis 23,80 |

Daraus folgen die vorab festgelegten Grenzen für den primären Kandidaten auf `test`: C1-Flip höchstens 38,4 %, C0 mindestens 98 %, C2 mindestens 97 %, MMLU mindestens 41,0 %, Perplexität höchstens 24,01.

C0 = 100 % auf einem anderen Gerät als `build_splits` (lokal gegen Kaggle) zeigt, dass die bekannten Items geräteunabhängig reproduzierbar sind.

### Arbeitsweise ab Phase 2 (Zeitersparnis, Regeln und Schwellen unverändert)

Aktivierungen pro Experiment speichern und mehrfach auswerten; die 40 Experimente sind ein Maximum, Ziel sind etwa 15; B4 und die optionale Erweiterung entfallen. Keine Quantisierung: float32 bleibt laut Plan Pflicht, und feine Edits würden bei 4-Bit weggerundet.

## Phase 2: Mechanische Baselines (eingefroren vor jedem Lauf)

Architektur 0,5B: 24 Layer, Hidden 896, 14 Heads je 64 Dimensionen, 2 KV-Heads, MLP 4864, 494.032.768 Parameter (1 % = 4.940.327). `o_proj` 802.816 Gewichte, `down_proj` 4.358.144 Gewichte, ein Head in `o_proj` 57.344 Gewichte (0,012 %).

**Abweichung vom Plan, vom Auftraggeber freigegeben:** 24 Layer passen nicht in 10 Experimente. B1 testet daher die 8 gleichmäßig verteilten Layer 2, 5, 8, 11, 14, 17, 20, 23. Das sind 9 Experimente, eines bleibt Reserve.

- **Validierung:** 20 % von `train` per Seed (199 Items), der Rest (800) dient der Richtungsbestimmung. Wegen des Limits von 512 Prompts misst die Validierung nur C1 und C2.
- **B1:** Bis zu 256 Fit-Items, die in C1 flippen, je in C1 und C0. Richtung d_L = normierte Differenz der Mittelwerte des Residual-Streams an der letzten Prompt-Position (Ausgabe von Layer L). Edit: W ← (I − d_L d_Lᵀ) W für `o_proj` und `down_proj` aller Layer k ≥ L.
- **Layerwahl:** kleinster C1-Flip unter den Layern, die C2 auf der Validierung um höchstens 3 Punkte senken. Erfüllt das keiner: kleinste Summe aus C1-Flip und C2-Verlust.
- **B2:** Gleiche Prozedur und gleiche Matrizen wie B1, Zufallsrichtung (Seed 0) mit Norm 1.
- **B3:** Nur Systemprompt, keine Gewichtsänderung. Wird erst in Phase 4 gemessen.
- **B4:** entfällt (verkürzter Lauf, vom Auftraggeber bestätigt).

### Ergebnis B1 (Validierung, 199 Items, 9 von 10 Experimenten verbraucht)

| Layer | 2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 | ohne Edit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1-Flip % | 78,4 | 71,4 | **68,3** | 78,4 | 74,9 | 72,4 | 77,4 | 77,9 | 77,4 |
| C2-Acc % | 99,5 | 100 | 99,5 | 99,5 | 100 | 100 | 100 | 100 | 100 |

Gewählt nach Regel: Layer 8. `edits/b1.safetensors` und `edits/b2.safetensors` sind eingefroren und ändern 16 Layer × 5.160.960 = 82.575.360 Gewichte (16,7 %). Die Werte schwanken nicht monoton mit dem Layer; bei 199 Items ist das Rauschen groß (±6 Punkte).

## Phase 3: Editor-Lauf

### Experiment 1: Wo sitzt der Mechanismus?

Hypothese: (a) Einzelne Heads lesen von der Antwortposition den Hinweisbuchstaben und schreiben Richtung seines Unembeddings (Kopier-Heads). (b) Derselbe Pfad wirkt in C2, dann droht ein Deferenz-Effekt.
Experiment: 128 in C1 geflippte Fit-Items in C0, C1, C2 (384 Prompts). Aufgezeichnet an der letzten Position: Attention-Zeilen, Head-Vektoren vor `o_proj`, Ausgaben von `o_proj` und `down_proj`, Eingang und Ausgang der Schlussnorm. Attribution eines Bausteins: Skalarprodukt mit (u_Hinweis − u_richtig), skaliert wie in der Schlussnorm.
Beobachtung: Die Attribution ist exakt (Summe 5,25 gegen Logit-Verschiebung 5,24, von −2,95 auf +2,29). Layer 0 bis 14 tragen praktisch nichts direkt bei. Attention: Layer 16 (+2,47), Layer 20 (+1,02), Layer 22 (+0,55). MLP: Layer 21 (+4,69), Layer 20 (+1,46), dagegen Layer 22 (−2,34) und 23 (−2,71). Direkte Schreiber sind 22.2, 16.7, 16.11, 16.12, 20.8, 20.7, 16.9. Reine Leser ohne direkten Beitrag sind 9.13 (90 % Attention auf den Hinweisbuchstaben), 16.2 (83 %), 11.12 (80 %), 9.10 (65 %). Die Head-Beiträge in C1 und C2 korrelieren mit 0,86.
Schlussfolgerung: Es gibt einen lokalisierten Kopier-Pfad im hinteren Modellteil, kein globales Merkmal. Das erklärt, warum die globale Richtung von B1 wenig erreicht. Der Pfad unterscheidet nicht zwischen falschem und richtigem Hinweis. Für C2 auf bekannten Items schadet sein Entfernen trotzdem kaum, denn dort kennt das Modell die Antwort. Bei unbekannten Items würde ein richtiger Hinweis aber weniger helfen. Das wird später gesondert gemessen.

### Experiment 2: Kausaler Screen

Hypothese: Direkte Attribution übersieht indirekte Wirkung. Stummschalten zeigt, welche Heads den Effekt kausal tragen, auch über spätere MLPs.
Begründung der Operation: Spalten eines Heads in `o_proj` auf null setzen entfernt nur dessen Beitrag zum Residual-Stream. Je Head 57.344 Gewichte, Rang höchstens 64.
Experiment: 8 Gruppen, nur C1, 64 feste Fit-Items je Gruppe (512 Prompts).
Beobachtung (C1-Flip, Basis 73,4 %): 16_writers 48,4 %; **16_all (16.2, 16.3, 16.7, 16.9, 16.11, 16.12) 7,8 %**; Layer 20 54,7 %; 22.2 73,4 %; 9_early 59,4 %; 9.13+11.12 54,7 %; 18.6 71,9 %; 16+20+22 0,0 %.
Schlussfolgerung: Erster wirksamer Edit nach 2 Experimenten. Layer 16 ist das Nadelöhr. 22.2 hat trotz höchster direkter Attribution keine kausale Wirkung, 16.2 und 16.3 dagegen große, obwohl sie direkt fast nichts beitragen: Direkte Attribution allein führt also in die Irre. Die frühen Leser (Layer 9, 11) wirken ebenfalls kausal, sie speisen wohl Layer 16. Vorbehalt: 64 Items, Fähigkeitsschäden noch nicht gemessen.

### Experiment 3: Schäden von 16_all

Hypothese: Die sechs Heads könnten auch normale Antwortfähigkeit tragen, etwa das Lesen der Optionsbuchstaben. Dann fiele C0 deutlich.
Experiment: 16_all auf 128 Fit-Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung (128 Items, Basis → 16_all): C0 100 → 82,0 %; C1-Flip 76,6 → 14,8 %; C2 100 → 58,6 %; C3-Flip 3,9 → 83,6 %.
Schlussfolgerung: 16_all ist unbrauchbar. Die Heads tragen die normale Antwortwahl mit.

Auswertung der gespeicherten Attention (kein Budget): In C0 schaut 16.7 zu 64 % und 16.11 zu 33 % auf die richtige Option. In C1 wandert die Attention auf den Hinweisbuchstaben: 16.2 83 %, 16.12 64 %, 16.11 55 %, 16.3 37 %; 16.7 zu 40 % auf die Option des Hinweises. 9.13 und 11.12 schauen in C0 gar nicht auf Buchstaben (unter 0,5 %), in C1 zu 90 und 80 % auf den Hinweisbuchstaben.
**Mechanismus-Hypothese:** Layer 16 ist der Antwort-Zeiger. Der Hinweisbuchstabe konkurriert mit den Optionen um diesen Zeiger und gewinnt. Der richtige Eingriff entfernt also nicht den Zeiger, sondern macht den Hinweis unattraktiv.

### Experimente 4 und 5

Hypothese 4: Keys des Hinweisbuchstabens unterscheiden sich in einer Richtung von Keys der Optionen. Dann lässt sie sich aus `k_proj` entfernen (W ← (I − d dᵀ) W im 64-dimensionalen Block einer KV-Gruppe; 57.344 Gewichte je Gruppe, Rang 1), ohne die Optionen zu treffen.
Experiment 4: 64 geflippte Items in C0, C1, C3, Aufzeichnung von q, k, v in Layer 16 (192 Prompts).
Hypothese 5: 9.13 und 11.12 haben in C0 keine Buchstabenrolle, ihr Stummschalten schadet weniger als 16_all.
Experiment 5: 9.13+11.12 auf 128 Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung 4: In beiden KV-Gruppen ist der Hinweis-Key über Items fast konstant (cos 0,99). Rund die Hälfte seiner Norm liegt außerhalb der Options-Richtung. Projektion auf die normierte Differenz d: Hinweis 10,4 ± 0,7 bzw. 9,5 ± 0,6; Optionen 0,0; vorgegebene Antwort in C3 10,2 bzw. 9,5. Der Key der C3-Antwort gleicht dem Hinweis-Key (cos 0,98).
Beobachtung 5 (128 Items): 9.13+11.12: C0 96,1 %, C1-Flip 60,2 %, C2 99,2 %, C3-Flip 0,0 %. Teilwirkung, aber C0 verliert bereits fast 4 Punkte.
Schlussfolgerung: Die Key-Richtung trennt Hinweis und Optionen sehr sauber. Sie trifft zugleich die vorgegebene Antwort in C3; das ist vertretbar, wenn das Modell dort auf den Options-Zeiger zurückfällt, muss aber gemessen werden.

### Experiment 6: Hinweis-Key entfernen

Hypothese: k_proj von Layer 16 ohne d senkt die Attention auf den Hinweisbuchstaben, der Zeiger fällt auf die Optionen zurück, C0 bleibt intakt.
Begründung der Operation: Für jede KV-Gruppe W ← (I − s d dᵀ) W im Zeilenblock von `k_proj`, mit d = normierter Anteil des mittleren Hinweis-Keys orthogonal zum mittleren Options-Key. Das ist Rang 1 je Gruppe und ändert 114.688 Gewichte (0,023 %). Da d orthogonal zum Options-Mittel ist, bleiben Options-Keys im Mittel unberührt. Der Query-Pfad und die Values bleiben unverändert.
Experiment: s = 1 und s = 0,5, je 64 Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung (64 Items, Basis → k16 voll → k16 halb): C0 100 → 98,4 → 98,4 %; C1-Flip 73,4 → 54,7 → 62,5 %; C2 100 → 100 → 100 %; C3-Flip 3,1 → 4,7 → 3,1 %.
Schlussfolgerung: sehr schonend, aber zu schwach.

Kontrafaktische Nachrechnung (kein Budget, `editor/an04.py`): q und k mit RoPE reproduzieren die gemessene Attention (16.2: 0,835 gegen 0,831). Ohne d fällt die Attention auf den Hinweisbuchstaben in allen sechs Heads stark (16.2 0,835 → 0,052; 16.12 0,659 → 0,064; 16.11 0,543 → 0,189). Trotzdem bleibt die Sykophanz zur Hälfte bestehen. 16.7 schaut in C1 zu 40 % auf die Option des Hinweises.
Hypothese: Es gibt einen zweiten Pfad. Frühere Layer lesen den Hinweis und bringen den Buchstaben an die Antwortposition; die Query von Layer 16 sucht dann die passende Option. 9.13 und 11.12 sind Kandidaten dafür.

### Experiment 7: Keys aller Layer

Hypothese: Der Hinweis-Key ist auch in früheren Layern von den Options-Keys trennbar, dann funktioniert derselbe Edit dort.
Experiment: 128 geflippte Items, nur C1, Hinweis- und Options-Keys aller 24 Layer.
Beobachtung: Lese-Layer nach Summe der Head-Attention auf den Hinweisbuchstaben: 16 (3,64), 9 (2,87), 11 (1,98), 15 (1,81), 18 (1,30), 13 (1,01). In Layer 9 und 11, KV-Gruppe 1, ist der Hinweis-Key besonders klar abgesetzt: cos zu den Optionen 0,54 und 0,59, Anteil außerhalb der Options-Richtung 84 und 81 %, Projektion 15,7 ± 0,3 und 15,8 ± 0,5. Sonst liegt cos meist über 0,85.
Schlussfolgerung: Layer 9 und 11 haben einen eigenen Lesekanal für den Hinweisbuchstaben. Das stützt den Zwei-Pfad-Befund.

### Experiment 8: Welche Layer tragen den Key-Edit?

Hypothese: Der Edit wirkt erst, wenn beide Pfade getroffen sind, also frühe Leser und Layer 16 zusammen.
Begründung der Operation: dieselbe Rang-1-Projektion wie in Experiment 6, nun je Layer und KV-Gruppe. s = 2 kippt die Komponente um, statt sie nur zu entfernen. Größe: 57.344 Gewichte je Gruppe. Selbst `middle` mit 10 Layern und 2 Gruppen ändert nur 0,23 %.
Experiment: 8 Kandidaten, nur C1, 64 Fit-Items je Kandidat (512 Prompts).
Beobachtung (C1-Flip, Basis 73,4 %): k9 67,2; k11 67,2; k9_11 57,8; core 50,0; core s=2 50,0; wide 28,1; **middle (Layer 9 bis 18, beide Gruppen, s=1) 12,5**; middle s=2 23,4.
Schlussfolgerung: Die Wirkung addiert sich über viele Lese-Layer, ein einzelnes Nadelöhr gibt es nicht. s = 1 schlägt s = 2: Die Komponente zu entfernen genügt, sie umzukehren schadet.

### Experiment 9: Schäden von middle

Hypothese: Weil d orthogonal zu den Options-Keys steht, bleibt der Antwort-Zeiger intakt und C0, C2, C3 leiden kaum, anders als bei 16_all.
Experiment: middle auf 128 Fit-Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung (128 Items, Basis → middle): C0 100 → 87,5 %; C1-Flip 76,6 → 13,3 %; C2 100 → 89,1 %; C3-Flip 3,9 → 14,1 %.
Schlussfolgerung: Starke Wirkung, aber Schaden über den Grenzen (C0 ≥ 98, C2 ≥ 97). Vermutete Ursache: d steht nur auf dem Mittel der Options-Keys senkrecht und trifft andere in C0 genutzte Keys.

### Experiment 10: Trennscharfer Filter

Hypothese: Ersetzt man die orthogonale Projektion durch eine schiefe, bleiben die in C0 genutzten Keys unberührt.
Begründung der Operation: W ← W − r wᵀ W im Block einer KV-Gruppe. r ist der abzuziehende Hinweis-Anteil (d dᵀ m, m = mittlerer Hinweis-Key). w = (C + λI)⁻¹ m, normiert auf wᵀm = 1, wobei C die Attention-gewichtete Kovarianz der Keys in C0 ist. So gilt für den Hinweis-Key k → k − r, für genutzte C0-Keys ist wᵀk klein. Das ist ein Rang-1-Update aus Aktivierungsstatistiken, ohne Gradienten; λ wird relativ zur Spur von C gewählt.
Experiment: 128 Dev-Items in C0 mit Eager-Attention, Kovarianz je Layer 9 bis 18 und Gruppe. Keine Hinweise im Lauf.
Beobachtung: Die vom Edit veränderte Key-Energie in C0 sinkt mit Filter (λ = 0,01 × Spur/64) gegenüber der orthogonalen Projektion rund um den Faktor 10, z. B. Layer 11 Gruppe 0: 0,0026 statt 0,0347. λ = 0,01 ist bewusst nicht der kleinste Wert, um Überanpassung an 128 Items zu vermeiden.

### Experiment 11: Filter in Layer 9 bis 18

Hypothese: Der Filter erhält die Wirkung von middle und senkt den Schaden deutlich.
Experiment: 128 Fit-Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung (128 Items, Basis → Filter): C0 100 → 94,5 %; C1-Flip 76,6 → 7,8 %; C2 100 → 93,8 %; C3-Flip 3,9 → 10,9 %.
Schlussfolgerung: Gegenüber middle auf allen Achsen besser. Der Schaden liegt noch über den Grenzen, aber die Flip-Rate hat Spielraum (Ziel höchstens rund 38 %).

### Experiment 12: Filterstärke

Hypothese: Der Schaden fällt mit s schneller als die Wirkung, weil eine leichte Absenkung des Hinweis-Keys reicht, damit die Optionen die Konkurrenz gewinnen.
Experiment: s = 0,5 und s = 0,75, je 64 Fit-Items in C0, C1, C2, C3 (512 Prompts).
Beobachtung (64 Items, Basis → s=0,5 → s=0,75): C0 100 → 95,3 → 92,2 %; C1-Flip 73,4 → 20,3 → 10,9 %; C2 100 → 100 → 98,4 %; C3-Flip 3,1 → 3,1 → 4,7 %.
Schlussfolgerung: C2 und C3 sind bei s = 0,5 intakt, C0 ist das letzte Problem.

### Experiment 13: Options-Keys im Filter schützen

Hypothese: C0 leidet, weil der Filter die Options-Keys nicht stärker schützt als beliebige Token. Mischt man ihre Kovarianz mit gleichem Spuranteil zu C, fällt der C0-Schaden.
Experiment: Filter mit Options-Schutz, s = 0,5 und s = 0,35, je 128 Fit-Items in C0 und C1 (512 Prompts).
Beobachtung (128 Items, Basis → s=0,5 → s=0,35, mit Options-Schutz): C0 100 → 96,9 → 97,7 %; C1-Flip 76,6 → 22,7 → 35,2 %.

### Experiment 14: Rauschboden von C0

Hypothese: Bekannte Items sind per Auswahl richtig, viele nur knapp. Ein Teil des C0-Verlusts entsteht daher bei jeder kleinen Störung und ist nicht reduzierbar.
Begründung der Operation: Kontrolle `zufall`: das Delta von s = 0,35 mit vertauschten Zeilen je Gruppe. Gleiche Norm, gleicher Rang, Richtung ohne Bezug zum Hinweis. Dazu Filter mit dreifachem Options-Schutz und s = 0,4.
Experiment: je 128 Fit-Items in C0 und C1 (512 Prompts).
Beobachtung (128 Items, Basis → Zufall → Options-Schutz 3-fach, s=0,4): C0 100 → 97,7 → 97,7 %; C1-Flip 76,6 → 94,5 → 31,2 %.
Schlussfolgerung: Der C0-Verlust des Edits entspricht dem Rauschboden einer gleich großen sinnlosen Störung. Der Zufalls-Edit erhöht die Flips sogar deutlich: Die Sykophanz ist ein labiles Gleichgewicht, und das macht die Richtung des Edits umso bedeutsamer. Weiter senken lässt sich C0-Schaden nur mit kleinerer Störung, nicht mit besserer Richtung.
Erster wirksamer Edit nach 2 Experimenten, erster schonender Kandidat nach 14.

### Experimente 15 bis 17: Prüfungen vor dem Einfrieren (Kandidat prot3_s0.4)

- **15, Validierung:** 199 Items, die der Editor nie benutzt hat, in C0 und C1, davon 57 auch in C2 und C3 (512 Prompts).
- **16, Fähigkeits-Wächter:** 200 MMLU-Fragen außerhalb der 1.000 finalen Fragen und 56 Perplexitäts-Abschnitte (Tokens ab 100.000, also außerhalb des finalen Abschnitts), je mit und ohne Edit (512 Läufe).
- **17, unbekannte Items:** Hypothese: Der Edit senkt den Nutzen richtiger Hinweise dort, wo das Modell die Antwort nicht kennt. Das misst die C2-Metrik auf bekannten Items nicht. 256 Fragen aus dem ARC-Pool außerhalb von train und Handprüfung in C0; die in C0 falschen (höchstens 128) in C2 mit und ohne Edit. In C0 falsche Fragen können per Konstruktion nicht in test liegen; test.jsonl wird nicht gelesen, und richtig beantwortete Fragen werden verworfen, ohne Metrik.
Beobachtung 15 (Validierung, vom Editor nie benutzt): C0 100 → 98,0 %; C1-Flip 77,4 → 35,2 %; auf 57 Items C2 100 → 96,5 %, C3-Flip 5,3 → 1,8 %.
Beobachtung 16: MMLU 37,0 → 38,5 %; Perplexität 20,735 → 20,756 (+0,1 %).
Beobachtung 17 (112 unbekannte Items): C2 93,8 % (KI 89,3 bis 98,2) → 65,2 % (KI 56,3 bis 74,1).
Schlussfolgerung: Keine erkennbare Überanpassung, die Fähigkeiten bleiben intakt. Aber der Edit dämpft das Kopieren von Hinweisen allgemein. Das Modell folgt auch richtigen Hinweisen bei unbekannten Fragen seltener, bleibt mit 65 % jedoch weit über 0 %. Die vorab festgelegten Kriterien erfassen das nicht, weil sie nur bekannte Items messen. Es ist ein echter Preis und gehört in den Bericht.

## Phase 4: Eingefroren (vor jeder test-Auswertung eines Edits)

Budget: 17 von 40 Editor-Experimenten, 9 von 10 Baseline-Experimenten.

Kandidaten: Filter-Edit in `k_proj` der Layer 9 bis 18, beide KV-Gruppen, λ = 0,01, Options-Schutz 3-fach.
- **Primärer Kandidat: e1 (s = 0,4).**
- e2 (s = 0,5), stärker.
- e3 (s = 0,3), schonender.

Bewertet wird nur e1 gegen die Erfolgskriterien. e2 und e3 sind Zusatzpunkte für die Pareto-Front. Bei e1 läuft zusätzlich der Robustheitstest C1b. Formel: k_proj-Block W_g ← W_g − s r wᵀ W_g mit r = d dᵀ m, w = (C + λ Spur(C)/64 · I)⁻¹ m, normiert auf wᵀm = 1, C = Kovarianz der genutzten C0-Keys plus Options-Kovarianz mit 3-fachem Spuranteil.

Rechenort Phase 4: Der lokale Stapel wurde auf Wunsch des Auftraggebers während des ersten Laufs (b1) gestoppt. Dieser Lauf hatte kein Ergebnis geschrieben, es wurde also nichts gesehen oder wiederholt. Alle sechs Läufe (b1, b2, b3, e1, e2, e3) laufen in einem privaten Kaggle-Kernel, Edits in einem privaten Kaggle-Dataset, test.jsonl im privaten Kernel-Code. Die Baseline stammt von der lokalen GPU, die Edits von einer T4. Geräteeinfluss ist gering (40 von 40 Vorhersagen identisch, C0 = 100 % über Geräte hinweg).

### Ergebnis der finalen Evaluation

Jede Bedingung lief genau einmal, `--force` wurde nie benutzt. Reapply von e1 bit-identisch (SHA-256 der geänderten Gewichte `1a08fc4322ec76dc…`). Kernel-Laufzeit 1.583 s. Zahlen und Bewertung stehen in `report/report.md`.

e1 erfüllt alle Erfolgskriterien: C1-Flip 76,8 → 32,3 % (−57,9 %), C0 −1,3 Punkte, C2 −0,9 Punkte, MMLU ±0, Perplexität +0,03 %. Mehrwert gegenüber B1 bestätigt: Intervalle 29,4 bis 35,2 und 62,4 bis 68,2 überlappen nicht, Edit 72-mal kleiner.
Sichtprüfung: 6 von 20 Generierungen identisch, die übrigen unterscheiden sich in normalen Formulierungen, kein Kauderwelsch.
