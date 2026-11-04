# Weight-Space Programming gegen Sykophanz: Bericht

**Kernaussage:** Ein gradientenfreier Gewichtsedit an 0,23 % der Parameter senkt die C1-Flip-Rate von Qwen2.5-0.5B auf dem Test-Split von 76,8 % (95-%-KI 74,2 bis 79,4) auf 32,3 % (KI 29,4 bis 35,2), also um 57,9 % relativ, und erfüllt damit alle vorab festgelegten Erfolgskriterien.

Der wichtigste Vorbehalt steht außerhalb dieser Kriterien: Bei Fragen, die das Modell nicht kennt, hilft auch ein *richtiger* Hinweis nach dem Edit weniger (93,8 % → 65,2 % auf 112 Items). Der Edit dämpft das Kopieren von Hinweisen allgemein. Er lehrt das Modell nicht, gute von schlechten Hinweisen zu unterscheiden.

Das Ergebnis gilt für das 0,5B-Modell, nicht für das ursprüngliche Ziel 1,5B. Dessen C1-Flip-Rate lag auf `train` mit 15,0 % (KI 12,8 bis 17,2) unter der Go-Schwelle von 30 %. Wie im Plan vorgesehen, kam deshalb 0,5B zum Einsatz (`train`: 79,9 %, KI 77,4 bis 82,4).

## 1. Ergebnistabelle

Test-Split mit 999 bekannten Items; MMLU mit 1.000 festen Fragen; Perplexität auf 99.840 Tokens. In Klammern steht das 95-%-Bootstrap-Intervall.

| Bedingung | C1-Flip % | C3-Flip % | C0-Acc % | C2-Acc % | MMLU % | Perplexität | Geänderte Parameter % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ausgangsmodell | 76,8 (74,2–79,4) | 6,5 (5,0–8,1) | 100,0 | 100,0 | 43,0 (40,0–46,0) | 22,87 | 0 |
| B1 Difference-of-Means | 65,3 (62,4–68,2) | 12,4 (10,4–14,4) | 93,9 (92,4–95,3) | 99,8 | 42,9 (39,9–45,9) | 25,33 | 16,72 |
| B2 Zufallsrichtung | 80,0 (77,5–82,5) | 6,0 (4,6–7,5) | 98,1 (97,2–98,9) | 100,0 | 43,1 (40,1–46,2) | 23,29 | 16,72 |
| B3 Systemprompt | 71,2 (68,5–74,0) | 22,0 (19,5–24,6) | 91,9 (90,2–93,5) | 99,9 | 42,7 (39,6–45,8) | 22,87 | 0 |
| **e1 (primär, s = 0,4)** | **32,3 (29,4–35,2)** | 4,9 (3,6–6,3) | 98,7 (97,9–99,3) | 99,1 (98,5–99,6) | 43,0 (40,0–46,1) | 22,88 | 0,23 |
| e2 (s = 0,5) | 24,8 (22,1–27,5) | 4,7 (3,4–6,1) | 98,6 (97,8–99,3) | 97,9 (97,0–98,7) | 43,0 (39,9–46,1) | 22,88 | 0,23 |
| e3 (s = 0,3) | 41,9 (38,8–45,0) | 5,9 (4,5–7,4) | 98,9 (98,2–99,5) | 99,8 | 43,2 (40,2–46,3) | 22,87 | 0,23 |

B4 und die optionale Erweiterung entfielen im verkürzten Lauf. Die Perplexität von B3 ist identisch mit dem Ausgangsmodell, weil sie ohne Prompt gemessen wird und B3 keine Gewichte ändert.

### Prüfung gegen die Erfolgskriterien (nur e1)

| Kriterium | Grenze | Gemessen | Erfüllt |
| --- | --- | --- | --- |
| C1-Flip relativ | mindestens −50 % | −57,9 % (bei den ungünstigsten Intervallgrenzen −52,5 %) | ja |
| C0-Verlust | höchstens 2 Punkte | 1,3 | ja |
| C2-Verlust | höchstens 3 Punkte | 0,9 | ja |
| MMLU-Verlust | höchstens 2 Punkte | 0,0 | ja |
| Perplexität | höchstens +5 % | +0,03 % | ja |
| Vorsprung vor B1 | Intervalle überlappen nicht | 29,4–35,2 gegen 62,4–68,2 | ja |

**Robustheit:** Die im Editor-Lauf nie verwendete Formulierung `My teacher said it's ({falsch}).` fällt von 87,3 % (KI 85,2 bis 89,4) auf 38,8 % (KI 35,8 bis 41,9), also um 55,5 % relativ. Der Edit passt demnach nicht nur auf den Wortlaut des ersten Hinweises.

**Sichtprüfung:** 6 von 20 offenen Generierungen sind identisch. Die übrigen weichen in gewöhnlichen Formulierungen ab, Kauderwelsch tritt nicht auf. Ein Beispiel: „Guten Morgen, wie geht es Ihnen?“ wird zu „…wie geht es dir?“.

## 2. Pareto-Plot

![C1-Flip-Rate gegen MMLU-Verlust und gegen C0-Verlust](pareto.png)

MMLU trennt die Bedingungen nicht: Alle Punkte liegen innerhalb von 0,3 Punkten, weit unter der Unsicherheit von ±3 Punkten. Deshalb steht rechts der aussagekräftigere C0-Verlust. Die drei Editor-Kandidaten dominieren dort alle Baselines. e1, e2 und e3 bilden untereinander eine Abwägung zwischen Flip-Rate und C2.

## 3. Mechanismus

### Befund

Sykophanz entsteht in diesem Modell nicht durch ein einzelnes Merkmal. Sie entsteht, weil der Hinweisbuchstabe mit den Antwortoptionen um denselben Zeiger konkurriert.

1. **Layer 16 ist der Antwort-Zeiger.** In C0 schaut Head 16.7 von der Antwortposition zu 64 % auf die Markierung der richtigen Option, 16.11 zu 33 %.
2. **Der Hinweis stiehlt diesen Zeiger.** In C1 schauen 16.2 (83 %), 16.12 (64 %) und 16.11 (55 %) auf den Hinweisbuchstaben.
3. **Es gibt einen zweiten Pfad.** Die Heads 9.13 und 11.12 schauen in C0 praktisch nie auf Buchstaben (unter 0,5 %), in C1 zu 90 und 80 % auf den Hinweis. Head 16.7 sucht in C1 zu 40 % die Option des Hinweises. Der Buchstabe erreicht die Antwortposition also schon vor Layer 16. Darauf deutet auch die kontrafaktische Rechnung: Der Key-Edit allein in Layer 16 senkt die Attention dort fast vollständig (16.2: 83 % → 5 %), die Flip-Rate fällt aber nur auf 55 %.
4. **Die Wirkung ist über Layer verteilt.** Einzelne Layer senken die Flip-Rate kaum, Layer 9 und 11 allein nur auf 67 %, zusammen auf 58 %; erst alle Layer 9 bis 18 zusammen wirken stark.

Der Attribution zufolge entsteht die Logit-Verschiebung (+5,24) in Layer 16 bis 23. Sie zeigt aber nur direkte Beiträge und führte in die Irre: 22.2 hatte die höchste Attribution und keine kausale Wirkung, 16.2 fast keine Attribution und große Wirkung. Erst das gezielte Stummschalten hat das gezeigt.

### Warum nicht einfach Heads abschalten

Sechs Heads in Layer 16 stumm zu schalten senkt die Flip-Rate auf 15 %, zerstört aber das Modell: C0 82 %, C2 59 %, C3-Flip 84 %. Die Heads sind der Zeiger selbst. Der Edit muss also nur den Hinweis unattraktiv machen und den Zeiger erhalten.

### Der Edit

Betroffen ist `k_proj` der Layer 9 bis 18, in beiden KV-Gruppen g. Die Keys entscheiden, wie stark eine Position angeschaut wird. Für jeden Zeilenblock W_g:

```
W_g ← W_g − s · r wᵀ W_g

m  = mittlerer Key des Hinweisbuchstabens (128 geflippte train-Items)
o  = mittlerer Key der Optionsmarkierungen
d  = normierter Anteil von m orthogonal zu o
r  = d dᵀ m
C  = in C0 genutzte Key-Kovarianz (Attention-gewichtet) + Options-Kovarianz mit 3-fachem Spuranteil
w  = (C + λ·Spur(C)/64 · I)⁻¹ m, normiert auf wᵀm = 1, λ = 0,01
s  = 0,4
```

Das ist eine schiefe Rang-1-Projektion: Hinweis-Keys verlieren einen Teil von r, weil wᵀm = 1 gilt. In C0 genutzte Keys bleiben fast unberührt, weil w gegen sie gerichtet ist. Gegenüber der orthogonalen Projektion sinkt die veränderte Key-Energie so etwa um den Faktor 10.

Umfang: 10 Matrizen, 1.146.880 Gewichte (0,232 %), Rang 2 je Matrix (Rang 1 je KV-Gruppe). Embeddings und LM-Head bleiben unverändert. Eine erneute Anwendung ergibt bit-identische Gewichte. Der Edit ist kleiner als das 1-%-Limit und 72-mal kleiner als B1.

### Geprüfte Hypothesen

- **Kopier-Heads: bestätigt, aber anders als erwartet.** Der Kopierpfad ist nicht durch wenige Heads zu dämpfen, weil dieselben Heads die Antwort auswählen.
- **Deferenz-Richtung: bestätigt.** Die Head-Beiträge in C1 und C2 korrelieren mit 0,86. Das ist derselbe Pfad, unabhängig davon, ob der Hinweis stimmt.
- **Widerspruchs-Pfad: nur teilweise geklärt.** Die vorgegebene Antwort in C3 hat fast denselben Key wie der Hinweis (cos 0,98), der Edit erhöht C3 aber nicht (6,5 → 4,9 %, Intervalle überlappen). Einen eigenen Pfad für „Are you sure?“ habe ich nicht gesondert untersucht, weil C3 bei diesem Modell ohnehin selten kippt.

## 4. Budget

| Größe | Wert |
| --- | --- |
| Editor-Experimente | 17 von 40 |
| Baseline-Experimente | 9 von 10 |
| Bis zum ersten wirksamen Edit | 2 Experimente (noch schädlich) |
| Bis zum ersten schonenden Kandidaten | 14 Experimente |
| Wanduhrzeit | rund 4,5 Stunden, davon rund 2 Stunden ein abgebrochener lokaler Lauf |
| GPU-Zeit Kaggle | rund 0,8 Stunden |
| GPU-Zeit lokal | rund 2,5 Stunden, davon 2 Stunden ohne Ergebnis |
| Kosten | keine Kaggle- oder API-Kosten; die Arbeit des Editors läuft über das Claude-Kontingent |

Das Modell-Audit ist für beide Modelle bestanden: nur Zahlentensoren, kein Fremdcode, kein Verbindungsversuch bei Laden und Inferenz. Das 0,5B-Audit lief auch im Kaggle-Kernel.

## 5. Ehrliche Grenzen

- **Ein Modell, eine Aufgabenfamilie, MC-Format.** Es gibt keinen Beleg für 1,5B, für offene Antworten oder für andere Modelle. Die Hinweise haben immer die Form `(X)`. Ein Hinweis ohne Klammern oder in Worten wurde nicht getestet.
- **Der Edit löst das Vertrauensproblem nicht.** Er senkt das Kopieren allgemein. Die Kriterien messen nur bekannte Items und erfassen diesen Preis nicht. Die Zahl 65,2 % stammt aus einem einzelnen Zusatztest auf 112 Items außerhalb von `test`, sie ist also vorläufig. Sie gilt für e1.
- **Es bleibt fast ein Drittel Flips.** 32,3 % ist weit von null entfernt. Ein kleineres s senkt den Schaden, ein größeres die Flips. e2 ist bei C1 besser, nähert sich bei C2 aber der Grenze.
- **Rauschboden.** Eine sinnlose Störung gleicher Größe kostete auf `train` ebenfalls rund 2 Punkte C0. Etwa dieser Betrag ist also nicht dem Mechanismus zuzuschreiben, sondern der Auswahl knapp richtiger Items. Dieselbe Kontrolle erhöhte die Flips auf 94,5 %: Das Verhalten ist labil, und das gilt auch für Eingriffe in die Gegenrichtung.
- **Editor gegen B1.** Der Editor leistet hier klar mehr als B1, und auch mehr als B3. Das ist aber ein Vergleich gegen eine fest vorgegebene Baseline, die nur 8 von 24 Layern testen konnte, weil mehr nicht ins Budget passte. Der Editor hatte zudem fast doppelt so viele Experimente wie B1 (17 gegen 9). Eine bessere Direktionsbestimmung könnte B1 also noch verbessern.
- **Vergleichbarkeit über Geräte.** Die Baseline lief auf der lokalen GPU, alle anderen Bedingungen auf einer Kaggle-T4. Auf 40 Stichproben-Items stimmten die Vorhersagen beider Geräte überein, und C0 blieb über Geräte hinweg bei 100 %.
- **Sicherheitsverhalten.** Ablehnungs- und Sicherheitsverhalten waren kein Ziel des Edits. Gemessen wurde es aber auch nicht. Dass es unverändert blieb, ist also nicht belegt.
- **Positionseffekt.** Bei 1,5B stieg die Flip-Rate mit der Hinweisposition von 8 % (A) auf 25 % (D). Bei 0,5B wurde das nicht ausgewertet.

**Abweichungen vom Plan:** Zielmodell 0,5B statt 1,5B (planmäßiger No-Go-Pfad); B1 mit 8 statt allen Layern (freigegeben); verkürzter Lauf ohne B4 (freigegeben); Phase 4 auf Kaggle statt lokal (auf Wunsch); der erste lokale Lauf von B1 wurde abgebrochen, bevor er ein Ergebnis schrieb.

## 6. Vergleich mit dem Ausgangs-Post

Bezug ist der Blogbeitrag [„Weight-Space Programming“](https://rikiyat.github.io/blog/weight-space-programming/). Die Zuordnung zu MIT CSAIL stammt aus der Aufgabenstellung und wurde hier nicht eigenständig geprüft.

| Punkt | Post („Weight-Space Programming“) | Dieser Versuch |
| --- | --- | --- |
| Sykophanz | wurde nur erkannt, Experiment 7 | wurde gesenkt |
| Modell | Gemma 3 270M, eigens feinjustiert | Qwen2.5-0.5B, unverändert veröffentlicht |
| Editor | GPT-6 Astra | Claude Opus 5.5 |
| Bevorzugte Edit-Stelle | MLP-Spalten, Schleifen-Richtung | `k_proj`, Rang 1 je KV-Gruppe |
| Größe des Edits | 0,016 % (Qwen3.5-0.8B) | 0,23 % |

**Bestätigt:**
- Ein Sprachmodell kann die Gewichte eines kleinen Modells ohne Gradienten gezielt ändern.
- Schon wenige Experimente genügen für einen ersten wirksamen Edit.
- Attention-Muster sind für Sykophanz aussagekräftig. Der Post fand bei Gemma einen Head, der von der Antwortposition auf die Nutzerantwort schaut (10.2). Hier zeigt sich dasselbe Muster, allerdings auf mehrere Heads verteilt.
- Edits können Hindernisse entfernen. Der Fähigkeitsstand (MMLU) bleibt gleich, neue Fähigkeiten entstehen nicht.

**Neu:**
- Eine Reparatur statt nur einer Erkennung, gemessen auf einem blinden Test-Split gegen mechanische Baselines.
- Der Zeiger-Befund: Der kopierende Pfad und die normale Antwortwahl sind dieselben Heads. Der Eingriff muss deshalb auf der Key-Seite ansetzen.
- Ein allgemeines Rezept für schonende Edits: statt orthogonal zu projizieren, einen Filter aus der Nutzungs-Kovarianz verwenden.
- Der Preis: Bei unbekannten Fragen helfen auch richtige Hinweise weniger.

Die Zahlen sind nicht direkt mit dem Post vergleichbar, weil Modell, Aufgabe und Sykophanz-Quelle verschieden sind.
