# Audio8-TTS für ComfyUI – Umsetzungsplan

## 1. Ziel und Umfang

Native ComfyUI-Nodes für `Edge0/Audio8-TTS-Realistic-0.6b` unter Windows 11 mit CUDA. Das MVP verwendet die Transformers/PyTorch-Implementierung und ComfyUIs vorhandenen `AUDIO`-Datentyp.

Das System soll Text-to-Speech, Zero-Shot Voice Cloning, wiederverwendbare Voice-Presets und lange Texte unterstützen. Referenzaudio wird direkt im Speicher verarbeitet. Es entstehen während der normalen Ausführung keine temporären WAV-Dateien. Modell und Processor werden über ComfyUIs Node-Caching wiederverwendet. Die vorhandene PyTorch-/CUDA-Installation von ComfyUI darf nicht ersetzt oder verändert werden.

**Im MVP enthalten:** Model Loader, Encode Reference, Save Voice, Load Voice und TTS; Text-only und referenzkonditionierte Ausgabe; Voice-Presets als `.safetensors`; automatisches Chunking langer Texte; Windows-Pfade und Fehlerbehandlung.

**Außerhalb des MVP:** SGLang, ONNX, Training/Fine-Tuning, Streaming, echtes Batch-TTS, Multi-GPU, Multi-Speaker-Prompts und Web-API. Kein automatischer Modell-Download: das MVP lädt ein lokal vorhandenes Modell.

## 2. Architekturentscheidungen

- Audio8 bleibt für Prompt-Aufbau und Modellverhalten zuständig. Die Runtime verwendet die öffentlichen Schnittstellen `encode_audio()`, `generate()`, `decode_audio()` und `processor(..., reference_codes=..., reference_text=...)`.
- Ein Voice-Preset enthält Codec-Codes und den zugehörigen Referenztext. Es ist kein Speaker-Embedding.
- Modell und Processor liegen gemeinsam in einem `Audio8ModelBundle`. `Audio8Reference` hält Codes auf CPU als `torch.long` mit Form `[10, T]`.
- ComfyUI-Nodes bleiben dünne Adapter im vorhandenen Stil des Repositories. Keine Migration auf eine neue Node-API.
- Modelle liegen unter `ComfyUI/models/audio8_tts/`, Presets unter `ComfyUI/models/audio8_voices/`. Für beide werden die bestehenden `folder_paths`-Muster verwendet.
- Gerät und Datentyp werden über ComfyUI-Modellverwaltung aufgelöst. CUDA/BF16 ist der Standard, CPU kann optional mit FP32 laufen. Keine separate Torch- oder Torchaudio-Installation.
- Transformers >=4.57,<6 verwenden. Der ursprüngliche Entwurf begrenzte vorsorglich auf <5; vollständige lokale CUDA/BF16-Tests mit Transformers 5.9.0 erlauben den erweiterten Bereich.
- Audio8s Codec wird lazy geladen und bleibt nach dem Laden auf dem Modellgerät. Kein aggressives Offloading und kein `torch.cuda.empty_cache()` pro Node.
- Modellabhängige Integrationstests sind optional, falls das Modell lokal fehlt; modellfreie Prüfungen müssen ohne CUDA und ohne Download laufen.

## 3. Geplante Projektstruktur

Die vorhandene Struktur hat Vorrang; es erfolgt keine unnötige Umstrukturierung.

```text
__init__.py
nodes.py
audio8_runtime.py
audio8_types.py
voice_presets.py
requirements.txt
README.md
tests/
```

`audio8_types.py` enthält die Datenobjekte und gemeinsame Referenzvalidierung. `audio8_runtime.py` verantwortet Laden, Audio-Konvertierung, Encoding, Generierung, Decoding sowie Gerät-/Datentypbehandlung. `voice_presets.py` behandelt Format, Validierung, Dateinamen und Ein-/Ausgabe. `nodes.py` registriert die ComfyUI-Adapter.

## 4. Phasen und Abnahmepunkte

### Phase 0 – Bestandsaufnahme und Schnittstellenvertrag

**Arbeiten:** Repository-Konventionen und ComfyUI-Audio-Nodes prüfen; aktuelle Audio8-Inferenz, Processor, Modellcode und Konfiguration untersuchen. Tatsächliche Argumente und Rückgaben, Tensorformen, Sample Rates, EOS-Signal, Codec-Ladeverhalten und Transformers-Anforderungen festhalten. `from_pretrained()`-Parameter mit der unterstützten Transformers-Version abgleichen.

**Ergebnis/Gate:** Ein kurzer, im Code oder Plan dokumentierter Schnittstellenvertrag liegt vor. Vor Implementierung sind insbesondere `generate()`-Ausgabe und `encode_audio()`-/`decode_audio()`-Formen verifiziert.

### Phase 1 – Typen, Validierung und Pfade

**Arbeiten:** `Audio8ModelBundle` und `Audio8Reference` definieren. Referenzcodes auf Form `[10, T]`, `T > 0`, ganzzahligen Inhalt und Wertebereich `0..4095` prüfen; Modell-ID, Codec-Metadaten und Referenztext prüfen. Modell- und Preset-Ordner mit `folder_paths` registrieren. Preset-Namen an der Speichergrenze validieren und Pfad-Traversal bzw. absolute Pfade verhindern.

**Ergebnis/Gate:** Daten- und Pfadgrenzen sind modellfrei prüfbar. Ungültige Formen, Werte, Metadaten und Namen liefern klare Fehler.

### Phase 2 – Modell laden

**Arbeiten:** Den lokalen Ordner `models/audio8_tts/Audio8-TTS-Realistic-0.6b/` laden. Processor und Modell einmal im Bundle halten, `eval()` setzen und Gerät/Datentyp gemäß ComfyUI-Konventionen anwenden. Unterstützte Datentypen: `auto`, `bfloat16`, `float16`, `float32`; CPU verwendet FP32. Fehlende Dateien oder inkompatible Transformers-Versionen verständlich melden.

**Ergebnis/Gate:** `Audio8 Model Loader` liefert ein Bundle, das ComfyUI bei unveränderten Loader-Eingaben wiederverwenden kann. Es wird kein Modell im TTS-Node neu geladen und kein Download ausgelöst.

### Phase 3 – Audio-Konvertierung und Referenz-Encoding

**Arbeiten:** ComfyUI-`AUDIO` mit Batchgröße 1 annehmen, Kanäle zu Mono mitteln und auf `model.config.codec_sample_rate` bringen. Für Audio8 die korrekte `[1, 1, samples]`-Form und Längenangabe erzeugen, `model.encode_audio()` aufrufen und gültige Codes als CPU-`long` normalisieren.

**Ergebnis/Gate:** `Audio8 Encode Reference` gibt eine validierte `AUDIO8_REFERENCE` aus. Keine WAV-Datei wird geschrieben. Mono- und Stereo-Eingaben sowie Batchgrößenfehler sind abgedeckt.

### Phase 4 – Text-only TTS als vertikaler Durchstich

**Arbeiten:** Kurzen Text durch Processor, `generate()`, `decode_audio()` und ComfyUI-Ausgabe führen. Sampling-Eingaben: Seed 42, `max_new_tokens=1024`, Temperatur 0.8, `top_p=0.95`, `top_k=50`, `greedy=false`; Generator gerätespezifisch und reproduzierbar erzeugen. Ausgabe auf gültige Samples kürzen und als Mono-`AUDIO` der Form `[1, 1, S]` zurückgeben.

**Ergebnis/Gate:** Text-only TTS funktioniert mit lokal verfügbarem Modell. Modellfehler und leere/ungültige Audioausgabe werden klar gemeldet.

### Phase 5 – Voice Conditioning und NO_EOS-Retry

**Arbeiten:** Optionale Referenz über `reference_codes` und `reference_text` an den Processor reichen. Kein eigenes Prompt-Packing. `output.finished` prüfen; bei fehlendem EOS genau den betreffenden Lauf/Chunk mit höherem `retry_max_new_tokens` wiederholen, falls das Limit größer ist. Gültige Codes ohne EOS dürfen mit Warnung decodiert werden; fehlende gültige Codes ergeben einen verständlichen Fehler.

**Ergebnis/Gate:** Text-only und Voice-Cloning-Pfade verwenden dieselbe Runtime. Eine gespeicherte Referenz wird nicht erneut encodiert.

### Phase 6 – Chunking langer Texte

**Arbeiten:** Chunker als eigene, modellfreie Logik implementieren. Bei Textlänge `<= max_chunk_chars` bleibt der direkte Einzelaufruf erhalten. Längere Texte werden bevorzugt an Absatz-, Satz-, danach Satzteilgrenzen und zuletzt Leerzeichen geteilt. Wörter werden nur im unvermeidbaren Grenzfall geteilt. Optionen: `chunking` (`auto`/`disabled`), Ziel 130 Zeichen, Maximum 150, Satzpause 180 ms, Absatzpause 350 ms. `max_chunk_chars` muss mindestens so groß wie `target_chunk_chars` sein. Pausen kommen nur zwischen Chunks, nie ans Ende.

Für einen kompletten TTS-Aufruf wird ein Generator angelegt und über alle Chunks weiterverwendet. Dieselbe Voice-Referenz wird jedem Chunk übergeben. NO_EOS-Retry gilt pro Chunk. Audiosegmente und Stille werden als Tensoren zusammengesetzt; keine temporären Dateien und standardmäßig kein Crossfade. Bei Chunkfehlern wird abgebrochen und Chunk-Index plus gekürzter Textanfang gemeldet.

**Ergebnis/Gate:** Chunkgrenzen, Paragraph-Pausen, Short-Text-Fast-Path, gemeinsamer Seedzustand und Fehlerverhalten sind ohne Modell testbar.

### Phase 7 – Voice-Presets speichern und laden

**Arbeiten:** `.safetensors` mit Tensor `codes` `[10, T]` und Metadaten `format`, `format_version`, `model_id`, `reference_text`, `codec_sample_rate`, `num_codebooks`, `codebook_size` und `frames` definieren. Metadatenwerte werden als Strings gespeichert. Atomisch in den Voice-Ordner schreiben (temporäre Datei im Zielordner, danach `os.replace`). Beim Laden alle Metadaten, Dimensionen, Wertebereiche und Modellkompatibilität validieren. Originalaudio und Quellpfade werden nicht gespeichert.

**Ergebnis/Gate:** Roundtrip erhält Codes und Referenztext exakt. Beschädigte, inkompatible oder manipulierte Dateien werden mit nachvollziehbarem Fehler abgelehnt. Dropdown listet `.safetensors` aus dem registrierten Voice-Ordner.

### Phase 8 – Nodes integrieren

**Arbeiten:** Die fünf Nodes `Audio8 Model Loader`, `Audio8 Encode Reference`, `Audio8 Save Voice`, `Audio8 Load Voice`, `Audio8 TTS` in der vorhandenen Registrierung verdrahten. Sockettypen sind `AUDIO8_MODEL`, `AUDIO8_REFERENCE` und ComfyUI `AUDIO`. Save-Ausgaben und Kategorien an bestehende Repository-Konventionen anpassen. Lange TTS-Läufe knapp und mit Chunkindex protokollieren; falls das Node-System Fortschritt unterstützt, diesen verwenden.

**Ergebnis/Gate:** Alle fünf Nodes erscheinen und ihre Datenflüsse entsprechen den Typverträgen. Fehler werden an der Node-Grenze verständlich weitergegeben.

### Phase 9 – Verifikation und Dokumentation

**Modellfreie Prüfungen:** Preset-Roundtrip und Metadaten, ungültige Referenzformen/-werte, Path Traversal, Mono-/Stereo-Konvertierung, Chunking, Pausen und deterministischer Generatorfluss.

**Modellintegration, sofern lokal verfügbar:** Import in ComfyUIs Python, Modellladen auf CUDA/BF16, Encoding, Text-only TTS, gecachte Referenz-TTS, Decode-Ausgabe und Direct-vs-Cached-Vergleich mit identischen deterministischen Einstellungen.

**Windows-Abnahme:** Windows-Pfade, atomisches Speichern, Dropdown-Erkennung nach Neustart und Verbindung mit ComfyUI Save Audio prüfen. Bestehende Torch-/CUDA-Version bleibt unverändert.

**Dokumentation:** README mit Installation über ComfyUIs Python, Hinweis Torch/Torchaudio nicht separat zu installieren, Modellpfad, den drei Minimalworkflows (direkte Referenz, Stimme speichern, gespeicherte Stimme verwenden), Optionen und MVP-Grenzen erstellen.

**Ergebnis/Gate:** Jede Akzeptanzanforderung ist einer modellfreien Prüfung, Modellintegration oder manuellen Windows-Prüfung zugeordnet. Nicht ausführbare Modelltests werden als offen dokumentiert, nicht als bestanden ausgegeben.

## 5. Abnahmekriterien

1. Das lokal abgelegte Audio8 Realistic 0.6B wird aus ComfyUI geladen und bei wiederholter Ausführung wiederverwendet.
2. CUDA/BF16 und optional CPU/FP32 funktionieren, ohne Torch/Torchaudio zu installieren oder zu verändern.
3. Text-only TTS liefert natives ComfyUI-`AUDIO` mit Form `[1, 1, S]`.
4. Referenzaudio wird ohne WAV-Zwischendatei mono konvertiert, passend resampelt und über `encode_audio()` in `[10, T]`-Codes umgewandelt.
5. Voice-Presets speichern Codes und Pflicht-Referenztext in validierten `.safetensors` und sind ohne Originalaudio wiederverwendbar.
6. Der Processor erhält gespeicherte Codes über seinen bestehenden `reference_codes`-Pfad.
7. Lange Texte werden an sinnvollen Grenzen verarbeitet, mit Satz-/Absatzpausen zusammengefügt und behalten reproduzierbares Seed-Verhalten.
8. Retry erfolgt pro Chunk; valide Audioausgabe ohne EOS bleibt nutzbar und wird protokolliert.
9. Path Traversal und ungültige oder inkompatible Presets werden abgewiesen.
10. Das Custom Node importiert in der vorgesehenen Windows-ComfyUI-Umgebung und die Ausgabe funktioniert mit Save Audio.
11. Direct-vs-Cached Reference ist bei identischen deterministischen Einstellungen funktional äquivalent, sofern der Integrationstest mit lokalem Modell ausgeführt werden kann.

## 6. Risiken und offene Verifikationen

- Upstream-Signaturen und die installierte Transformers-Version können vom Entwurf abweichen. Phase 0 muss dies klären, bevor Runtime-Code festgelegt wird.
- Der Codec wird lazy und separat verwaltet. Device-Wechsel nach dem ersten Codec-Aufruf sind im MVP nicht vorgesehen.
- Der Direct-vs-Cached Vergleich benötigt Modellgewichte und eine geeignete Audio-Referenz; er kann nicht als modellfreier Test gelten.
- Die maximale Chunklänge ist eine Chunker-Grenze, keine Garantie über Modellqualität. Untrennbare lange Tokens können sie ausnahmsweise überschreiten.

## 7. Nicht-Ziele nach MVP

Nach erfolgreicher MVP-Abnahme können Modell-Download, erweitertes ComfyUI-Offloading, weitere Audio8-Modelle oder Streaming separat geplant werden. Diese Punkte erweitern den MVP nicht stillschweigend.
