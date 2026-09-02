# Email Thread Triage: Classification with Contextual Summarization

Proyecto de curso — AIML339 (Artificial Intelligence Project), Victoria University of Wellington, 2026.
Supervisor: Aaron Chen.

## Pregunta de investigación

¿Clasificar cada email de un hilo por su acto de habla (request, propose, commit,
meeting, subjective, informative) antes de resumir el hilo con un LLM produce
mejores resúmenes que resumir el texto crudo directamente?

## Estructura del pipeline

1. **Clasificación multi-etiqueta**: cada email puede tener varias etiquetas
   activas a la vez (Request/Propose/Commit/Meeting/Subjective/Informative),
   no una sola clase. Se compara:
   - Baseline: TF-IDF + `OneVsRestClassifier(LogisticRegression)`
   - Modelo principal: BERT + LoRA, capa de salida sigmoide + BCE loss
2. **Resumen contextual**: un LLM (OpenAI API) genera resúmenes de cada hilo
   bajo dos condiciones — (A) texto crudo, (B) texto + etiquetas de
   clasificación predichas — evaluadas con ROUGE y revisión manual.
3. **Demo opcional**: aplicación del pipeline a una muestra pequeña y
   anonimizada del Gmail personal del autor (no forma parte de la
   evaluación formal).

## Estructura del repositorio

```
.
├── src/                        # Código fuente
│   ├── bc3_parse.py             # Parser de BC3 (corpus.xml + annotation.xml -> CSVs + split)
│   └── bc3_baseline_tfidf.py    # Baseline TF-IDF + OneVsRestClassifier
├── data/                        # Datos procesados (derivados de BC3)
│   ├── bc3_emails_labeled.csv   # 261 emails, columnas multi-etiqueta
│   ├── bc3_sentences.csv        # 3222 frases (para EDA/detalle)
│   ├── bc3_summaries.csv        # Resúmenes de referencia humanos (BC3)
│   └── bc3_split.json           # Split train/val/test a nivel de hilo (semilla fija)
├── results/                     # Salidas de evaluación
│   └── bc3_baseline_test_predictions.csv
├── docs/                        # Documentación del proyecto
│   └── bitacora-proyecto.md     # Bitácora detallada: qué se hizo, decisiones, dificultades
└── notebooks/                   # Notebooks de Colab (BERT+LoRA, resumen, ROUGE) — por añadir
```

## Dataset: BC3 Corpus

40 hilos de email reales (261 emails, 3222 frases) con resúmenes de
referencia humanos y anotaciones de acto de habla por frase y anotador.
Licencia: **Creative Commons Attribution-Share Alike 3.0**.

Referencia: Ulrich, J. (2008). *Supervised Machine Learning for Email
Thread Summarization* (Master's thesis, University of British Columbia).
https://dx.doi.org/10.14288/1.0051210

> Nota: los ficheros XML crudos originales (`corpus.xml`, `annotation.xml`)
> no están incluidos en este repo — solo los CSVs ya procesados por
> `bc3_parse.py`. Ver `docs/bitacora-proyecto.md` para el detalle de
> procedencia y verificación de autenticidad del dataset.

## Decisiones metodológicas clave

- **Multi-etiqueta, no single-label**: un email puede tener varios actos de
  habla a la vez (61% de los emails los tiene). Ver `docs/bitacora-proyecto.md`
  sección 1 para la justificación completa.
- **Split a nivel de hilo** (no de email individual), para evitar fuga de
  información entre train/val/test.
- **Split fijo** (no cross-validation) por coste computacional en las etapas
  posteriores del pipeline (BERT+LoRA + LLM de resumen) — limitación
  documentada explícitamente en el informe final.

## Estado del proyecto

Ver `docs/bitacora-proyecto.md` para el registro completo y actualizado de
cada etapa (completadas, en curso, pendientes).

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
cd src
python bc3_baseline_tfidf.py
```

## Autor

Mister — 300699324, Data Science and Engineering, Victoria University of Wellington.
