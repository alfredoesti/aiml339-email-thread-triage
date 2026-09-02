# Email Thread Triage — Contexto del proyecto (AIML339)

Este archivo se carga automáticamente al iniciar Claude Code en este repo.
Contiene el contexto esencial para no tener que re-explicar el proyecto
en cada sesión nueva. El registro completo y detallado está en
`docs/bitacora-proyecto.md` — léelo si necesitas más profundidad en
alguna decisión.

## Qué es este proyecto

Pregunta de investigación: ¿clasificar cada email de un hilo por su acto
de habla (Request/Propose/Commit/Meeting/Subjective/Informative) antes de
resumir el hilo con un LLM produce mejores resúmenes que resumir el texto
crudo directamente?

Pipeline de dos etapas:
1. Clasificador multi-etiqueta (BERT+LoRA, con TF-IDF+OneVsRestClassifier
   como baseline) sobre el corpus BC3.
2. Resumen vía LLM (OpenAI API), comparando Condición A (texto crudo) vs
   Condición B (texto + etiquetas predichas), evaluado con ROUGE + revisión
   manual.
3. Demo opcional no evaluada: aplicación a una muestra anonimizada del
   Gmail personal del autor.

## Decisión metodológica clave (importante, no la repitas por error)

La clasificación es **multi-etiqueta**, no single-label. Un email puede
tener varias categorías activas a la vez (61% de los emails las tiene).
Por eso:
- Las 6 columnas `label_Request`, `label_Propose`, `label_Commit`,
  `label_Meeting`, `label_Subjective`, `label_Informative` son binarias
  independientes, no una sola clase categórica.
- El baseline usa `OneVsRestClassifier`, no regresión logística simple.
- BERT+LoRA (pendiente) debe usar capa de salida sigmoide + BCE loss,
  no softmax + cross-entropy.
- Las métricas son F1 micro/macro y Hamming loss, estándar en
  multi-etiqueta — no accuracy simple.

Este cambio se apartó del Plan report original (que hablaba de
single-label) y ya fue comunicado como decisión consciente.

## Estado actual (ver docs/bitacora-proyecto.md para el detalle completo)

- ✅ Tarea 1: datos BC3 parseados, split por hilo (28/6/6 hilos →
  187/37/37 emails), verificación de autenticidad del dataset completa.
- ✅ Tarea 2: baseline TF-IDF + OneVsRestClassifier entrenado y evaluado.
  Test: Micro F1 0.56, Macro F1 0.41. Se decidió NO usar cross-validation
  (multiplicaría x5 el coste de Colab/API en todas las etapas del
  pipeline) — split fijo, limitación documentada en el informe.
- ⬜ Tarea 3: clasificador BERT+LoRA multi-etiqueta — siguiente paso.
- ⬜ Tareas 4-12: evaluación del clasificador, pipeline de resumen A/B,
  ROUGE, demo Gmail, informe final, presentación.

## Reglas de reproducibilidad (no negociables)

- El split train/val/test es **por hilo completo** (`data/bc3_split.json`,
  semilla fija), nunca por email individual — evita fuga de contexto
  compartido entre splits.
- Cualquier vectorizador/tokenizador se ajusta (`fit`) SOLO con train;
  val/test usan `transform` únicamente.
- El split debe ser idéntico entre baseline y BERT+LoRA para que la
  comparación sea válida — no lo renegocies por modelo.

## Estilo de trabajo preferido

- Explicar cada bloque de código paso a paso: qué hace, por qué esa
  decisión, qué observar — pausando para confirmación antes de seguir.
  Nada de bloques grandes de código sin explicar primero.
- Preguntas conceptuales de "por qué" antes de aceptar una decisión de
  implementación son bienvenidas y hay que responderlas con detalle.
- Prosa fluida (no listas de viñetas) en las secciones de metodología
  del informe final; estilo conciso y directo.
- Después de cerrar cada tarea, escribir juntos una mini-memoria (qué se
  hizo, dificultades, observaciones) y añadirla a
  `docs/bitacora-proyecto.md`, para tener el informe final medio escrito
  al terminar el proyecto.
- **Antes de terminar cualquier sesión de trabajo** en la que se haya
  avanzado, completado o modificado algo relevante del proyecto —
  aunque el usuario no lo pida explícitamente — proponer proactivamente
  una entrada de bitácora resumiendo lo hecho en esa sesión (qué se
  hizo, decisiones tomadas, resultados, dificultades) y ofrecer
  añadirla a `docs/bitacora-proyecto.md` antes de cerrar. Si la sesión
  no tocó nada del proyecto (solo dudas puntuales, exploración sin
  cambios), no hace falta.

## Estructura del repo

```
src/            Código fuente reutilizable (parser, baseline, futuros
                scripts de entrenamiento)
data/           CSVs procesados de BC3 (no el XML crudo — ver .gitignore)
results/        Predicciones y métricas de evaluación
docs/           Bitácora detallada del proyecto
notebooks/      Para trabajo exploratorio y entrenamiento en Colab
                (BERT+LoRA, resumen, ROUGE) — .py para código estable,
                .ipynb para exploración/visualización iterativa
```

## Idioma

Conversación en español; código, comentarios de código y el informe
final en inglés (formato IEEE de dos columnas).
