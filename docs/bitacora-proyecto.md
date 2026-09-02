## 1. Datos: BC3 preparado (split por hilo)
**Estado:** Completado (31 ago–1 sep 2026, revisado a multi-etiqueta el 1 sep)

**Qué se hizo:**
- Fuente oficial de descarga de BC3 (UBC, `cs.ubc.ca/labs/lci/bc3/download.html`) caída (error de servidor, no de licencia). Se envió petición directa a Giuseppe Carenini, Gabriel Murray y Shafiq Joty para obtener el acceso oficial; pendiente de respuesta.
- Mientras tanto, se localizó una copia de los ficheros reales de BC3 (`corpus.xml`, `annotation.xml`) en dos repositorios públicos de GitHub independientes: `dailykirt/ML_Enron_email_summary` (capstone que usa BC3 como parte de su trabajo) y `RemedyHealthcare/bc3` (mirror dedicado solo al corpus, con `corpus.dtd`/`annotation.dtd`/README oficial).
- Se escribió un parser en Python (`bc3_parse.py`) que lee `corpus.xml` (hilos/emails/frases) y `annotation.xml` (etiquetas de acto de habla por frase y anotador + resúmenes de referencia), y genera el split train/val/test **a nivel de hilo completo** (semilla fija, 70/15/15).

**Verificación de autenticidad del dataset (importante para Ética y Privacidad del informe):**
- Comparación byte a byte (MD5) de `corpus.xml` y `annotation.xml` entre los dos repos de GitHub, subidos por autores distintos en años distintos: **hashes idénticos** en ambos casos. Prácticamente descarta manipulación independiente.
- Los DTDs oficiales de `RemedyHealthcare/bc3` (`corpus.dtd`, `annotation.dtd`) coinciden exactamente con la estructura parseada (thread > listno/name/DOC; DOC > Received/From/To/Cc?/Subject/Text; labels > meta*/prop*/meet*/req*/subj*/cmt*).
- Confirmación independiente de tercera fuente: la **tesis de máster original de Jan Ulrich** ("Supervised Machine Learning for Email Thread Summarization", UBC, 2008), alojada oficialmente y de descarga libre en el repositorio institucional de UBC (open.library.ubc.ca/cIRcle, DOI 10.14288/1.0051210). Confirma textualmente los 40 hilos, el origen de las categorías de acto de habla (inspiradas en Carvalho & Cohen: Propose/Request/Commit/Deliver/Meeting), y que se añadieron "Meta Sentences" y "Subjective Sentences" como anotaciones aparte, con 3 anotadores por hilo.
- Conclusión: triple verificación (repo A = repo B byte a byte; ambos coinciden con el DTD oficial; y la propia tesis del autor describe la misma metodología y cifras). No se detectó ninguna anomalía ni señal de dataset modificado o falso.
- Aclaración sobre licencias: el corpus está bajo **Creative Commons Attribution-Share Alike 3.0** (no MIT). La licencia MIT que declara el repo `dailykirt` aplica solo a su propio código, no relicencia los datos de terceros que redistribuye. CC-BY-SA no exige ningún trámite de "obtención" para uso académico, solo atribución y compartir bajo la misma licencia si se redistribuye.
- Fuente adicional para citar en Background/Metodología: Ulrich, J. (2008). *Supervised Machine Learning for Email Thread Summarization* (Master's thesis, University of British Columbia). https://dx.doi.org/10.14288/1.0051210 — mejor referencia que el paper corto de AAAI08 para explicar la metodología de anotación en detalle.

**Decisión metodológica revisada (1 sep 2026): de single-label a multi-etiqueta + acuerdo entre anotadores**

Primera versión (descartada): se asignaba a cada email UNA sola etiqueta, tomando el acto de habla más frecuente entre todas las frases/anotadores de ese email (mayoría de voto simple), con "Informative" como categoría residual inventada para emails sin ninguna etiqueta.

Problema detectado al revisarlo: en el corpus, las etiquetas de acto de habla viven a nivel de FRASE, no de email; un email puede tener frases con varias etiquetas distintas (no excluyentes) y, además, 2–3 anotadores humanos etiquetan cada hilo por separado y no siempre coinciden. Reducir todo eso a una sola etiqueta por mayoría de voto tira información real por la borda (ejemplo real: hilo `007-7484738`, email 4, un anotador marcó Propose+Meeting+Subjective y otro marcó Propose+Meeting+Request+Subjective en el mismo email).

Decisión tomada, con recomendación explícita: 
1. **Multi-etiqueta por email**: la etiqueta final de un email es ahora la UNIÓN de todas las categorías que cualquier anotador marcó en cualquiera de sus frases (Request/Propose/Commit/Meeting/Subjective/Informative como columnas binarias independientes, no una sola clase). Justificación: es más fiel al dato, no pierde información, y es perfectamente factible técnicamente — BERT+LoRA solo necesita cambiar la capa de salida a sigmoide+BCE loss por clase en vez de softmax+cross-entropy, y el baseline usa `OneVsRestClassifier` de scikit-learn en vez de regresión logística simple. El F1 se calcula igual, por etiqueta y promediado (micro/macro F1), estándar en clasificación multi-etiqueta.
2. **No se filtra ni se descarta nada por desacuerdo entre anotadores** (el corpus ya es pequeño, perder más datos sería peor que el ruido). En su lugar, se añadió una métrica de **acuerdo entre anotadores por email** (Jaccard medio entre cada par de anotadores sobre el conjunto de categorías que usaron en ese email), guardada como columna `annotator_agreement`, para reportarla como dato descriptivo en el EDA/Ética del informe, no para limpiar el dataset.
3. Se mantiene una columna auxiliar `primary_label` (una sola etiqueta "representativa") por si hace falta una vista simplificada, elegida por un orden de prioridad basado en accionabilidad para resumen (Commit > Request > Propose > Meeting > Subjective > Informative), no por frecuencia estadística — pero el análisis y el clasificador principal usan las columnas multi-etiqueta.

**Nota de gobernanza del proyecto:** este cambio (single-label → multi-etiqueta) se aparta de la descripción original del Plan report ("labels each email... in one of a small set of categories"). Es una mejora metodológica defendible y el briefing permite que el plan cambie, pero **hay que mencionarlo explícitamente en el próximo check-in con el supervisor** para dejar constancia de que es una decisión consciente.

**Resultado (EDA, versión multi-etiqueta):**
- 40 hilos, 261 emails, 3222 frases (sin cambios).
- Nº de emails en los que aparece cada etiqueta (un email puede contar en varias a la vez): Subjective 194, Request 117, Propose 88, Meeting 85, Commit 63, Informative 27.
- 160 de 261 emails (61%) tienen más de una etiqueta activa a la vez — confirma que el enfoque multi-etiqueta era necesario, no cosmético.
- Acuerdo entre anotadores (Jaccard medio, solo emails con ≥2 anotadores, 200/261 emails): media 0.622, mediana 0.578. 54 emails con acuerdo perfecto (1.0), 7 emails con acuerdo nulo (0.0, desacuerdo total) — dato a comentar en el informe como variabilidad esperada de anotación humana subjetiva.
- Split por hilos: 28 train / 6 val / 6 test → 187 / 37 / 37 emails (sin cambios, sigue a nivel de hilo).

**Dificultades / observaciones para el informe:**
- El corpus real no tiene las 4 categorías "request/commitment/informative/other" descritas de forma simplificada en el check-in; son 5 actos de habla reales (Request, Propose, Commit, Meeting, Subjective) + "Informative", categoría residual que no existe en el corpus original y que hay que justificar como decisión propia.
- Con multi-etiqueta, "Commit" pasa de solo 7 emails (bajo el criterio de mayoría antiguo) a 63 emails en los que aparece junto a otras etiquetas — el desbalance de clases sigue existiendo pero es menos extremo de lo que parecía con single-label. Aun así, seguirá siendo la clase más difícil de aprender bien.
- Origen de los datos: mientras no se confirme el acceso oficial vía los autores, el dataset usado proviene de mirrors de terceros en GitHub (verificados exhaustivamente, ver arriba), no de la descarga oficial de UBC. Dejarlo explícito en Ética y Privacidad del informe y sustituir por el canal oficial en cuanto respondan los autores.
- Limitación reconocida: la métrica de acuerdo entre anotadores usada (Jaccard sobre conjuntos de categorías por email) es una simplificación — no distingue si el desacuerdo es sobre qué frase exacta llevaba la etiqueta o sobre la categoría en sí, y no es una métrica estándar de la literatura (como Cohen's Kappa) pero es adecuada para un EDA descriptivo.

## 2. Baseline: TF-IDF + regresión logística
**Estado:** Completado (2 sep 2026)

**Qué se hizo:**
- Script `bc3_baseline_tfidf.py`: TF-IDF (unigramas+bigramas, max_features=5000,
  min_df=2, stopwords inglés) + OneVsRestClassifier(LogisticRegression(class_weight='balanced')),
  multi-etiqueta sobre las 6 columnas label_*.
- Vectorizador ajustado (fit_transform) SOLO con train; val/test solo transform,
  para evitar fuga de información (mismo principio que el split por hilo).
- Se reutiliza el split fijo de bc3_split.json (187/37/37 emails, 28/6/6 hilos).

**Resultado (test, 37 emails):**
- Micro F1: 0.56 | Macro F1: 0.41 | Hamming loss: 0.35
- Por etiqueta: Subjective fuerte (F1 0.86), Meeting decente (0.64),
  Request/Commit mediocres (~0.40), Propose floja (0.15),
  Informative sin soporte en test (0 casos, dataset muy pequeño).
- Caída fuerte train→val→test (Micro F1 0.88→0.64→0.56): esperado dado
  el tamaño del dataset, no es un bug del código.

**Decisión tomada sobre el split (discutida y cerrada):**
Se consideró cross-validation por hilo para tener métricas más robustas,
pero se descartó: encarecería x5 el coste de Colab y de la API del LLM
en TODAS las etapas del pipeline (BERT+LoRA + las dos condiciones de
resumen + ROUGE), no solo en el baseline, y habría que mantenerlo
comparable en ambos clasificadores. Se mantiene el split fijo y esta
limitación se documentará explícitamente en el informe (sección de
Limitaciones/Metodología).

**Pendiente de decidir (no bloqueante):** no se probaron variantes de
hiperparámetros de TF-IDF (max_features, con/sin bigramas) — queda como
posible ajuste fino si sobra tiempo, no es prioritario para el baseline.

**Guardado:** bc3_baseline_tfidf.py, bc3_baseline_test_predictions.csv
(predicciones de test por etiqueta, para comparar con BERT+LoRA en Tarea 3).

## 3. Clasificador: BERT + LoRA
**Estado:** Notebook escrito, pendiente de ejecución en Colab (2 sep 2026)

**Qué se hizo:**
- Notebook `notebooks/bc3_bert_lora.ipynb` diseñado y escrito completamente, listo para ejecutar en Google Colab con GPU.
- El notebook tiene 5 bloques: configuración y carga de datos, tokenización y Dataset, modelo BERT+LoRA, bucle de entrenamiento, evaluación y guardado de predicciones.
- La capa de salida usa `problem_type="multi_label_classification"` de HuggingFace, que aplica sigmoide + BCE loss por etiqueta de forma independiente — correcto para multi-etiqueta, sin softmax.
- LoRA configurado con `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, sobre las matrices `query` y `value` de las capas de atención de `bert-base-uncased`. Parámetros entrenables estimados: ~2% del total.
- El bucle guarda el mejor checkpoint según Micro F1 en validación y lo restaura antes de la evaluación final, para evitar quedarse con un modelo sobreajustado.
- Las predicciones de test se guardan en `bc3_bert_lora_test_predictions.csv` con el mismo esquema de columnas que el baseline, para comparación directa.

**Dificultades / observaciones para el informe:**
- El notebook está diseñado para Colab pero aún no se ha ejecutado — los resultados reales (Micro F1, Macro F1, Hamming loss en test) quedan pendientes de la ejecución. Hasta entonces no hay cifras que comparar con el baseline (Micro F1 0.56, Macro F1 0.41).
- `DATA_PATH` en el Bloque 1 tiene un comentario explícito recordando ajustar la ruta cuando se ejecute en Colab (subir el CSV o montar Google Drive).

**Pendiente inmediato:** ejecutar el notebook en Colab, anotar los resultados aquí, y compararlos con el baseline en la Tarea 4.

## 4. Evaluación del clasificador (BERT+LoRA vs baseline)
_Pendiente_

## 5. Resumen — Condición A (texto crudo)
_Pendiente_

## 6. Resumen — Condición B (con contexto de clasificación)
_Pendiente_

## 7. Evaluación de resúmenes: ROUGE + revisión manual
_Pendiente_

## 8. Demo: Gmail personal anonimizado
_Pendiente_

## 9. Fechas confirmadas del curso
_Pendiente_

## 10. Check-ins restantes
_Pendiente — recordar mencionar el cambio a multi-etiqueta en el próximo check-in con el supervisor._

## 11. Informe final
_Pendiente_

## 12. Presentación final
_Pendiente_
