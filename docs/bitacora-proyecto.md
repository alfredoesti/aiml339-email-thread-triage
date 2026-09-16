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

**Ejecución en Colab y resultados finales (3 sep 2026):**

- Notebook ejecutado en Google Colab con GPU T4.
- Se detectaron y corrigieron dos problemas durante la ejecución: los nombres de las capas de LoRA no coincidían (`target_modules` ampliado a `["query", "key", "value", "dense"]`, subiendo los parámetros entrenables de 0.27% a 1.21%), y el desbalance de clases no estaba controlado (se añadió BCE con `pos_weight` por etiqueta calculado sobre train).
- Se añadieron `torch.backends.cudnn.deterministic = True` y `torch.backends.cudnn.benchmark = False` para reducir la variabilidad entre ejecuciones.
- Entrenamiento final: 5 épocas, mejor checkpoint en época 5 (val Micro F1: 0.684).

**Resultados en test (37 emails):**
- Micro F1: 0.637 | Macro F1: 0.525 | Hamming loss: 0.369
- Por etiqueta: Subjective (0.84), Request (0.60), Meeting (0.59), Commit (0.60), Propose (0.51), Informative (0.00 — sin soporte en test).
- BERT+LoRA supera al baseline TF-IDF en Micro F1 (0.637 vs 0.56) y Macro F1 (0.525 vs 0.41).

**Dificultades / observaciones para el informe:**
- Alta variabilidad entre ejecuciones (Micro F1 entre 0.56 y 0.64 en distintas ejecuciones con misma configuración) — limitación real del dataset pequeño (187 emails de train). Documentar como limitación en el informe.
- El modelo tiende a recall alto y precisión más baja (sesgo de los pesos por clase), lo que significa que predice de más antes que de menos. Razonable para un dataset pequeño pero a mencionar en el análisis.
- Informative sigue en F1 0.00 — no hay ningún caso en el set de test, igual que en el baseline. No es un fallo del modelo sino del split.

**Guardado:** `notebooks/bc3_bert_lora.ipynb` (con outputs), `results/bc3_bert_lora_test_predictions.csv`.

## 4. Evaluación del clasificador (BERT+LoRA vs baseline)
**Estado:** Completado (3 sep 2026)

**Qué se hizo:**
- Notebook `notebooks/bc3_comparison.ipynb` con tabla resumen de métricas globales, tabla por etiqueta con delta, gráfico de barras comparativo e interpretación escrita.
- Gráfico guardado en `results/bc3_comparison_f1.png`.

**Conclusión principal:** BERT+LoRA supera al baseline en Micro F1 (0.637 vs 0.560) y especialmente en Macro F1 (0.525 vs 0.410). La mejora es mayor en las etiquetas que requieren comprensión semántica (Propose +0.36, Commit +0.20, Request +0.19). Meeting es la única etiqueta donde el baseline gana ligeramente (0.64 vs 0.59), porque sus señales léxicas son muy claras. Informative queda en 0.00 en ambos modelos por ausencia de casos en test.

## 5. Resumen — Condición A (texto crudo)
**Estado:** Completado (14 sep 2026)

**Qué se hizo:**
- Script `src/bc3_summarize.py`: carga los 6 hilos de test, construye un prompt con el texto crudo de cada email (encabezado con From/Subject + cuerpo) y llama a Gemini 3.6 Flash para obtener un resumen de 150-200 palabras en prosa.
- La instrucción de sistema es idéntica para ambas condiciones para que la única variable sea el contenido del prompt.
- Resultados guardados en `results/bc3_summaries_condition_a.csv` (columnas: listno, thread_name, n_emails, summary_a).

**Decisión de API:** se usó Gemini 3.6 Flash (google-genai, capa gratuita) en lugar de OpenAI, que era la opción original del plan. Razón: el autor ya tenía clave de Google AI Studio activa. El cambio no afecta la validez del experimento — lo que se evalúa es el efecto de añadir etiquetas al prompt, no el modelo en sí. Se documenta como decisión consciente.

**Nota técnica:** `gemini-2.0-flash` fue deprecado durante el desarrollo; se migró primero a `gemini-3.6-flash` y después a `gemini-1.5-flash` (límite del free tier: 1.500 req/día vs. 20 de 3.6-flash). La clave de API se carga desde un fichero `.env` en la raíz del repo (excluido de git vía `.gitignore`).

**Actualización de prompts (16 sep 2026):**
El system prompt original era genérico ("cover: main topic, key discussion points, any decisions, any action items"). Se reemplazó por un prompt con orden de prioridad explícito alineado con el estilo de los anotadores del BC3: (1) tema y estado, (2) decisiones y compromisos con nombres exactos, (3) action items con responsable y plazo, (4) resto solo si influyó en una decisión. Se añadió la instrucción "do not invent names, dates, or details" para evitar alucinaciones. El system prompt es **idéntico** en Condición A y B para que la única variable experimental sea la presencia de etiquetas.

El bloque de etiquetas de Condición B también se mejoró: en lugar de "usa las etiquetas para entender el rol", ahora da instrucciones concretas por etiqueta (Commit → inclúyelo en decisiones; Subjective → solo si cambió el resultado; etc.) con la advertencia de usarlas como señales, no como guión literal.

## 6. Resumen — Condición B (con contexto de clasificación)
**Estado:** Completado (14 sep 2026)

**Qué se hizo:**
- Mismo pipeline que Condición A, pero el prompt de cada email incluye las etiquetas predichas por BERT+LoRA (`results/bc3_bert_lora_test_predictions.csv`): se añade `| Labels: Request, Commit` al encabezado de cada email, junto con una breve explicación del significado de cada categoría al inicio del prompt.
- Resultados guardados en `results/bc3_summaries_condition_b.csv` (columnas: listno, thread_name, n_emails, summary_b).
- Notebook de visualización `notebooks/bc3_summarization.ipynb`: carga los dos CSVs y muestra los resúmenes A y B lado a lado por hilo, sin lógica de generación (no requiere API key).

**Observación preliminar (cualitativa):** los resúmenes de Condición B tienden a ser ligeramente más largos (media ~1175 chars vs ~1175 chars en A — diferencia mínima en longitud), pero la evaluación cuantitativa queda para la Tarea 7 con ROUGE.

## 7. Evaluación de resúmenes: ROUGE + revisión manual
**Estado:** ROUGE automático completado (14 sep 2026); revisión manual pendiente.

**Qué se hizo:**
- Script `src/bc3_rouge.py`: calcula ROUGE-1, ROUGE-2 y ROUGE-L (F1, con stemming) para los resúmenes generados en ambas condiciones, contra los resúmenes de referencia del corpus BC3.
- Múltiples anotadores (3 por hilo): se computa ROUGE contra cada anotador por separado y se guarda el máximo de los tres (estándar DUC/TAC y del propio paper de BC3). Justificación: los anotadores humanos también difieren entre sí; penalizar al modelo por coincidir con solo uno de ellos sería injusto.
- Resultados guardados en `results/bc3_rouge_scores.csv`. Gráfico en `results/bc3_rouge_comparison.png`.
- Notebook `notebooks/bc3_rouge.ipynb`: explicación de ROUGE, tabla por hilo con winner A/B, gráfico de barras comparativo, y análisis interpretativo escrito con los números reales.

**Resultados — prompts originales (14 sep 2026):**
- ROUGE-1: A=0.3928, B=0.3990 (+0.006)
- ROUGE-2: A=0.1027, B=0.1124 (+0.010)
- ROUGE-L: A=0.2008, B=0.2239 (+0.023)

**Resultados — prompts mejorados (17 sep 2026, modelo gemini-3.5-flash):**
- ROUGE-1: A=0.4094, B=0.4309 (+0.0215)
- ROUGE-2: A=0.1113, B=0.1332 (+0.0219)
- ROUGE-L: A=0.2200, B=0.2272 (+0.0072)

Resultados por hilo (prompts mejorados):

| Hilo | R1-A | R1-B | R2-A | R2-B | RL-A | RL-B |
|------|------|------|------|------|------|------|
| 059 Phone connection | 0.451 | 0.487 | 0.137 | 0.150 | 0.229 | 0.237 |
| 061 Non-geek guidelines | 0.459 | 0.480 | 0.144 | 0.142 | 0.269 | 0.232 |
| 063 WAI-ER-IG Welcome | 0.429 | 0.443 | 0.144 | 0.179 | 0.217 | 0.265 |
| 067 Graphics/Web Design | 0.364 | 0.348 | 0.050 | 0.078 | 0.171 | 0.192 |
| 015 SWADEurope postcard | 0.350 | 0.406 | 0.075 | 0.086 | 0.193 | 0.194 |
| 058 Next face to face | 0.405 | 0.422 | 0.118 | 0.164 | 0.241 | 0.244 |

**Conclusión principal:** Los prompts mejorados suben ambas condiciones (~+0.017 en ROUGE-1 de media). Condición B sigue superando a A en las tres métricas globales. B gana a A en ROUGE-1 en 5 de 6 hilos y en ROUGE-2 en todos los hilos. El único hilo donde A supera a B en ROUGE-1 es `067` (discusión técnica/subjetiva sobre diseño web accesible, donde las etiquetas de acto de habla añaden menos valor estructural). En ROUGE-L la mejora de B sobre A es más modesta (+0.007) que en los prompts originales (+0.023), aunque los valores absolutos son más altos en ambas condiciones.

**Evaluación multi-referencia (ya implementada desde el inicio):** ROUGE se calcula contra cada anotador por separado y se guarda el **máximo** de los tres (estándar DUC/TAC y del propio paper de BC3). Esto es correcto: penalizar al modelo por coincidir con solo uno de tres anotadores sería injusto dado que los propios anotadores humanos difieren entre sí. El código en `bc3_rouge.py` (líneas 57-60) implementa `max(s[metric].fmeasure for s in scores_per_ref)`.

**Dificultades / observaciones para el informe:**
- Los valores absolutos de ROUGE (~0.41 ROUGE-1) son normales para resúmenes abstractivos contra referencias de estilo extractivo. No indican mala calidad — simplemente que ROUGE penaliza la paráfrasis.
- La calidad de Condición B depende de la calidad de las predicciones de BERT+LoRA (Micro F1=0.637). Errores de clasificación se propagan al prompt y pueden desorientar al modelo.
- Limitación principal: 6 hilos de test son insuficientes para extraer conclusiones robustas. El efecto observado es consistente con la hipótesis pero no concluyente.
- **Nota técnica (17 sep 2026):** `gemini-1.5-flash` fue deprecado por Google entre sesiones. Se migró a `gemini-3.5-flash` para la re-ejecución con prompts mejorados. El cambio de modelo puede introducir variación en los resultados, aunque la dirección del efecto (B > A) se mantiene.

## 8. Demo: Gmail Triage Agent
**Estado:** Primera versión funcional ejecutada en bandeja real (16 sep 2026). Pendiente de mejoras y configuración del Task Scheduler.

**Cambio de enfoque respecto al plan original (importante):** la demo se rediseñó completamente. En lugar de anonimizar hilos de Gmail y resumirlos con el pipeline BC3, se implementó un **agente de triaje automático** que clasifica cada hilo de la bandeja de entrada en 5 categorías (Universidad / Trabajo / Personal / Spam / Otro) y aplica la etiqueta `Triage/*` correspondiente directamente en Gmail. Razones: más útil en la práctica, más demostrable en tiempo real ante el supervisor, y mantiene el espíritu de "aplicación a datos reales" que justificaba la tarea.

**Qué se implementó:**
- `src/gmail_auth.py`: módulo OAuth2 reutilizable con token persistente (no vuelve a pedir permiso tras la primera autorización). Credenciales en `credentials.json` (gitignoreado). Proyecto Google Cloud: `gen-lang-client-0886504393` ("Default Gemini Project"), misma cuenta que la Gemini API key. Gmail API habilitada, pantalla de consentimiento tipo Externo, usuario de prueba `alfreditoestirado@gmail.com`.
- `src/gmail_triage_agent.py`: agente principal. Lógica incremental: primera ejecución procesa los últimos 10 hilos; ejecuciones siguientes solo procesan hilos nuevos desde el último timestamp (`results/gmail_triage/last_run.txt`). Clasificación via Gemini. Log acumulativo en `results/gmail_triage/triage_log.csv`.
- `scripts/setup_gmail_labels.py`: script one-time para crear las etiquetas `Triage/*` en Gmail.
- `scripts/run_triage.py`: punto de entrada para el Task Scheduler. Incluye flag `--reset` para reiniciar el estado y opción `--help`.
- `notebooks/gmail_triage_demo.ipynb`: notebook de visualización para la demo (lee el log, muestra distribución por categoría y últimos hilos clasificados).

**Categorías finales (tras ajustes durante la primera ejecución):**
- `Triage/Universidad`, `Triage/Trabajo`, `Triage/Personal`, `Triage/Spam`, `Triage/Otro`.
- Se eliminó `Triage/Newsletters`: la única newsletter real es Bloomberg, que va a Personal. Todo lo demás no solicitado (Uber Eats, Ticketmaster, Chipotle, Subway, etc.) va a Spam.

**Modelo usado:** `gemini-1.5-flash` (1.500 peticiones/día en el free tier). Se descartó `gemini-3.6-flash` porque su free tier solo permite 20 peticiones/día, lo cual se agotó durante las pruebas de ajuste del primer día.

**Primera ejecución real (resultados):**
- 10 hilos de la bandeja personal clasificados correctamente en su mayoría.
- Bloomberg ("Bond market tumble", "The AI boom continues") → Personal ✓
- Subway, Hot Wheels, Shake Shack, Chipotle → Spam ✓
- Oferta de Data Scientist CaixaBank/Accenture → Trabajo ✓

**Dificultades / pendiente de mejorar:**
- El límite de 5 peticiones/minuto del free tier requiere 13 segundos de pausa entre llamadas — el run de 10 hilos tarda ~2 minutos. Para el Task Scheduler no es un problema, pero para una demo en vivo hay que anticiparlo.
- La clasificación no es perfecta todavía: algunos correos promocionales ambiguos caen en "Otro" en lugar de "Spam". Pendiente de afinar el prompt o añadir ejemplos adicionales de remitentes concretos.
- El agente todavía **no está configurado como tarea programada** en Windows Task Scheduler — queda pendiente para la siguiente sesión.
- El notebook de demo `notebooks/gmail_triage_demo.ipynb` no tiene outputs todavía (no se ejecutó con datos reales de forma completa). Se completará cuando el agente tenga un run limpio.

**Archivos clave:**
- Código: `src/gmail_auth.py`, `src/gmail_triage_agent.py`, `scripts/run_triage.py`, `scripts/setup_gmail_labels.py`
- Resultados: `results/gmail_triage/triage_log.csv`, `results/gmail_triage/agent.log`
- Demo: `notebooks/gmail_triage_demo.ipynb`

---

## RESUMEN PARA CHECK-IN (16 sep 2026) — Avance desde Tarea 4

El último check-in cubrió hasta la Tarea 4 (comparación baseline vs BERT+LoRA). Desde entonces se han completado las Tareas 5, 6, 7 y avanzado en la 8:

**Tarea 5 — Resumen LLM Condición A (texto crudo):**
Script `src/bc3_summarize.py`. Se generaron resúmenes de 150-200 palabras para los 6 hilos de test del corpus BC3 usando solo el texto crudo como input al LLM. Modelo: Gemini (Google AI Studio, free tier). Resultados en `results/bc3_summaries_condition_a.csv`.

**Tarea 6 — Resumen LLM Condición B (texto + etiquetas de acto de habla):**
Mismo pipeline que A, pero el prompt incluye las etiquetas predichas por BERT+LoRA para cada email (`results/bc3_bert_lora_test_predictions.csv`): se añade al encabezado de cada email `| Labels: Request, Commit` junto con una explicación del significado de cada categoría. Resultados en `results/bc3_summaries_condition_b.csv`. Visualización lado a lado en `notebooks/bc3_summarization.ipynb`.

**Tarea 7 — Evaluación ROUGE (Condición A vs B):**
Script `src/bc3_rouge.py`. ROUGE-1, ROUGE-2 y ROUGE-L calculados contra los resúmenes de referencia del corpus BC3 (máximo de los 3 anotadores, estándar DUC/TAC). Resultados finales (prompts mejorados, 17 sep 2026):
- ROUGE-1: A=0.4094 vs B=0.4309 (+0.0215)
- ROUGE-2: A=0.1113 vs B=0.1332 (+0.0219)
- ROUGE-L: A=0.2200 vs B=0.2272 (+0.0072)
Conclusión: Condición B supera a A en las tres métricas. B gana en 5 de 6 hilos en ROUGE-1 y en todos en ROUGE-2. Gráfico en `results/bc3_rouge_comparison.png`. Notebook con análisis en `notebooks/bc3_rouge.ipynb`.

**Tarea 8 — Demo Gmail Triage Agent (en progreso):**
Agente funcional que clasifica correos reales de la bandeja personal en 5 categorías y aplica etiquetas automáticamente en Gmail. Primera ejecución completada sobre 10 hilos reales. Pendiente: Task Scheduler, mejora del prompt de clasificación, notebook de demo con outputs reales.

**Limitación importante a mencionar en el check-in:** con solo 6 hilos de test, las diferencias ROUGE entre A y B son consistentes con la hipótesis pero no estadísticamente significativas. El resultado es prometedor, no concluyente.

## 9. Fechas confirmadas del curso
_Pendiente_

## 10. Check-ins restantes
_Pendiente — recordar mencionar el cambio a multi-etiqueta en el próximo check-in con el supervisor._

## 11. Informe final
_Pendiente_

## 12. Presentación final
_Pendiente_
