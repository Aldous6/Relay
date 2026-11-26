Relay v7 · evidencia local por similitud. Informe de desarrollo reutilizado.

La primaria alcanzó el piso de calidad, pero no consiguió menor costo histórico que BERT101; el objetivo competitivo no se alcanzó en esta ejecución. Frente a BERT101, el costo fue 4.46 % mayor. Esto no confirma superioridad, no inferioridad ni ahorro neto en producción.

La primaria obtuvo 3,473/4,406 aciertos (78.82 %) con costo histórico 3.258349 por 1,000 prompts. Éxito significa score guardado ≥ 0.75; estos son porcentajes absolutos de éxito, no PGR ni CPT del paper.

Se conservaron 20,368 filas de ajuste, 4,383 de calibración y 4,406 de validación del train exterior. El test histórico consumido quedó fuera de esta ejecución. La validación intervino antes en v2–v6 y volvió a usarse para seleccionar: sus métricas e intervalos son exploratorios.

Los candidatos mantienen las identidades históricas gpt-4-1106-preview y mistralai/mixtral-8x7b-chat. El resultado no permite sustituirlos por otros modelos ni extrapolarlo a tareas completas de agentes sin evaluación nueva.

V7 combina un predictor global multiclase con los 64 vecinos de mayor similitud coseno de una representación híbrida de TF-IDF y 14 atributos del prompt. Vocabulario, escalado, predictor global, banco de vecinos y costos se ajustan únicamente en fit. Los objetivos son strong_only, weak_only y equal; equal conserva tanto ambos aciertos como ambos fallos. El combinador logístico se ajusta solo en calibración. Las filas de calibración y validación nunca se incorporan al banco de vecinos.

La idea está inspirada en la ponderación por similitud del [paper RouteLLM](https://arxiv.org/abs/2406.18665). V7 es una adaptación propia con top-k léxico y outcomes umbralizados; no reproduce el SW semántico con Bradley–Terry del artículo. La referencia BERT conserva el checkpoint publicado de [RouteLLM](https://github.com/lm-sys/RouteLLM) y sus predicciones archivadas, con selección local de umbral. Su supervisión externa difiere de los outcomes locales de Relay. No se evaluaron MF ni SW oficiales ni se igualaron los regímenes de entrenamiento.

La procedencia de BERT fija el checkpoint routellm/bert_gpt4_augmented, revisión 86237e3df400762178ea98379477b8296e66d5e4, con máximo 512 tokens y truncamiento activado. V7 procesa el texto completo con el vocabulario aprendido; esto no equivale a comprensión semántica completa. La posible contaminación del checkpoint externo respecto a benchmarks no está resuelta.

Se predeclararon una primaria con vecinos y una ablación global sin vecinos. Cada una usa 101 cuantiles de la puntuación de validación calculados sin outcomes. La puntuación es la ganancia prevista de éxito dividida por la diferencia positiva de costo previsto; se elige el fuerte únicamente cuando supera el corte. Las acciones se persistieron antes de calcular métricas. El corte no es una probabilidad.

BERT101 se seleccionó por menor costo entre puntos con éxito al menos igual al fuerte fijo. El piso exigido a v7 fue 78.80 %, el máximo entre BERT101 y fuerte fijo. Se eligió el punto de menor costo que alcanzaba ese piso, sin usar intervalos para seleccionar. Alcanzar el piso y ser más barato que BERT son resultados separados. El ensamble v4 se reeligió con ese piso; v6 y su BERT de 16 puntos conservan las políticas archivadas. La ablación global no sustituye a la primaria aunque resulte mejor.

Punto de la primaria: índice 1, corte 9.1773832091. Modo del bundle: pairwise_utility. Piso de calidad cumplido: sí.

| Política | Aciertos / prompts | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: | ---: |
| Relay v7 · primaria con vecinos | 3473/4406 | 78.82 % | 3.258349 |
| Ablación global v7 · sin vecinos | 3475/4406 | 78.87 % | 3.138356 |
| RouteLLM BERT publicado · 101 puntos | 3472/4406 | 78.80 % | 3.119096 |
| Relay · ensamble v4 | 3476/4406 | 78.89 % | 3.050169 |
| Relay v6 · política archivada | 3473/4406 | 78.82 % | 3.270807 |
| RouteLLM BERT · selección archivada de v6 (16 puntos) | 3468/4406 | 78.71 % | 3.107786 |
| Modelo fuerte fijo | 3467/4406 | 78.69 % | 3.301473 |
| Modelo barato fijo | 2416/4406 | 54.83 % | 0.135108 |

Todas las políticas seleccionan una llamada sobre los mismos prompts y el mismo par histórico. Los costos suman exclusivamente las llamadas seleccionadas guardadas en el dataset; excluyen representación, búsqueda, entrenamiento, reintentos y precios vigentes. No equivalen a facturación actual ni ahorro neto del harness.

| Primaria v7 frente a | Diferencia de éxito | IC 95 % de diferencia | Reducción relativa del costo histórico | IC 95 % de reducción |
| --- | ---: | ---: | ---: | ---: |
| Fuerte fijo | +0.14 pp | [-0.02, +0.32] pp | 1.31 % | [+0.89, +1.77] % |
| RouteLLM BERT publicado · 101 puntos | +0.02 pp | [-0.20, +0.25] pp | -4.46 % | [-5.64, -3.38] % |
| Relay · ensamble v4 | -0.07 pp | [-0.36, +0.23] pp | -6.83 % | [-8.12, -5.61] % |
| Relay v6 · política archivada | +0.00 pp | [-0.14, +0.14] pp | 0.38 % | [-0.04, +0.84] % |
| RouteLLM BERT · selección archivada de v6 (16 puntos) | +0.11 pp | [-0.11, +0.36] pp | -4.84 % | [-6.04, -3.74] % |
| Ablación global v7 · sin vecinos | -0.05 pp | [-0.27, +0.18] pp | -3.82 % | [-4.69, -3.00] % |

Los intervalos usan 5,000 réplicas de bootstrap pareado por prompt, semilla 42, después de elegir con esta misma validación. No corrigen la selección repetida entre versiones, no son intervalos confirmatorios de superioridad y no validan no inferioridad. Una reducción negativa significa mayor costo de v7.

| Primaria v7 frente a | Diferencia de costo histórico / 1,000 | IC 95 % de diferencia |
| --- | ---: | ---: |
| Fuerte fijo | -0.043124 | [-0.058551, -0.029463] |
| RouteLLM BERT publicado · 101 puntos | +0.139253 | [+0.105602, +0.175273] |
| Relay · ensamble v4 | +0.208180 | [+0.172087, +0.244698] |
| Relay v6 · política archivada | -0.012458 | [-0.027494, +0.001444] |
| RouteLLM BERT · selección archivada de v6 (16 puntos) | +0.150563 | [+0.116651, +0.186749] |
| Ablación global v7 · sin vecinos | +0.119993 | [+0.094548, +0.146058] |

| Candidato v7 | Rutas baratas | Aciertos perdidos frente al fuerte | Aciertos recuperados | Aciertos del barato seleccionado | MSE de ganancia prevista |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primaria con vecinos | 45 (1.02 %) | 4 | 10 | 38 | 0.286196 |
| Ablación global | 177 (4.02 %) | 14 | 22 | 143 | 0.289018 |

Aciertos perdidos son casos enviados al barato donde solo acertaba el fuerte; recuperados, casos donde solo acertaba el barato. Elegir barato cuando ambos fallan puede ahorrar costo, pero conserva el fallo. El MSE mide la diferencia prevista de éxito frente a la diferencia observada; no garantiza calibración individual ni detección de tareas desconocidas.

En 60 solicitudes residentes, la mediana fue 591.517 ms y p95 658.402 ms. En modo pairwise_utility incluye representación completa, búsqueda de vecinos sin caché, predicción y política; el respaldo fijo omite el predictor. Excluye carga inicial, generación e infraestructura. No permite comparar latencias con BERT medidas en otro proceso ni constituye un SLA.

| Candidato v7 | Ajuste (s) | Predicción por lote de validación (s) |
| --- | ---: | ---: |
| Primaria con vecinos | 141.45 | 88.35 |
| Ablación global | 57.82 | 4.53 |

El bundle conserva la primaria y delega generación al harness. La matriz dispersa y las clases almacenadas contienen información derivada de fit, aunque no guarden los prompts originales como strings. La licencia del autor sigue pendiente y la procedencia del dataset y de sus artefactos requiere revisión antes de distribuirlos; la licencia del código original no se adjudica a los datos.

Una conclusión competitiva independiente requiere congelar el candidato y compararlo en familias y documentos nuevos del catálogo de destino, con criterio predeclarado, incertidumbre, tareas completas y recursos del router incluidos. Esta ejecución no hizo llamadas pagadas, no volvió a evaluar el test histórico y no promocionó automáticamente ninguna política.

Evidencia: [resumen](summary.json), [configuración y snapshot](development_started.json), [acciones congeladas antes de métricas](actions_frozen.json), [bundle del SDK](bundle.json), [procedencia BERT](../routellm-bert-secondary-v1/selection.json) y [procedencia del informe](report_v7_provenance.json). El informe verifica los artefactos guardados y su coherencia sin reentrenar, deserializar modelos ni cambiar la selección.
