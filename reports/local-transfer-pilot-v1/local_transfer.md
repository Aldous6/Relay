Relay · piloto prospectivo sintético local. Los modelos, políticas y parámetros de generación se fijaron antes de crear esta cohorte. Es una prueba controlada de transferencia; no representa validación productiva ni una victoria general sobre RouteLLM.

Se generaron 64 casos nuevos con valores aleatorios y ocho plantillas, sobre documentos e historial. Las respuestas de referencia se calculan al crear los casos; nunca entran en los callbacks de enrutado o generación. Se rechazaron duplicados normalizados internos y solapamientos exactos con las huellas conocidas de RouterBench. Esto no certifica ausencia de similitudes de formato ni independencia respecto al preentrenamiento general.

Se mantuvo Relay pairwise_hybrid_a5 con corte 20 y BERT con umbral 0.5, ambos provenientes del catálogo histórico. En esta prueba, la acción strong se proyecta a llama3:latest y weak a qwen2.5:1.5b. Esos nombres heredados no garantizan que Llama tenga mayor calidad local. Los digests de ambos modelos instalados y la semilla de la cohorte están congelados en policy_lock.json.

| Política | Respuestas exactas | Llama / Qwen | Tokens de entrada reportados | Tokens de salida reportados | Fallos de generación |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relay v3 congelado | 43/64 (67.19 %) | 64 / 0 | 14296 | 250 | 0 |
| RouteLLM BERT congelado | 43/64 (67.19 %) | 64 / 0 | 14296 | 250 | 0 |
| Siempre Llama 3 | 43/64 (67.19 %) | 64 / 0 | 14296 | 250 | 0 |
| Siempre Qwen 2.5 1.5B | 46/64 (71.88 %) | 0 / 64 | 15692 | 276 | 0 |

Se conservaron 128 llamadas únicas de colección, compartidas entre las cuatro políticas. Cada política selecciona una respuesta por caso; los totales por política son reconstrucciones sobre esos mismos resultados observados, no cuatro ejecuciones independientes. La comparación pareada tuvo 0 casos correctos solo con Relay y 0 solo con BERT.

La mejor referencia fija por exactitud en esta muestra fue Siempre Qwen 2.5 1.5B. Esta descripción no selecciona una nueva política ni prueba que un router aporte valor frente a un modelo fijo. Los resultados por plantilla son:

| Familia sintética | Relay | BERT | Siempre Llama | Siempre Qwen |
| --- | ---: | ---: | ---: | ---: |
| Consulta de un registro | 8/8 | 8/8 | 8/8 | 8/8 |
| Suma de lotes | 0/8 | 0/8 | 0/8 | 0/8 |
| Filtro y suma | 0/8 | 0/8 | 0/8 | 0/8 |
| Actualización hasta una fecha | 3/8 | 3/8 | 3/8 | 6/8 |
| Relación de dos pasos | 8/8 | 8/8 | 8/8 | 8/8 |
| Conflicto de fuentes | 8/8 | 8/8 | 8/8 | 8/8 |
| Dato ausente | 8/8 | 8/8 | 8/8 | 8/8 |
| Corrección en historial | 8/8 | 8/8 | 8/8 | 8/8 |

La métrica exige igualdad textual después de normalizar Unicode, mayúsculas y espacios. Penaliza respuestas con explicación o formato extra aunque contengan un dato correcto; mide salida exacta para tareas restringidas, no calidad semántica general. Las ocho plantillas y el tamaño pequeño limitan la generalización; no se calcula ni afirma no inferioridad de 0.5 pp con este piloto.

Los tokens son los contadores devueltos por cada modelo, con tokenizadores distintos. No son dólares ni una medida uniforme de cómputo. El costo monetario permanece desconocido. La colección incluye posibles cargas de modelos y efectos de caché; sumar duraciones seleccionadas no simula fielmente una política de servicio ni acredita un SLA o ahorro neto. Los tiempos de routing guardados incluyen además auditoría de cobertura BERT y no son el benchmark de latencia emparejado anterior.

Se usa una sola respuesta por caso y modelo, con temperatura 0 y seed 42 configuradas; no se ha estimado variabilidad entre generaciones. Un fallo de proveedor se conserva como fallo y no se reintenta ocultamente. Las políticas no se entrenan ni reajustan con este resultado. Para adaptar el router al catálogo local, estos casos tendrían que declararse desarrollo y reservar otra cohorte antes de volver a medir.

Fuentes locales: [resumen](evaluation/summary.json), [cohorte](cohort.jsonl), [políticas congeladas](policy_lock.json), [rutas](evaluation/routes.json), [respuestas](evaluation/responses.json) y [código archivado](execution_sources.zip). Parámetros y contadores del proveedor: [API chat de Ollama](https://docs.ollama.com/api/chat) y [parámetros de Ollama](https://docs.ollama.com/modelfile).
