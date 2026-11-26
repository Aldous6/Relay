Relay v3 · resultados de desarrollo. Esta ejecución amplía el ajuste y prueba el objetivo pareado después de haber consumido el test original. No es una nueva evaluación independiente ni confirma superioridad.

El par común es gpt-4-1106-preview frente a mistralai/mixtral-8x7b-chat. Se usaron 20,368 prompts de ajuste, 4,383 de calibración y 4,406 de validación. El umbral del score que define éxito fue 0.75. El piso de selección de desarrollo fue 78.19 %, calculado con el modelo fuerte fijo menos 0.5 pp.

| Candidato y punto de desarrollo | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| GPT-4 fijo | 78.69 % | 3.301473 |
| RouteLLM BERT · umbral 0.5 | 78.71 % | 3.107786 |
| absolute_hybrid_full · parámetro 0.8 | 78.67 % | 3.196976 |
| pairwise_hybrid_a3 · parámetro 20 | 78.67 % | 3.296870 |
| pairwise_hybrid_a4 · parámetro 20 | 78.92 % | 3.183647 |
| pairwise_hybrid_a5 · parámetro 20 | 78.62 % | 3.017791 |

El candidato seleccionado para una futura evaluación es pairwise_hybrid_a5, con parámetro 20. Frente al punto BERT de desarrollo tuvo Δéxito=-0.09 pp, IC pareado del 95 % [-0.43 pp, +0.25 pp]; Δcosto por 1,000=-0.089995, IC [-0.140019, -0.039916]. La reducción relativa de costo histórico fue 2.90 %, con intervalo descriptivo [1.29 %, 4.46 %]; un valor negativo significa mayor costo.

Estos intervalos reutilizan la validación que seleccionó modelo, regularización y punto operativo. Son diagnósticos descriptivos con sesgo de selección; no prueban no inferioridad ni victoria confirmatoria. Se exploraron cuatro modelos Relay con nueve puntos cada uno frente a un checkpoint BERT con nueve umbrales. Igualar puntos por modelo no iguala búsqueda, entrenamiento ni supervisión.

| Medición CPU individual con modelo residente | p50 | p95 | n |
| --- | ---: | ---: | ---: |
| Relay · pairwise_hybrid_a5 | 22.86 ms | 31.43 ms | 60 |
| RouteLLM BERT | 641.08 ms | 1062.69 ms | 60 |

La [medición CPU emparejada](latency_cpu.json) usa los mismos 60 prompts de desarrollo, inferencia individual, orden alternado de backends y 2 calentamientos por backend. Se recomputan representación y predicción, sin caché de vectores ni resultados; los modelos permanecen cargados. El límite de pools nativos y los hilos de Torch registrados son 4 y 4, respectivamente. BERT conserva su truncamiento oficial a 512 tokens mientras Relay procesa el texto completo; recibir los mismos prompts no implica retener el mismo contexto.

Estos tiempos describen esta muestra pequeña y no aleatoria, hardware y carga de CPU. No constituyen un SLA, una garantía para otros documentos o hardware ni ahorro monetario neto. Se excluyen red, colas, generación del LLM, reintentos y renderizado. Las inicializaciones tienen fronteras diferentes —la de BERT incluye verificaciones e imports adicionales— y las cachés del sistema operativo no se reiniciaron: no se comparan sus cifras como arranque en frío controlado.

El costo histórico de las políticas excluye cómputo del router, entrenamiento y reintentos. Una diferencia medida de latencia no confirma superioridad de calidad ni gasto neto de producción.

| Objetivo | Candidato | Brier diagnóstico | Log-loss diagnóstico |
| --- | --- | ---: | ---: |
| Dos éxitos absolutos · promedio binario | absolute_hybrid_full | 0.182142 | 0.542825 |
| Tres resultados pareados · multicategoría | pairwise_hybrid_a3 | 0.472741 | 0.782468 |
| Tres resultados pareados · multicategoría | pairwise_hybrid_a4 | 0.464488 | 0.768256 |
| Tres resultados pareados · multicategoría | pairwise_hybrid_a5 | 0.470542 | 0.777558 |

Los Brier y log-loss de objetivos distintos no se comparan como una misma escala de rendimiento: el Brier pareado suma los errores de tres clases por prompt y tiene rango [0,2]; el absoluto promedia dos problemas binarios. Soportes de ajuste/calibración/validación, notas de calibración y detalles están en [diagnostics.json](diagnostics.json). La normalización de tres calibradores one-vs-rest no garantiza calibración conjunta.

La trazabilidad de esta ejecución conserva predicciones y costos previstos antes de calcular las curvas. Las acciones de cada parámetro son deterministas y reconstruibles a partir de esos archivos; solo las rutas de los puntos elegidos se guardaron después de seleccionar. No se afirma que todas las rutas se persistieron antes de consultar outcomes. Esta nota documenta la implementación y no cambia las reglas de selección ni las métricas.

El [barrido operativo de 101 puntos](../relay-v3-operating-v1/relay_v3_operating.md) es una ablación secundaria posterior a estos primeros resultados. Conserva los mismos artefactos y reutiliza validación con búsqueda adicional; no reemplaza el candidato de la grilla inicial ni se promueve automáticamente a la demo.

No se usó ni reevaluó el test exterior consumido para esta revisión. La demo conserva la opción anterior y permite activar de forma explícita el candidato v3 de la grilla inicial como experimento; esa activación no equivale a validación productiva ni a promoción automática. BERT tiene supervisión externa y Relay ajuste local; la comparación no aísla el efecto del método. Una victoria requiere congelar un candidato y su punto y evaluarlos en una cohorte nueva, con límite inferior del IC de Δéxito>−0.005 y superior del IC de Δcosto<0, además del overhead para una afirmación productiva.

Fuentes locales: [resumen y curvas completos](summary.json), [protocolo v3](../../docs/relay_v3_protocolo.md), [diagnósticos](diagnostics.json). Los diagnósticos se calcularon después del entrenamiento con las predicciones almacenadas, sin reajustar probabilidades, cambiar el punto elegido ni generar respuestas nuevas.
