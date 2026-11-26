Relay · evaluación de episodios. Informe de integridad y prueba de infraestructura.

Esta ejecución usa respuestas programadas y entornos sintéticos en memoria. Comprueba contratos, escalado, persistencia y contabilidad; no mide calidad de LLMs, ahorro real ni capacidad general de agentes. Los importes LLM son ficticios de la simulación y no corresponden a facturas. El ROI no se evalúa.

Se registraron 12 tareas completas y 36 episodios tarea–política, con los mismos identificadores de tarea para todos los controles. Partición declarada: development. Familias: aggregate_records, file_lookup, repair_function. Cada política comenzó con un entorno propio; la unidad de éxito es el episodio completo según su verificador, incluyendo los fallos en el denominador.

Catálogo declarado: simulated/strong y simulated/weak. El control principal es Siempre fuerte. La cascada usa errores anteriores registrados por el harness; no puede consultar la referencia privada para decidir otra llamada después de una respuesta final.

Se verificaron el hash de congelación, el hash de tareas, el snapshot ZIP y sus fuentes, los sobres de journal/resumen y toda la cadena de eventos. Cada llamada completada conserva la decisión y solicitud persistidas antes de generación; los cierres conservan las llamadas y sus costos. Se repitieron annotate_costs y summarize desde los eventos guardados con la implementación local identificada en la procedencia, sin generación, ejecución de herramientas ni evaluación del contenido de respuestas. No se extrajo ni ejecutó código del ZIP.

| Política | Episodios | Éxitos / episodios | Éxito en esta prueba | Llamadas registradas | Reintentos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Siempre fuerte | 12 | 12/12 | 100.00 % | 47 | 0 |
| Siempre barato | 12 | 12/12 | 100.00 % | 55 | 8 |
| Barato y escalar tras error | 12 | 12/12 | 100.00 % | 55 | 8 |

Las llamadas registradas incluyen intentos interrumpidos de resultado desconocido cuando existen; no se reejecutan al reanudar. Los reintentos son un subconjunto de las llamadas, no un cargo adicional sumado dos veces. Una respuesta final incorrecta termina el episodio y no autoriza escalado con información del evaluador privado.

| Política | LLM ficticio (USD) | LLM de reintentos (subconjunto) | Costo total (USD) | Total / tarea (USD) | Episodios con costo completo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Siempre fuerte | 0.065800 | 0.000000 | desconocido | desconocido | 0/12 |
| Siempre barato | 0.007700 | 0.001120 | desconocido | desconocido | 0/12 |
| Barato y escalar tras error | 0.042980 | 0.011200 | desconocido | desconocido | 0/12 |

El costo total requiere LLM, router, verificación y herramientas. Un componente sin medición o tarifa permanece desconocido: no se reemplaza por cero. Medir segundos no establece su costo monetario. El total por tarea incluye episodios fallidos; un subtotal LLM no constituye costo total ni demuestra ahorro del harness.

| Política | Llamadas con costo LLM conocido | Llamadas con costo router desconocido | Verificación desconocida | Herramientas desconocidas |
| --- | ---: | ---: | ---: | ---: |
| Siempre fuerte | 47/47 | 47 | 47 | 47 |
| Siempre barato | 55/55 | 55 | 55 | 55 |
| Barato y escalar tras error | 55/55 | 55 | 55 | 55 |

| Política | Routing p50 (ms) | Routing p95 (ms) | Llamadas con tiempo routing | Episodio completo p50 (ms) | Episodio completo p95 (ms) | Episodios con tiempo total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Siempre fuerte | 0.270 | 0.495 | 47/47 | 813.195 | 2309.891 | 12/12 |
| Siempre barato | 0.319 | 0.485 | 55/55 | 1135.296 | 3067.396 | 12/12 |
| Barato y escalar tras error | 0.314 | 0.475 | 55/55 | 1198.065 | 2893.062 | 12/12 |

Routing mide la selección de política; el tiempo completo mide el episodio ejecutado. Los cuantiles usan únicamente mediciones disponibles y la cobertura aparece al lado. En una simulación con respuestas programadas, estos tiempos no estiman latencia de proveedores ni desempeño de un router aprendido que no haya participado.

| Política frente al fuerte | Diferencia de éxito de esta prueba | IC 95 % por grupos | Reducción de costo total | Estado de adopción / ROI |
| --- | ---: | ---: | ---: | --- |
| Siempre barato | +0.00 pp | [+0.00, +0.00] pp | desconocido | not_assessed_simulation |
| Barato y escalar tras error | +0.00 pp | [+0.00, +0.00] pp | desconocido | not_assessed_simulation |

Los intervalos son diagnósticos del conjunto guardado: 2000 réplicas, semilla 42, emparejadas por episode_id y remuestreadas por source_group. En respuestas programadas no son evidencia estadística sobre calidad de LLMs. Con costo total desconocido, la reducción de costo total también permanece desconocida.

Los criterios configurados son ahorro total mínimo 30.00 %, pérdida máxima de éxito 2.00 puntos porcentuales y p95 de routing menor que 200.000 ms. Son parámetros para una evaluación futura: la simulación no puede validar ROI y no hay adopción automática.

| Política | Cierres registrados |
| --- | --- |
| Siempre fuerte | final_answer: 12 |
| Siempre barato | final_answer: 12 |
| Barato y escalar tras error | final_answer: 12 |

El siguiente uso evaluable requiere fijar modelos, tareas y grupos independientes, callbacks reales autorizados, presupuestos y medición completa de recursos. Los resultados de esta prueba no sustituyen una comparación de Relay con RouteLLM ni modifican los resultados históricos de routing de una sola llamada.

Evidencia: [manifiesto](manifest.json), [snapshot de fuentes](source_bundle.zip), [journal íntegro](journal.json), [resumen](summary.json) y [procedencia del informe](report_provenance.json). Los hashes comprueban consistencia respecto a los archivos declarados; no autentican a su autor. Este informe no reentrena, no abre datos crudos ni deserializa modelos.
