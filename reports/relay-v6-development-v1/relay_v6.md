Relay v6 · riesgo de perder calidad al elegir el modelo barato. Informe de desarrollo reutilizado.

La política pasó el criterio de selección sobre las medias de desarrollo. Frente al BERT seleccionado, Relay costó 5.25 % más; no se consiguió el objetivo de ahorrar frente a esa referencia. Este resultado no confirma superioridad, no inferioridad ni ahorro neto en producción.

Se conservaron 20,368 filas de ajuste, 4,383 de calibración y 4,406 de validación del train exterior. El test histórico consumido no se volvió a evaluar. Esta validación ya había intervenido en versiones anteriores: los resultados e intervalos son exploratorios.

El par conserva sus identidades históricas: gpt-4-1106-preview frente a mistralai/mixtral-8x7b-chat. Éxito significa score guardado ≥ 0.75. Este bundle no está validado para reemplazar esos candidatos por modelos locales o de API actuales.

V6 reutiliza la representación híbrida existente de TF-IDF de palabras/caracteres y 14 atributos. Aprende dos probabilidades marginales calibradas por separado: que solo acierte el fuerte y que acierte el barato. La política solo permite el barato con riesgo previsto suficientemente bajo, éxito previsto suficientemente alto y costo previsto menor. Estas probabilidades no constituyen una garantía individual ni una distribución conjunta calibrada.

Se fijaron 16 combinaciones para Relay y 16 umbrales para BERT antes de entrenar. La grilla BERT incluye 0.5; se conserva además el control de 101 puntos de la búsqueda anterior. Igualar el número de puntos no iguala la supervisión ni elimina las diferencias del historial de búsqueda. Relay aprende con outcomes locales de RouterBench; el checkpoint BERT mantiene supervisión externa. No se evalúa RouteLLM MF.

El piso de selección v6 fue 78.71 %: el máximo entre el éxito del fuerte fijo y el BERT seleccionado en la grilla de 16. Se exigió costo estrictamente menor que el fuerte fijo y se eligió por menor costo entre medias factibles, sin utilizar intervalos para elegir. Piso cumplido por la política exportada: sí.

Punto seleccionado: riesgo máximo 0.1, éxito mínimo previsto del barato 0.5. Modo del bundle: guarded.

| Política | Aciertos / prompts | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: | ---: |
| Relay v6 · política seleccionada | 3473/4406 | 78.82 % | 3.270807 |
| Modelo fuerte fijo | 3467/4406 | 78.69 % | 3.301473 |
| Modelo barato fijo | 2416/4406 | 54.83 % | 0.135108 |
| RouteLLM BERT · grilla de 16 | 3468/4406 | 78.71 % | 3.107786 |
| RouteLLM BERT · umbral publicado 0.5 | 3468/4406 | 78.71 % | 3.107786 |
| RouteLLM BERT · control previo de 101 | 3472/4406 | 78.80 % | 3.119096 |

Las políticas se miden con los mismos prompts y resultados históricos de sus llamadas seleccionadas. Los costos excluyen cómputo del router, entrenamiento, reintentos y precios vigentes. No son una medición de facturación ni ahorro total en un harness.

| Relay v6 frente a | Diferencia de éxito | IC 95 % de diferencia | Reducción relativa del costo histórico | IC 95 % de reducción |
| --- | ---: | ---: | ---: | ---: |
| Fuerte fijo | +0.14 pp | [+0.02, +0.27] pp | 0.93 % | [+0.63, +1.27] % |
| BERT · 16 puntos | +0.11 pp | [-0.14, +0.36] pp | -5.25 % | [-6.41, -4.20] % |
| BERT · control previo 101 | +0.02 pp | [-0.23, +0.27] pp | -4.86 % | [-5.99, -3.84] % |

Una reducción negativa significa mayor costo para Relay. Los intervalos proceden de bootstrap pareado por prompt después de seleccionar con esta misma validación: no tienen interpretación confirmatoria de superioridad.

| Relay v6 frente a | Diferencia de costo histórico / 1,000 | IC 95 % de diferencia |
| --- | ---: | ---: |
| Fuerte fijo | -0.030666 | [-0.041743, -0.020732] |
| BERT · 16 puntos | +0.163021 | [+0.131158, +0.198320] |
| BERT · control previo 101 | +0.151710 | [+0.121002, +0.186761] |

La política eligió el barato en 40 prompts (0.91 %). En esas rutas perdió 1 acierto que el fuerte sí conseguía y recuperó 7 que solo conseguía el barato; este último acertó 37 de las llamadas seleccionadas. Brier marginal de daño: 0.191865; Brier de éxito barato: 0.212855.

En 60 solicitudes con el modelo residente, la política exportada tuvo mediana 11.226 ms y p95 14.363 ms. La medición incluye representación, predicción y decisión cuando el modo es guarded; el modo fijo omite el predictor. No incluye carga inicial, generación ni infraestructura de servicio, y no permite comparar latencia de BERT en otro proceso o medición.

El ajuste tardó 55.34 segundos y la predicción por lote de validación 4.48 segundos en esta ejecución. No se realizaron llamadas pagadas ni se inició una aplicación.

El siguiente criterio de promoción requiere congelar este candidato y evaluarlo en una cohorte independiente representativa de los modelos y tareas del harness, incluyendo el overhead completo. Reutilizar el test histórico o las mismas familias ya observadas no convierte este resultado en confirmación.

Evidencia: [resumen](summary.json), [configuración y snapshot](development_started.json), [rutas persistidas antes de métricas](actions_frozen.json), [bundle para el SDK](bundle.json) y [procedencia del informe](report_v6_provenance.json). Este informe se deriva de los artefactos guardados, sin reentrenar ni modificar la selección.
