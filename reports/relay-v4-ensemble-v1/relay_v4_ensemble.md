Relay v4 · ensemble fijo 50/50. Esta mezcla se declaró antes de completar los resultados v4 y no aprende pesos nuevos. Sus componentes ya fueron ajustados y seleccionados durante desarrollo; la mezcla y el corte no constituyen una evaluación independiente.

El par es gpt-4-1106-preview frente a mistralai/mixtral-8x7b-chat, con score de éxito ≥ 0.75. Se reutilizan 4,406 prompts de validación; los componentes comparten 20,368 filas de ajuste y 4,383 de calibración. Se verificaron preparación, orden de prompts, clases y soportes antes de mezclar.

Se promedian las tres probabilidades y los costos previstos de los componentes con pesos 0.5/0.5. La ganancia esperada es P(strong_only)−P(weak_only), dividida entre el costo previsto adicional estrictamente positivo. Ambos fallos permanecen como equal. Las predicciones, el artefacto y las 101 acciones se guardaron antes de calcular métricas de la mezcla.

| Política y corte de desarrollo | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| GPT-4 fijo | 78.69 % | 3.301473 |
| RouteLLM BERT · búsqueda 101 v4 · 0.55394751 | 78.30 % | 3.032820 |
| Ensemble fijo 50/50 · 25.48634 | 78.23 % | 2.786580 |

El piso de selección es 78.19 %, correspondiente al fuerte fijo menos 0.5 pp. Se seleccionó por menor costo con media de éxito al menos igual al piso, sin usar intervalos para escoger el corte.

| Punto Relay frente al BERT de su búsqueda | Δéxito, IC 95 % | Δcosto / 1,000, IC 95 % | Reducción relativa de costo | Diagnóstico conjunto |
| --- | ---: | ---: | ---: | --- |
| Ensemble fijo 50/50 | -0.07 pp, [-0.57 pp, +0.43 pp] | -0.246240, [-0.309930, -0.186861] | 8.12 % [6.23 %, 10.10 %] | no cumple |

El diagnóstico conjunto del punto ya elegido no cumple los límites de calidad y costo. Los intervalos son posteriores a selección sobre validación reutilizada: no prueban una victoria, ni garantizan cobertura nominal. No se reentrena, se ajustan pesos, se ensancha el margen ni se buscan otros intervalos para conseguir una conclusión favorable.

Las dos representaciones y predictores se ejecutan al inferir; el costo de router y su latencia no se suponen gratuitos. Esta ejecución no mide una ventaja de latencia ni gasto neto de producción. Los costos de la tabla son llamadas históricas seleccionadas y excluyen entrenamiento, cómputo del router y reintentos.

No se consulta ni reevalúa el test exterior. Una afirmación independiente requiere congelar artefacto, componentes y corte y evaluarlos sobre una cohorte nueva. La adaptación local de Relay y la supervisión externa de BERT siguen siendo distintas, y no se ha comprobado transferencia a familias nuevas o documentos productivos.

Fuentes: [resumen](summary.json), [curva completa](curve.json), [acciones congeladas](actions_frozen.json), [reproducción de inferencia](inference_replay.json), [diagnósticos](diagnostics.json), [v4](../relay-v4-development-v1/summary.json) y [referencia BERT](../relay-v4-operating-v1/summary.json).
