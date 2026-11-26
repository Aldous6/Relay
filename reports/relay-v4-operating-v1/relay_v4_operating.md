Relay v4 · ablación exploratoria de política. Esta grilla de 101 puntos se declaró antes de entrenar los nuevos candidatos, junto a la comparación inicial de nueve puntos. La validación ya había sido reutilizada en etapas anteriores: la predeclaración no la convierte en una evaluación independiente. Se conservan todos los candidatos y los estudios anteriores.

Se reutilizan las mismas 4,406 filas de validación del train exterior, sin reentrenar. El par es gpt-4-1106-preview frente a mistralai/mixtral-8x7b-chat. El piso de selección es 78.19 % de éxito, correspondiente al fuerte fijo menos 0.5 pp.

La política Relay divide la ganancia esperada de éxito por el costo previsto adicional positivo. La ganancia es P(éxito fuerte)−P(éxito débil) en el objetivo absoluto y P(strong_only)−P(weak_only) en el pareado. Así se compara el objetivo bajo la misma política. Se comprobaron costos adicionales positivos para todos los prompts; no se eliminaron casos que violaran la condición.

Cada grilla usa 101 cuantiles de predicciones, sin outcomes para fijar sus posiciones, con extremos nextafter que incluyen siempre fuerte y siempre débil. Pueden existir cuantiles o acciones duplicadas y se informa su número efectivo. Para BERT y Relay se usa score>umbral, con igualdad enviada al débil; esta convención explícita difiere del ≥ usado en la primera referencia BERT. Algunos umbrales de utilidad pueden ser negativos: son cortes de esta puntuación operativa, no el λ no negativo de la política anterior. Las matrices completas de acciones se guardaron antes de calcular métricas.

| Candidato con política común | Éxito | Costo histórico / 1,000 | Fracción débil |
| --- | ---: | ---: | ---: |
| GPT-4 fijo | 78.69 % | 3.301473 | 0.00 % |
| routellm_bert | 78.30 % | 3.032820 | 7.01 % |
| structural_hgb15 | 78.19 % | 2.972467 | 7.01 % |
| svd32_hgb15 | 78.28 % | 2.899302 | 8.01 % |
| svd64_hgb15 | 78.23 % | 2.886826 | 8.01 % |
| svd64_hgb31 | 78.26 % | 2.820809 | 9.01 % |

| Relay frente al BERT seleccionado aquí | Δéxito (pp), IC 95 % | Reducción relativa de costo, IC 95 % |
| --- | ---: | ---: |
| structural_hgb15 | -0.11 [-0.57, +0.36] | +1.99 % [+0.33, +3.65] |
| svd32_hgb15 | -0.02 [-0.54, +0.50] | +4.40 % [+2.65, +6.16] |
| svd64_hgb15 | -0.07 [-0.59, +0.45] | +4.81 % [+2.99, +6.63] |
| svd64_hgb31 | -0.05 [-0.54, +0.48] | +6.99 % [+5.09, +8.85] |

El candidato seleccionado solo para una evaluación futura es svd64_hgb31, con corte 22.7582182694. Su artefacto fuente es reports/relay-v4-development-v1/svd64_hgb31.joblib. La selección no cambia el router de la demo.

La misma validación eligió candidatos y cortes tras una búsqueda adicional guiada por resultados anteriores. Los intervalos pareados son descriptivos después de selección y no respaldan una victoria confirmatoria ni su cobertura nominal. No se descartan los candidatos negativos. Las probabilidades de BERT tienen un objetivo de preferencia externo y Relay recibió ajuste local; no se aísla el algoritmo bajo supervisión idéntica.

Los costos son llamadas históricas seleccionadas, sin cómputo del router, entrenamiento o reintentos. No hay aquí medición comparable de overhead de ambos ni ahorro monetario productivo. El test exterior anterior no se consulta ni se reevalúa. Para afirmar victoria se debe congelar un candidato y corte y usar una cohorte nueva con criterio compuesto de calidad y costo, como describe el protocolo v3.

Fuentes y curvas: [summary.json](summary.json), [selection.json](selection.json), [acciones congeladas](actions_frozen.json) y [etapa v3 original](../relay-v3-development-v1/summary.json). Los hashes de fuentes y artefactos quedan en el resumen.
