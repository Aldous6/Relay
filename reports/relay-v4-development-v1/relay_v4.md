Relay v4 · desarrollo no lineal. Esta ejecución prueba otra familia de modelos después de conocer resultados anteriores. No es una cohorte independiente y no confirma superioridad sobre RouteLLM.

El par permanece gpt-4-1106-preview frente a mistralai/mixtral-8x7b-chat. Se usaron 20,368 prompts de ajuste, 4,383 de calibración y 4,406 de validación. Éxito significa score ≥ 0.75; el piso común es 78.19 %, el éxito del fuerte fijo menos 0.5 pp.

Los nuevos candidatos usan HistGradientBoosting con las 14 variables estructurales que conservan magnitud y, en tres variantes, TF-IDF reducido por SVD. Vocabulario, proyección, escaladores, clasificadores y estimaciones de costo se ajustan solamente en fit; la calibración usa su bloque disjunto. Esta arquitectura prueba interacciones de contenido, formato y longitud; no demuestra comprensión semántica ni que esas interacciones sean la causa de una mejora.

La primera tabla conserva la búsqueda inicial de nueve puntos por modelo. El candidato y corte se eligen por menor costo entre puntos cuya media cumple el piso. No se eligen usando los intervalos.

| Política y corte de desarrollo | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| GPT-4 fijo | 78.69 % | 3.301473 |
| RouteLLM BERT · búsqueda 9 · 0.5 | 78.71 % | 3.107786 |
| structural_hgb15 · 10 | 78.73 % | 3.149649 |
| svd32_hgb15 · 20 | 78.35 % | 2.958674 |
| svd64_hgb15 · 20 | 78.51 % | 2.937503 |
| svd64_hgb31 · 20 | 78.46 % | 2.962018 |

| Punto Relay frente al BERT de su búsqueda | Δéxito, IC 95 % | Δcosto / 1,000, IC 95 % | Reducción relativa de costo | Diagnóstico conjunto |
| --- | ---: | ---: | ---: | --- |
| structural_hgb15 | +0.02 pp, [-0.27 pp, +0.32 pp] | +0.041864, [+0.004310, +0.081369] | -1.35 % [-2.62 %, -0.14 %] | no cumple |
| svd32_hgb15 | -0.36 pp, [-0.79 pp, +0.07 pp] | -0.149112, [-0.197387, -0.101092] | 4.80 % [3.26 %, 6.32 %] | no cumple |
| svd64_hgb15 | -0.20 pp, [-0.66 pp, +0.23 pp] | -0.170282, [-0.221281, -0.120101] | 5.48 % [3.89 %, 7.06 %] | no cumple |
| svd64_hgb31 | -0.25 pp, [-0.66 pp, +0.16 pp] | -0.145767, [-0.194681, -0.096412] | 4.69 % [3.12 %, 6.23 %] | no cumple |

La segunda búsqueda utiliza 101 cuantiles de puntuaciones por router, con una política común de ganancia prevista dividida entre costo adicional previsto positivo. Las matrices completas de acciones se guardan antes de calcular sus métricas. El presupuesto de búsqueda no elimina el sesgo de reutilizar validación ni la diferencia entre supervisión local de Relay y supervisión externa de BERT.

| Política y corte de desarrollo | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| routellm_bert · 0.55394751 | 78.30 % | 3.032820 |
| structural_hgb15 · 18.818691 | 78.19 % | 2.972467 |
| svd32_hgb15 · 21.544496 | 78.28 % | 2.899302 |
| svd64_hgb15 · 21.203071 | 78.23 % | 2.886826 |
| svd64_hgb31 · 22.758218 | 78.26 % | 2.820809 |

| Punto Relay frente al BERT de su búsqueda | Δéxito, IC 95 % | Δcosto / 1,000, IC 95 % | Reducción relativa de costo | Diagnóstico conjunto |
| --- | ---: | ---: | ---: | --- |
| structural_hgb15 | -0.11 pp, [-0.57 pp, +0.36 pp] | -0.060353, [-0.111302, -0.009671] | 1.99 % [0.33 %, 3.65 %] | no cumple |
| svd32_hgb15 | -0.02 pp, [-0.54 pp, +0.50 pp] | -0.133518, [-0.187980, -0.079162] | 4.40 % [2.65 %, 6.16 %] | no cumple |
| svd64_hgb15 | -0.07 pp, [-0.59 pp, +0.45 pp] | -0.145993, [-0.202700, -0.089936] | 4.81 % [2.99 %, 6.63 %] | no cumple |
| svd64_hgb31 | -0.05 pp, [-0.54 pp, +0.48 pp] | -0.212010, [-0.270318, -0.152864] | 6.99 % [5.09 %, 8.85 %] | no cumple |

El diagnóstico conjunto del punto ya elegido exige límite inferior del IC de Δéxito > −0.5 pp y límite superior del IC de Δcosto < 0. Cumplirlo aquí es descriptivo: los intervalos se calculan después de elegir con la misma validación y no tienen cobertura confirmatoria garantizada. No se busca otro de los 101 intervalos para conseguir una conclusión favorable.

Como control histórico se conservan todos los puntos seleccionados en el barrido v3 de 101 cuantiles. Comparten población, piso, par y el mismo punto BERT, pero proceden de una búsqueda previa y mayor historial acumulado de selección. No se borran los controles que resulten desfavorables ni se atribuye una diferencia exclusivamente al algoritmo.

| Política y corte de desarrollo | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| Control v3 · absolute_hybrid_full · 25.241258 | 78.37 % | 2.734335 |
| Control v3 · pairwise_hybrid_a3 · 30.353586 | 78.30 % | 2.805949 |
| Control v3 · pairwise_hybrid_a4 · 28.574465 | 78.21 % | 2.459792 |
| Control v3 · pairwise_hybrid_a5 · 24.942899 | 78.19 % | 2.781037 |

| Punto Relay frente al BERT de su búsqueda | Δéxito, IC 95 % | Δcosto / 1,000, IC 95 % | Reducción relativa de costo | Diagnóstico conjunto |
| --- | ---: | ---: | ---: | --- |
| Control v3 · absolute_hybrid_full | +0.07 pp, [-0.43 pp, +0.57 pp] | -0.298485, [-0.364008, -0.234832] | 9.84 % [7.82 %, 11.87 %] | cumple descriptivamente |
| Control v3 · pairwise_hybrid_a3 | +0.00 pp, [-0.50 pp, +0.50 pp] | -0.226870, [-0.294366, -0.158837] | 7.48 % [5.28 %, 9.55 %] | cumple descriptivamente |
| Control v3 · pairwise_hybrid_a4 | -0.09 pp, [-0.68 pp, +0.52 pp] | -0.573028, [-0.650984, -0.496829] | 18.89 % [16.61 %, 21.15 %] | no cumple |
| Control v3 · pairwise_hybrid_a5 | -0.11 pp, [-0.61 pp, +0.39 pp] | -0.251783, [-0.316053, -0.189088] | 8.30 % [6.27 %, 10.32 %] | no cumple |

La búsqueda inicial registró svd64_hgb15 para evaluación futura; la búsqueda de 101 cuantiles registró svd64_hgb31 con corte 22.7582182694. Son selecciones de desarrollo, no autorizaciones de promoción automática. Los candidatos, curvas, artefactos y reglas completos están en los resúmenes enlazados.

| CPU residente, candidato de la búsqueda inicial | p50 | p95 | n |
| --- | ---: | ---: | ---: |
| Relay · svd64_hgb15 | 39.32 ms | 48.29 ms | 60 |
| RouteLLM BERT | 509.17 ms | 1033.78 ms | 60 |

La medición compara los mismos 60 prompts, con modelos cargados, 2 calentamientos y orden alternado en un proceso CPU. Recalcula representación y predicción sin caché de atributos. BERT conserva su límite de 512 tokens y Relay procesa el texto completo. Se mide el candidato y punto iniciales, no todos los candidatos o todos los cortes de 101 cuantiles.

Son tiempos de esta muestra pequeña y hardware, no un SLA ni ahorro monetario neto. Las inicializaciones tienen fronteras distintas y las cachés del sistema operativo no se reinician; no se comparan como arranques en frío controlados.

| Candidato no lineal | Brier multiclase | Log-loss multiclase | MSE de ganancia prevista |
| --- | ---: | ---: | ---: |
| structural_hgb15 | 0.470752 | 0.780202 | 0.291831 |
| svd32_hgb15 | 0.468530 | 0.774882 | 0.289776 |
| svd64_hgb15 | 0.468078 | 0.774640 | 0.290391 |
| svd64_hgb31 | 0.471654 | 0.779449 | 0.291507 |

El Brier es la media por prompt de la suma de errores cuadrados de tres clases, rango [0,2]. Estos diagnósticos y soportes no garantizan calibración individual ni se comparan como si fueran el mismo objetivo que la preferencia externa de BERT. Las notas y soportes completos están en [diagnostics.json](diagnostics.json).

La mayor flexibilidad de los árboles puede aprender formatos de benchmark y sobreajustar la validación reutilizada. No se ha evaluado generalización a familias nuevas, conversaciones o documentos productivos. Los costos son estimaciones históricas de las llamadas elegidas y excluyen cómputo del router, entrenamiento y reintentos. El test exterior anterior no se usa ni se reevalúa en este informe.

Una afirmación futura de mejora sustancial exige congelar un candidato y un corte y evaluarlos en una cohorte nueva, con criterio compuesto de calidad y costo y overhead contabilizado si se afirma utilidad productiva. No se ensancha el margen ni se eliminan candidatos negativos para fabricar una victoria.

Fuentes: [entrenamiento y búsqueda inicial](summary.json), [ablación de política de 101 cuantiles](../relay-v4-operating-v1/summary.json), [protocolo v4](../../docs/relay_v4_protocolo.md) y [control v3](../relay-v3-operating-v1/summary.json). El informe presenta los valores almacenados, sin reentrenar, recalibrar, cambiar cortes ni generar respuestas nuevas.
