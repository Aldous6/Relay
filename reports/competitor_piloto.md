Primer piloto del router directo, 8 de noviembre de 2026. Resultados exploratorios dentro del train original. No se ha evaluado el test reservado ni ejecutado RouteLLM. Los números siguientes no acreditan desempeño productivo ni superioridad frente a otros routers.

Se ajustaron once predictores de éxito por variante con 6,000 prompts, se calibraron con 4,383 diferentes y se evaluaron políticas sobre otros 4,406. Se conservaron los casos donde ningún modelo alcanza score≥0.75. El umbral de probabilidad de la política fue 0.75, fijado antes de ejecutar. El piloto usa el mismo subconjunto de ajuste para ambas representaciones. Quedaron 14,368 filas del fit interior sin usar en este piloto.

| Política | Tasa de score≥0.75 | Costo histórico estimado por 1,000 prompts |
| --- | ---: | ---: |
| GPT-4-1106-preview fijo, elegido por éxito en fit | 78.69 % | 3.301473 |
| Router directo estructural | 79.23 % | 2.188876 |
| Router directo híbrido léxico | 78.23 % | 1.792719 |

Los costos corresponden a los importes históricos guardados por RouterBench y excluyen cómputo local del router, entrenamiento y reintentos. No representan facturación comercial actual. La variante híbrida contiene TF-IDF y 14 variables; todavía no contiene un encoder semántico.

La diferencia híbrido frente al modelo fijo es −0.45 puntos porcentuales de éxito, con intervalo bootstrap pareado del 95 % de −1.54 a +0.70 puntos. La reducción relativa del costo histórico de las llamadas es 45.70 %, con intervalo de 44.47 a 46.97 %. No se ha fijado un margen confirmatorio de calidad, y estos resultados no establecen no inferioridad. La selección o ajuste posterior según esta validación exige una evaluación final independiente.

Frente al router estructural, el híbrido reduce costo un 18.10 % y pierde aproximadamente un punto porcentual de éxito al umbral inicial. El intervalo de la diferencia de éxito es de −1.91 a −0.11 puntos. Por tanto, el enriquecimiento léxico no mejora todas las medidas al mismo punto operativo. La curva de desarrollo muestra otras combinaciones de costo y éxito, sin convertir la más favorable a posteriori en un resultado confirmatorio.

La medición piloto de inferencia incluyó representación desde texto, estimación de éxito/costo y selección con una solicitud por vez, proceso caliente y 100 filas de validación. Las medianas observadas fueron 31.21 ms para estructural y 39.00 ms para híbrido, con p95 de 41.31 y 52.67 ms. Son mediciones de esta ejecución y selección de longitudes, no un SLA. No se ejecutaron LLMs durante este entrenamiento ni se midió su latencia.

`reports/competitor-development-pilot-v1/summary.json` conserva parámetros, versiones, hashes, curvas y diagnósticos. `paired_comparisons.json` conserva 2,000 réplicas pareadas con semilla 42. Los intervalos remuestrean prompts y no respaldan generalización entre familias de benchmarks. Los modelos serializados y las predicciones se conservan localmente; sus archivos grandes están excluidos de Git.
