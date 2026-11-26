Relay · piloto semántico no lineal. Los resultados reutilizan validación y no son una prueba independiente.

Se conservan exactamente las 6,000 filas del piloto, 4,383 de calibración y 4,406 de validación. El encoder E5 está congelado, usa los pesos y la tokenización previamente fijados y cubre el texto mediante ventanas; su caché contiene atributos sin outcomes. Los nuevos árboles se ajustan solo en fit y su calibración es disjunta.

| Router | Éxito | Costo histórico / 1,000 |
| --- | ---: | ---: |
| BERT · 101 puntos | 78.30 % | 3.032820 |
| e5_hgb15 | 78.26 % | 3.026376 |
| e5_hgb31 | 78.23 % | 2.996173 |
| semantic_linear_control | 78.26 % | 2.883303 |

Los puntos se eligen por menor costo entre medias de éxito que cumplen el piso del fuerte fijo menos 0.5 pp. Se guardaron todas las acciones antes de calcular métricas nuevas. El control semántico lineal usa las mismas filas y representación, pero también cambia el objetivo; no se aísla exclusivamente el clasificador. Los modelos v3/v4 de 20,368 filas son comparaciones de pipelines con cantidades de ajuste distintas.

Los intervalos completos, soportes y diagnósticos están en summary.json. Son posteriores a selección y no confirman no inferioridad. El costo es histórico de llamadas y excluye encoder, entrenamiento y reintentos. No hay ahorro productivo ni promoción automática. No se consultó el test exterior para este piloto.
