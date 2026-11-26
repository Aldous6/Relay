Resultados de la fase semántica de Relay, 8 de noviembre de 2026. Son comparaciones exploratorias dentro del entrenamiento exterior, no resultados del test final ni una comparación ejecutada contra RouteLLM.

Se reutilizaron exactamente las mismas 6,000 filas de ajuste, 4,383 de calibración y 4,406 de validación del piloto anterior, verificando huellas y archivos. El objetivo de éxito fue score≥0.75 y la política usó probabilidad mínima 0.75, ambos fijados antes de esta ejecución. El encoder fue E5-small ONNX qint8 congelado, con ventanas de 507 tokens de cuerpo, prefijo y especiales hasta 512, y solapamiento de 64. La agregación es una media ponderada y no aprende relaciones entre ventanas.

| Política | Éxito en validación | Costo histórico por 1,000 llamadas |
| --- | ---: | ---: |
| GPT-4-1106-preview fijo, elegido por éxito en fit | 78.69 % | 3.301473 |
| Estructural | 79.23 % | 2.188876 |
| Híbrida léxica | 78.23 % | 1.792719 |
| Semántica | 78.96 % | 1.915311 |
| Semántica + 14 variables | 78.39 % | 1.880958 |

Los costos son los estimados históricos guardados por RouterBench. Excluyen cómputo del router, entrenamiento y reintentos, y no son tarifas actuales. Con estos puntos operativos no hay una variante que mejore todas las medidas: añadir complejidad no acreditó una ventaja general.

La semántica reduce 41.99 % el costo histórico de llamadas frente al fijo, con intervalo bootstrap pareado del 95 % entre 40.87 y 43.13 %. La diferencia de éxito es +0.27 puntos porcentuales, con intervalo entre −0.79 y +1.41 puntos. Esto no demuestra superioridad de calidad ni no inferioridad sin un margen preespecificado. Los intervalos son de desarrollo y remuestrean prompts, no familias de benchmarks.

Frente al híbrido léxico, la semántica cuesta un 6.84 % más en llamadas guardadas y aumenta 0.73 puntos el éxito observado. El intervalo de éxito entre −0.16 y +1.61 puntos incluye cero. La diferencia de costo por 1,000 es +0.122592, con intervalo +0.077976 a +0.165867. Con estos resultados no se justifica declarar al encoder ganador frente a TF-IDF. Frente a la referencia estructural, la semántica reduce costo 12.50 % y tiene −0.27 puntos de éxito observado, también con incertidumbre.

La combinación de semántica y las 14 variables no mejoró el resultado de calidad de la semántica sola en este punto operativo. Redujo aproximadamente 1.79 % su costo de llamadas y perdió 0.57 puntos de éxito observado. Es una observación del experimento y configuración presentes, no una prueba de que ambas fuentes nunca se complementen. Las curvas de desarrollo para otros umbrales quedan disponibles en el manifiesto, sin seleccionar retrospectivamente una como victoria confirmatoria.

En las 4,406 filas de validación se verificó cobertura de todos los tokens sin truncación. Solo 30 prompts requirieron múltiples ventanas y el máximo fue 825 tokens de cuerpo. Por tanto, esta población no acredita calidad de routing para documentos de decenas de miles de tokens. Una prueba nativa adicional procesó un texto sintético de 11,715 tokens en 27 ventanas y cubrió todos ellos, sin truncar. Es una prueba funcional de cobertura, no una evaluación de respuestas de un LLM para ese texto nuevo.

El procesamiento de ese texto sintético tomó 13.34 segundos en esta CPU y configuración, incluyendo la primera carga de esa instancia del encoder. Este costo es material para la futura producción: el tratamiento de contexto largo necesita un presupuesto de latencia y optimización antes de desplegar. No se presenta la cobertura completa como si fuera gratuita. Las mediciones de solicitudes cortas del piloto tuvieron p50/p95 de 43.41/166.13 ms para semántica y 40.48/153.18 ms para híbrida semántica, con 100 filas, proceso caliente y caché desactivada. Son medidas de ejecuciones separadas, no un benchmark controlado que permita afirmar que la híbrida es más rápida.

Se comprobó serialización sin sesiones ONNX, carga perezosa con verificación de archivos, recomendación para un prompt nuevo, igualdad de tokens de una ventana con la referencia del tokenizer y repetición exacta en un ejemplo sin caché. Las pruebas unitarias cubren además rechazo de fuga, caché por texto exacto, invalidación por configuración, máscaras, particiones y cobertura. El entorno base y el entorno semántico conservan separación de dependencias; no se levantó ninguna aplicación.

La evidencia está en `reports/relay-semantic-development-v1/summary.json`, `paired_comparisons.json`, los Parquet de cobertura y `native_integration.json`. La referencia oficial RouteLLM, la evaluación por familias, conversaciones reales, el catálogo contemporáneo y el test final siguen pendientes. El siguiente diseño debería investigar cuándo compensa activar semántica frente a la ruta estructural/léxica y cómo reutilizar contexto estable, midiendo ahorro neto y calidad por tarea. Esa política nueva debe ajustarse en desarrollo y congelarse antes de su evaluación independiente.
