Análisis exploratorio de desarrollo de la cascada, 18 de noviembre de 2026. No es una nueva validación ni activa el ahorro.

Se auditaron las observaciones originales y la recuperación de infraestructura. La decisión usa únicamente public_passed: si falla, suma un episodio 120B desde el estado original. Se incluyen costos de intentos fallidos y reintentos internos. Las etiquetas de aceptación independientes se consultan después de fijar la ruta.

| Política | Aceptadas | Costo tarifario total USD | USD por aceptada | Pérdidas / ganancias frente a 120B |

|---|---:|---:|---:|---:|

| fixed_baseline | 23/24 | 0.00776850 | 0.00033776 | 0 / 0 |

| fixed_candidate | 22/24 | 0.00516165 | 0.00023462 | 2 / 1 |

| public_cascade | 23/24 | 0.00549255 | 0.00023881 | 1 / 1 |

| historical_relay_adaptive | 23/24 | 0.00722580 | 0.00031417 | 0 / 0 |

| historical_bert_0.50 | 23/24 | 0.00776850 | 0.00033776 | 0 / 0 |

| historical_bert_0.58 | 23/24 | 0.00722580 | 0.00031417 | 0 / 0 |

La cascada pública reconstruida reduce el costo tarifario por aceptada en 29.30%, con 1 escalamiento(s). Pierde QuixBugs-quicksort y gana QuixBugs-mergesort. Igualar la aceptación agregada no elimina la pérdida pareada. El gate de cero pérdidas no se cumple.

Bootstrap pareado por grupo: 10000 réplicas, semilla 20261118; intervalo percentil descriptivo del 95% para ahorro [0.006814454829762373, 0.508959247960749]; para diferencia de aceptación [-0.125, 0.125]. Límite binomial exacto unilateral del 95% para pérdida pareada: 18.29%. Estos intervalos no corrigen selección posterior de la política, contaminación de benchmarks ni dependencia entre familias.

Con cero pérdidas en 24 grupos, el límite exacto aún sería 11.73%. Para que ese límite sea como máximo 2%, se necesitan al menos 149 grupos independientes sin pérdidas. Es un cálculo de precisión condicionado a cero pérdidas, no una promesa de potencia ni de éxito.

No se usa accepted o private_passed para decidir el escalamiento. No se reconstruye una cascada oracular para anunciar ahorros. El verificador operativo deja pasar una solución incorrecta; debe desarrollarse y congelarse antes de una reserva independiente.

Los costos corresponden a tarifas simuladas con uso observado; Groq Free no generó una factura con ese importe. No se descuenta el cómputo local como si fuera gratuito. La latencia neta y el ahorro monetario completo siguen sin medirse. Las ramas históricas comparten resultados entre controles y no son ejecuciones independientes de cada política.

BERT es el checkpoint publicado de RouteLLM con scoring local y umbrales históricos congelados; no se ejecutó RouteLLM MF ni Not Diamond. Los resultados anteriores permanecen intactos. El protocolo nuevo está en docs/openhands_cascade_v1.md.

Métodos: [NIST, límites binomiales exactos](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm) y [SciPy, bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
