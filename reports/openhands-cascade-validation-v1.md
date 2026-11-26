Informe de validación prospectiva de Relay, 18 de noviembre de 2026.

Se completaron 43 tareas nuevas para Relay con la política congelada antes de adquirir respuestas. La cascada real aceptó 42/43 y 120B fijo 43/43. Hubo 1 pérdidas y 0 ganancias pareadas; una ganancia no compensa una pérdida para el criterio de protección.

El ahorro de costo tarifario por tarea aceptada frente a 120B fue indefinido por costo/aceptación insuficiente. Costo total de los intentos de la cascada: USD None; referencia: USD 0.01239765. Costo por aceptada: USD None y USD 0.0002883174418604651, respectivamente. Se suman ambos intentos cuando hay escalamiento, incluidos los intentos fallidos; las ramas adquiridas solo como controles se contabilizan aparte.

El intervalo percentil pareado del 95%, con 10000 remuestreos y semilla 20261118, es no disponible. El límite binomial exacto unilateral del 95% de pérdida pareada es 10.56%. Cero pérdidas observadas o un intervalo bootstrap de calidad [0,0] no prueban daño poblacional nulo.

Criterio empírico del piloto: no cumplido: observed_paired_quality_loss, incomplete_cost_accounting, less_than_10_percent_savings_per_accepted. Evidencia fuerte: insuficiente: observed_paired_quality_loss, incomplete_cost_accounting, less_than_10_percent_savings_per_accepted, savings_interval_does_not_exclude_zero, harm_upper_bound_exceeds_two_percent.

La decisión utilizó pruebas públicas del original y del parche. La aceptación final se evaluó después de cerrar la ruta, fuera del agente. La referencia reutiliza 120B cuando ese modelo ya se ejecutó en la cascada; de lo contrario se adquirió después. No son ejecuciones independientes de cada política.

Los importes son un escenario tarifario con uso observado, no ahorro facturado por Groq Free. El costo local no está monetizado y no se declara ahorro neto. Estos benchmarks públicos no demuestran eficacia en cualquier repositorio ni superioridad frente a RouteLLM o Not Diamond.

Composición de la cohorte nueva: function-implementation: 42, function-repair: 1. La cohorte histórica tenía 12 tareas HumanEval y 12 QuixBugs; una mezcla distinta limita la comparación directa de sus porcentajes de ahorro.

El JSON adjunto conserva cada tarea, pérdidas, ganancias, costos, usos y hashes de código, suite, evaluadores y bitácoras. Los resultados históricos de 12,68% se mantienen separados; esta cohorte no se usa para reajustar la política y luego presentarla como una nueva validación.
