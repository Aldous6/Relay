# Relay y OpenHands: resultado del piloto público

Se completaron 48 episodios sobre 24 tareas públicas. La selección usó doce tareas de desarrollo; los resultados siguientes pertenecen a las doce reservadas. Las decisiones quedaron congeladas antes de evaluar esas tareas.

fixed_strong: 11/12 tareas aceptadas; costo LLM tarifario total USD 0.00342810; por tarea aceptada USD 0.00031165; ahorro relativo por tarea aceptada frente a 120B fijo 0.00 %.

fixed_cheap: 9/12 tareas aceptadas; costo LLM tarifario total desconocido; por tarea aceptada desconocido; ahorro relativo por tarea aceptada frente a 120B fijo desconocida.

relay_exploratory: 11/12 tareas aceptadas; costo LLM tarifario total USD 0.00342810; por tarea aceptada USD 0.00031165; ahorro relativo por tarea aceptada frente a 120B fijo 0.00 %.

bert_transferred_fixed_0.5: 11/12 tareas aceptadas; costo LLM tarifario total USD 0.00342810; por tarea aceptada USD 0.00031165; ahorro relativo por tarea aceptada frente a 120B fijo 0.00 %.

bert_transferred_development_calibrated: 11/12 tareas aceptadas; costo LLM tarifario total USD 0.00303345; por tarea aceptada USD 0.00027577; ahorro relativo por tarea aceptada frente a 120B fijo 11.51 %.

La regla de desarrollo conservó el modelo fuerte. En esta evaluación no produjo ahorro frente a usarlo siempre.

La referencia BERT calibrada obtuvo 11/12 aceptaciones frente a 11/12 de Relay y un costo de llamadas por tarea aceptada 11.51 % menor. Su selección usa el prompt de cada tarea; Relay fija una ruta para toda la actividad. Este resultado no valida una ventaja de Relay. La inferencia local de BERT tardó 8.40 segundos en las doce tareas, más 71.38 segundos de carga inicial; ese cómputo no está monetizado.

Los importes son un escenario sin caché con tokens informados por Groq y tarifas publicadas, no una factura pagada de Groq Free. Incluyen las llamadas de las tareas fallidas cuando hay uso informado; no convierten intentos sin uso en costo cero. El cómputo local, el desarrollo y la adquisición de evidencia no están monetizados.

La adquisición completa conservó 5 intentos previos y 52 del protocolo principal; 7 no informaron uso. El subtotal tarifario conocido de toda esa adquisición es USD 0.01216470 y no representa su costo completo.

El harness usa el agente y el ciclo reales de OpenHands. Se detiene al pasar ejemplos públicos y aplica después tests reservados, sin mostrarlos al modelo. Esta revisión eliminó el cierre adicional detectado en desarrollo; no se ejecutó una comparación completa del cierre antiguo contra el nuevo.

La comparación de políticas reutiliza ramas pareadas con el mismo agente y presupuesto. BERT es el checkpoint publicado de RouteLLM transferido al par GPT-OSS, con supervisión externa; el umbral calibrado se eligió únicamente en desarrollo. No se ejecutaron Not Diamond, MF ni SW.

HumanEval y QuixBugs son benchmarks históricos públicos que pueden estar presentes en el entrenamiento de los modelos. Esta muestra pequeña no acredita ausencia de deterioro poblacional, adopción comercial ni resultados en repositorios grandes. La política exploratoria no se aprobó para producción y los mínimos del SDK no se redujeron.

El [reporte JSON](harness-live-public-v2.json) conserva métricas, intentos, hashes y procedencia. El [protocolo](../docs/harness_public_pilot_v1.md) documenta la revisión antes de la validación.
