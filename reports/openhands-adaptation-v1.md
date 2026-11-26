La adaptación no cumple el gate de este piloto; el cargador conserva el modelo baseline.

Se verificaron 24 tareas nuevas para la selección de Relay y 48 episodios pareados. Cada política reutiliza las mismas dos ramas medidas por tarea; las rutas de Relay se congelaron antes de las llamadas.

| Política | Aceptadas | Coste LLM total | Coste por aceptada | Pérdidas pareadas frente a 120B | Ahorro por aceptada frente a 120B |
|---|---:|---:|---:|---:|---:|
| 120B fijo | 23/24 | USD 0.00776850 | USD 0.00033776 | 0 | 0.00% |
| 20B fijo | 21/24 | desconocido | desconocido | 3 | desconocido |
| Relay adaptado | 22/24 | desconocido | desconocido | 1 | desconocido |
| BERT umbral 0.50 | 23/24 | USD 0.00776850 | USD 0.00033776 | 0 | 0.00% |
| BERT umbral 0.58 | 22/24 | desconocido | desconocido | 1 | desconocido |



Relay seleccionó 20B en 5 tareas y 120B en 19. El límite superior Wilson unilateral del 95% para pérdidas pareadas observado es 16.68%; es descriptivo y no demuestra no inferioridad poblacional.

Motivos registrados por el gate: observed_paired_quality_loss, unknown_or_incomparable_selected_cost, cost_basis_changed_or_incomparable, insufficient_savings_per_accepted_task.

Frente a 20B fijo, Relay presenta 1 pérdidas y 2 ganancias pareadas; su variación de coste por aceptada es desconocido de ahorro. La diferencia es propia de esta muestra y no acredita superioridad general.

Frente a BERT umbral 0.50, Relay presenta 1 pérdidas y 0 ganancias pareadas; su variación de coste por aceptada es desconocido de ahorro. La diferencia es propia de esta muestra y no acredita superioridad general.

Frente a BERT umbral 0.58, Relay presenta 0 pérdidas y 0 ganancias pareadas; su variación de coste por aceptada es desconocido de ahorro. Las decisiones son idénticas: esta muestra no demuestra una ventaja incremental de routing frente a ese control.

El routing residente reconstruido suma 21.7555 s: 21.7481 s de encoder medidos anteriormente y 0.0074 s de replay de la cabeza con caché. La carga del encoder agrega 54.7696 s. Estos tiempos no son latencia end-to-end.

No se puede calcular un precio horario positivo de CPU de equilibrio con costes completos y ahorro LLM positivo.

El ajuste y selección automáticos registraron 4.5582 s; el trabajo histórico del encoder se conserva aparte en el JSON. No se supone amortización ni precio de cómputo.

La adquisición de las dos ramas conserva 52 despachos HTTP y 0 costes desconocidos; subtotal tarifario conocido USD 0.01282365, total USD 0.01282365. Este gasto de adquisición se informa separado del coste de ejecutar cada política.

Tras los primeros 6 episodios se registró una enmienda de ejecución para reintentar escrituras locales bloqueadas por Windows. Se verificaron 0 reintentos de IO, sin reintentos HTTP añadidos por ese mecanismo. El guard no estuvo presente desde el inicio; los fallos anteriores y sus observaciones permanecen íntegros en este resultado primario.

Los importes incluyen todos los intentos de las ramas seleccionadas, incluso fallos y reintentos. Los costes desconocidos no se convierten en cero. Son escenarios tarifarios sin caché, no una factura. El ahorro neto queda sin verificar porque falta monetizar cómputo y adquisición.

La referencia BERT usa el checkpoint publicado de RouteLLM, con otra supervisión y umbrales históricos congelados; conserva 120B si truncó el prompt. No es una ejecución del SDK oficial, no cubre todas las variantes RouteLLM y no se evaluó Not Diamond.

HumanEval y QuixBugs son fuentes públicas históricas potencialmente presentes en el preentrenamiento. Este piloto no demuestra preparación productiva, ausencia de deterioro poblacional ni resultados en repositorios de clientes. El gate anterior por workflow permanece intacto.
