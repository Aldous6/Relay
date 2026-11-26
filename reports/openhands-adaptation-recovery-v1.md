Análisis secundario de recuperación de infraestructura: 1 episodio(s) fallaron por PermissionError antes de cualquier llamada lógica o despacho HTTP y recibieron un único intento aislado. Se incluyeron todos los casos elegibles mediante esa regla, sin seleccionar por respuesta o ruta. No es una nueva prueba independiente ni una política reajustada. El resultado primario y sus fallos permanecen intactos.

La evaluación secundaria no cumple el gate; el cargador debe conservar baseline con este informe. Esta conclusión secundaria no cambia la aprobación del resultado primario.

| Política | Aceptadas | Coste LLM total | Coste por aceptada | Pérdidas pareadas frente a 120B | Ahorro por aceptada frente a 120B |
|---|---:|---:|---:|---:|---:|
| 120B fijo | 23/24 | USD 0.00776850 | USD 0.00033776 | 0 | 0.00% |
| 20B fijo | 22/24 | USD 0.00516165 | USD 0.00023462 | 2 | 30.54% |
| Relay adaptado | 23/24 | USD 0.00722580 | USD 0.00031417 | 0 | 6.99% |
| BERT umbral 0.50 | 23/24 | USD 0.00776850 | USD 0.00033776 | 0 | 0.00% |
| BERT umbral 0.58 | 23/24 | USD 0.00722580 | USD 0.00031417 | 0 | 6.99% |



Relay seleccionó 20B en 5 tareas y 120B en 19. El límite superior Wilson unilateral del 95% para pérdidas pareadas observado es 10.13%; es descriptivo y no demuestra no inferioridad poblacional.

Motivos registrados por el gate: insufficient_savings_per_accepted_task.

Frente a 20B fijo, Relay presenta 1 pérdidas y 2 ganancias pareadas; su variación de coste por aceptada es -33.90% de ahorro. La diferencia es propia de esta muestra y no acredita superioridad general.

Frente a BERT umbral 0.50, Relay presenta 0 pérdidas y 0 ganancias pareadas; su variación de coste por aceptada es 6.99% de ahorro. La diferencia es propia de esta muestra y no acredita superioridad general.

Frente a BERT umbral 0.58, Relay presenta 0 pérdidas y 0 ganancias pareadas; su variación de coste por aceptada es 0.00% de ahorro. Las decisiones son idénticas: esta muestra no demuestra una ventaja incremental de routing frente a ese control.

El routing residente reconstruido suma 21.7555 s: 21.7481 s de encoder medidos anteriormente y 0.0074 s de replay de la cabeza con caché. La carga del encoder agrega 54.7696 s. Estos tiempos no son latencia end-to-end.

El precio horario de CPU de equilibrio, condicionado a ese ahorro LLM y al tiempo residente reconstruido, es USD 0.089803/h. Excluye carga, entrenamiento y adquisición; no es una tarifa medida ni ahorro neto.

El ajuste y selección automáticos registraron 4.5582 s; el trabajo histórico del encoder se conserva aparte en el JSON. No se supone amortización ni precio de cómputo.

La adquisición de las dos ramas conserva 53 despachos HTTP y 0 costes desconocidos; subtotal tarifario conocido USD 0.01293015, total USD 0.01293015. Este gasto de adquisición se informa separado del coste de ejecutar cada política.

Los importes incluyen todos los intentos de las ramas seleccionadas, incluso fallos y reintentos. Los costes desconocidos no se convierten en cero. Son escenarios tarifarios sin caché, no una factura. El ahorro neto queda sin verificar porque falta monetizar cómputo y adquisición.

La referencia BERT usa el checkpoint publicado de RouteLLM, con otra supervisión y umbrales históricos congelados; conserva 120B si truncó el prompt. No es una ejecución del SDK oficial, no cubre todas las variantes RouteLLM y no se evaluó Not Diamond.

HumanEval y QuixBugs son fuentes públicas históricas potencialmente presentes en el preentrenamiento. Este piloto no demuestra preparación productiva, ausencia de deterioro poblacional ni resultados en repositorios de clientes. El gate anterior por workflow permanece intacto.

En cada episodio recuperado se conserva el fallo inicial y se suman sus tiempos medidos a los del intento aislado; no se incluye la espera entre ejecuciones. El cero inicial corresponde exclusivamente a llamadas LLM cuya ausencia se verificó en llamadas lógicas y bitácoras HTTP; el trabajo local no se considera gratis. La enmienda de IO se añadió después del prefijo inicial de la adquisición, no estuvo activa desde el comienzo.
