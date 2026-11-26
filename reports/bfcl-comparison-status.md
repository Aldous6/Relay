Actualización del 18 de noviembre de 2026: terminó la [validación prospectiva de Relay con OpenHands](relay-validacion-final-20261118.md) en 43 tareas nuevas. Relay aceptó 42/43 y 120B fijo 43/43; una petición de la cascada quedó con costo desconocido. No se confirmó ahorro verificable sin pérdida de calidad. Los 10 000 remuestreos y la auditoría independiente están documentados en el informe. El 12,68 % histórico permanece como desarrollo retrospectivo. No se reanudó BFCL; los procesos descritos abajo son históricos.

Actualización de dirección, 17 de noviembre de 2026: este documento conserva el corte histórico de BFCL; sus estados y procesos descritos abajo no son un estado vivo actual. La entrega de ese corte se concentraba en mantenimiento Python y en la [comparación completa nueva con RouteLLM-BERT](../outputs/maintenance-comparison/20261117-live-v3-complete/delivery_report.md). Esa comparación no demostró ahorro; el [cierre de trabajo](maintenance-live-delivery.md) registra resultados, corrección Windows y el alcance siguiente. En ese trabajo no se reanudó BFCL.

Relay: resultados históricos y continuación automática documentada.

Corte auditado del 13 de noviembre de 2026 a las 03:31:44 UTC. La revisión nueva sigue en ejecución sobre 19 conversaciones de desarrollo y tres controles. Estas tres conversaciones ya cerraron bajo las tres políticas; se comparan los mismos casos, incluidos los fallos.

| Política | Éxitos | Llamadas | USD, escenario tarifario sin caché |
| --- | ---: | ---: | ---: |
| Siempre fuerte, GPT-OSS 120B | 3/3 | 30 | 0.011682750 |
| Siempre barato, GPT-OSS 20B | 2/3 | 26 | 0.005468175 |
| Cascada reactiva 20B → 120B | 2/3 | 24 | 0.007654800 |

La cascada reduce 34.48% el escenario de tokens frente al fuerte, pero pierde un acierto. Cuesta 39.99% más que el barato con el mismo éxito observado. Este corte no satisface la conservación de calidad ni demuestra una ventaja competitiva. Tres conversaciones no permiten certificar un margen de dos puntos porcentuales. Los importes usan tarifas congeladas; no son facturación de Free ni ROI con todos los costos. El [informe auditado](../outputs/bfcl-comparison/20261112-development-v3/audit-cutoff-02/report.md) conserva cobertura, pendientes, costos, latencias y variación de respuestas; su [JSON sellado](../outputs/bfcl-comparison/20261112-development-v3/audit-cutoff-02/report.json) permite verificar el corte.

En el caso 99, la cascada falló antes de escalar: el barato pidió una aclaración y cerró el primer turno sin utilizar herramientas, sin generar un error público que activara la regla. El fuerte incorporado más tarde reparó el arranque del motor, pero no recuperó la omisión inicial. El [diagnóstico con hashes y trazas](../outputs/bfcl-comparison/20261112-development-v3/diagnostic-notes-case99.md) separa este fallo del barato fijo, que dejó el motor parado. Es un límite observado del control reactivo, no una demostración de que un predictor ya lo resuelva. Estos resultados de desarrollo no se convertirán en datos de ajuste.

La [revisión anterior](../outputs/bfcl-comparison/20261112-development-v2/audit-v1/report.md) conserva otros cuatro casos comunes: fuerte 2/4, barato 3/4 y cascada 3/4; escenario de tokens de la cascada 14.17% menor que fuerte y 26.25% mayor que barato. Cerró 14 de 72 episodios previstos y se detuvo por HTTP 400. Una reproducción separada identificó `output_parse_failed` y respaldó una corrección explícita del contrato de errores. No se reetiqueta el rechazo original ni se mezclan las tasas de ambas revisiones. El [protocolo](../docs/bfcl_comparison_protocol.md) conserva esa enmienda y las exclusiones.

El observador de continuación se inició y verificó en segundo plano: PID del worker 37980, arranque 2026-11-13 03:30:05 UTC, plazo para iniciar sesiones hasta 2026-11-16 03:30:05 UTC. Su [estado vivo](../outputs/bfcl-comparison/20261112-development-v3-auto/status.json) y [registro](../outputs/bfcl-comparison/20261112-development-v3-auto/worker.log) son la referencia operativa, no este corte estático. Al verificarse estaba en `awaiting_parent_close`, sin llamadas propias; la corrida original continuaba ejecutándose.

Al cerrarse la corrida, el observador delegará sólo los pares nunca intentados. El supervisor esperará al menos 24 horas ante cuota agotada o límite de asignación local, respetará un `Retry-After` mayor y generará un informe acumulado tras cada sesión. Otros fallos compartidos de infraestructura detendrán la continuación para revisión. Los episodios cerrados, censurados o con llamadas de estado incierto no se reenvían. Se mantiene el presupuesto de proveedor autorizado de US$0 sobre la cuenta declarada Free; no se activó facturación ni otro proveedor. El plazo incluye las esperas, por lo que no garantiza terminar todos los pendientes. Una sesión finita ya activa puede cerrar después del plazo.

El equipo debe permanecer encendido y conectado para que el proceso avance. Cambiar fuentes, helpers o tarifas congeladas detendrá la continuación; apagar el proceso no tiene reinicio automático. Para cancelar el observador y sus continuaciones, sin introducir una clave:

```powershell
.\.venv-groq\Scripts\python.exe tools\watch_bfcl.py --output outputs\bfcl-comparison\20261112-development-v3-auto --stop
```

Esa orden no cancela la corrida original que el observador vigila. Una llamada activa del supervisor puede esperar hasta el timeout HTTP antes de finalizar la cancelación. La credencial permanece en memoria; no está en archivos ni argumentos de proceso.

La [verificación de código](../outputs/bfcl-comparison/20261112-verification.json) registra 604 pruebas que pasaron en `.venv-groq` y 7 pruebas headless de la interfaz que pasaron en `.venv-relay`. El descubrimiento inicial no pudo importar Streamlit en el entorno Groq; esas siete pruebas se ejecutaron después en su entorno existente, sin servidor, revisión visual ni llamadas reales. No se presenta como una única ejecución verde. La [autocomprobación de referencias](../outputs/bfcl-comparison/reference-self-check.json) aceptó las 24 referencias de desarrollo en el runtime protegido, sin red; comprueba autoconsistencia local, no calidad de los modelos ni equivalencia completa con el evaluador oficial.

Estos resultados corresponden a controles de una adaptación local de BFCL. No hay todavía un predictor BFCL entrenado y validado, ni una comparación ejecutada de Relay aprendido contra RouteLLM o Not Diamond con este catálogo. La recolección automática produce evidencia sobre candidatos, errores y costos; no transforma una cascada heurística en un resultado de ML.
