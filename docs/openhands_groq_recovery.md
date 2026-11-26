Relay dev5 para OpenHands permite continuar una reparación cuando Groq devuelve HTTP 400 con el código estructurado `tool_use_failed`. El host debe habilitarlo en la fábrica del intento. La política experimental conserva el máximo de una rama fuerte y la inicia desde el original. Al agotarse el presupuesto, la tarea termina con los intentos registrados.

Para instalar el [wheel local](../dist/relay-harness/relay_harness-0.1.0.dev5-py3-none-any.whl) en el entorno Python 3.12 del agente que ya tiene OpenHands SDK 1.62.0, LiteLLM 1.105.7 y HTTPX 0.28.1, ejecutar desde la raíz del proyecto. En un entorno nuevo con acceso al índice de paquetes, el extra `[openhands]` del mismo wheel instala esas dependencias.

```powershell
python -m pip install --no-index --no-deps dist/relay-harness/relay_harness-0.1.0.dev5-py3-none-any.whl
python -m pip check
python -I -c "import relay_harness; print(relay_harness.__version__)"
```

Para verificar el wheel sin cambiar el entorno existente, el siguiente comando lo instala en una carpeta temporal y ejecuta una reparación con OpenHands y transporte HTTP simulado. Usar un destino nuevo. La salida persistente incluye los dos registros HTTP simulados, la identidad del wheel y la huella de cada archivo instalado. El ejemplo copiado fuera del checkout usa el paquete instalado.

```powershell
.\.venv-harness\Scripts\python.exe -I -B tools/verify_groq_recovery_wheel.py --output outputs/cascade-demo/my-wheel-check
```

En la [integración existente](openhands_cascade_usage.md), activar la opción en la llamada a `make_openhands_attempt`. El presupuesto y los nombres de modelos siguientes ilustran una tarea de hasta seis despachos y un máximo de salida de 1024 tokens por petición.

```python
from relay_harness import SpendBudget
from relay_harness.integrations.openhands_cascade import make_openhands_attempt

cheap = "groq/openai/gpt-oss-20b"
strong = "groq/openai/gpt-oss-120b"
attempt = make_openhands_attempt(
    agent_factory, prompt,
    reservation_usd_by_model={cheap: 0.005, strong: 0.010},
    tools_are_local=True, recover_tool_use_failed=True, max_iterations=3,
)
budget = SpendBudget(max_cost_usd=0.05, max_calls=6, max_seconds=600)
# Pasar attempt y budget a run_cascade con el snapshot y verificador del host.
```

La recuperación comprueba el destino y el cuerpo del POST. Exige chat síncrono directo de Groq, una sola generación, herramientas locales, nivel `on_demand`, máximo de salida explícito y las tarifas documentadas de ambos modelos. Una petición con uso íntegro se valora a esas tarifas. Si el rechazo identificado carece de uso, su `cost_usd` sigue en `null`. Relay registra el identificador de Groq cuando está disponible. El cuerpo de error, que puede incluir texto generado, queda fuera del registro económico. [Referencia de chat](https://console.groq.com/docs/api-reference#chat-create) y [errores de Groq](https://console.groq.com/docs/errors).

La cota tarifaria suma el precio de la ventana entera a tarifa de entrada sin caché y el del máximo de salida solicitado. Para 20B y 1024 tokens de salida asciende a USD 0,0101376. Para 120B en la misma configuración asciende a USD 0,0202752. El presupuesto ilustrativo de USD 0,05 permite admitir ambos despachos si no hubo gastos previos. Una reserva del host mayor también debe caber para el próximo despacho. Las cotas corresponden al escenario de tokens y tarifas fijado por la versión. La factura real y el costo del host requieren su propia evidencia. [Tarifas y límites de 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) y [120B](https://console.groq.com/docs/model/openai/gpt-oss-120b).

En `result.json`, `known_cost_subtotal_usd` suma los despachos con costo conocido, incluidos los fallidos. `cost_upper_bound_usd` suma el subtotal y los máximos justificables de los pendientes. Una petición de costo desconocido sin cota justificable deja también el máximo en `null` y bloquea más gasto. `cost_usd` solo contiene un total puntual cuando se puede sumar todo con una base común. Un rechazo local anterior al POST puede quedar en cero despachos. Si el SDK sale por una ruta de transporte que Relay no pudo observar, `http_calls_complete=false` y el gasto queda indeterminado aunque el contador observado marque cero. La aceptación independiente se evalúa después de cerrar la ruta. El [resultado histórico de las 43 tareas](../reports/relay-validacion-final-20261118.md) corresponde a dev4.
