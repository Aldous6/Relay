# Relay dev4: cascada verificable para OpenHands

La entrega `relay-harness 0.1.0.dev4` añade `run_cascade` a la integración existente. Primero ejecuta las pruebas del proyecto sobre una copia del original. Si no detectan el defecto, mantiene el modelo fuerte. Si detectan un fallo, permite intentar el barato y conserva su cambio solo si supera las comprobaciones. De lo contrario, inicia una única rama fuerte desde el original. Las pruebas iniciales inconclusas detienen la ejecución antes de llamar modelos.

El wheel está instalado en `.venv-harness`. La [verificación del paquete instalado](../reports/cascade-runtime-delivery-v1.json) comprobó los 16 archivos del manifiesto, las 11 fuentes históricas congeladas, la conservación del modelo fuerte en ambas validaciones anteriores rechazadas y el flujo completo con OpenHands real y respuestas simuladas. No hizo llamadas a proveedores. Pasaron 167 pruebas de cascada, integración, empaquetado y regresión; una prueba de enlaces simbólicos se omitió por los permisos de Windows. `pip check` no encontró dependencias incompatibles.

Esta regla sirve para reparación o implementación incompleta con pruebas declaradas. Detectar el fallo original no garantiza que las pruebas encuentren todos los errores de un parche. `operational_passed` significa que pasaron esas pruebas; `accepted` permanece `null` hasta una evaluación independiente. La cascada todavía es experimental y el modo predeterminado conserva el modelo fuerte.

## Comprobación local ejecutable

En el entorno autorizado ya existente, sin descargar dependencias:

```powershell
.\.venv-harness\Scripts\python.exe -I -B examples\relay_openhands_cascade.py
```

El ejemplo utiliza OpenHands 1.62.0 real con respuestas simuladas y red bloqueada. Ejecuta herramientas, crea dos conversaciones y dos copias limpias, comprueba el escalamiento y registra ambos consumos simulados. Los artefactos se guardan en una carpeta nueva de `outputs/cascade-demo/`. Es una comprobación de integración; no constituye evidencia de ahorro o calidad de los modelos.

## Conectar un harness existente

El host aporta una fábrica de agentes nuevos, sus herramientas locales y su prompt. La fábrica recibe `context.model`, `context.workspace` y `context.state_dir`; debe ligar las herramientas al workspace de ese intento. Conserva el SDK y los proveedores que ya usa. Una conversación inicializada no se reutiliza entre ramas.

```python
from pathlib import Path
import sys
from relay_harness import CheckCommand, CommandVerifier, SpendBudget, run_cascade
from relay_harness.integrations.openhands_cascade import make_openhands_attempt

def repair_with_relay(*, agent_factory, prompt, snapshot, output,
                      candidate, baseline, reservations, task_budget_usd):
    snapshot = Path(snapshot)
    protected = [str(path.relative_to(snapshot))
                 for path in (snapshot / "tests").rglob("*.py")]
    # Añadir aquí configuración/fixtures que controlen las pruebas del proyecto.
    if (snapshot / "pyproject.toml").is_file():
        protected.append("pyproject.toml")
    verifier = CommandVerifier.capture(
        snapshot,
        checks=[CheckCommand("project-tests", (
            sys.executable, "-I", "-B", "-m", "relay_harness.cascade_unittest",
            "--start-directory", "tests", "--min-tests", "1",
        ), timeout_seconds=60)],
        protected_files=protected,
    )
    attempt = make_openhands_attempt(
        agent_factory, prompt, reservation_usd_by_model=reservations,
        tools_are_local=True, max_iterations=3,
    )
    return run_cascade(
        snapshot_dir=snapshot, output_dir=output, task_id="repair-001",
        candidate_model=candidate, baseline_model=baseline,
        run_attempt=attempt, verifier=verifier,
        budget=SpendBudget(task_budget_usd, max_calls=6, max_seconds=300),
        mode="experimental", allow_experimental=True,
    )
```

`agent_factory(context)` debe devolver `Agent(llm=..., tools=..., condenser=None, critic=None, mcp_config={})`, con un `LLM` nuevo cuyo `model` coincida exactamente con `context.model`, `num_retries=0`, `api_mode="chat"`, `stream=False`, `max_output_tokens` y `timeout` explícitos y tarifas positivas `input_cost_per_token`/`output_cost_per_token`. Los secretos los conserva el mecanismo ya usado por el host; Relay no los solicita ni los incluye en su bitácora. La fábrica no genera respuestas ni inicia llamadas por su cuenta.

El runner incluido rechaza cero pruebas, errores al descubrir/importar tests y ejecuciones sin el mínimo de pruebas activas. Añade al path el workspace y su carpeta `src` cuando existe. Código 0 significa aprobación, 1 fallo de pruebas ejecutadas y 2 ejecución inconclusa. Se pueden usar otros runners con `CheckCommand`; `failure_returncodes` declara qué códigos representan fallos de pruebas y los demás se consideran inconclusos. El host conserva la responsabilidad de que sus pruebas comprueben realmente el código de la copia y cubran el contrato deseado.

La copia de entrada debe ser una instantánea preparada de la tarea, con código, pruebas y configuración necesarios, sin entornos virtuales ni secretos: como máximo 1000 archivos y 20 MB, sin enlaces ni junctions. `output` debe ser una ruta nueva fuera de esa copia. La implementación no aplica cambios al repositorio original; `selected_workspace` identifica el resultado para revisión e incorporación por el host. No es un sandbox del sistema operativo: las herramientas y pruebas son código del proyecto ejecutado con los permisos del usuario.

## Presupuesto y registros

La reserva por llamada es una estimación conservadora del host usando el contexto admitido, el máximo de salida y las tarifas configuradas. El control admite un despacho solo si cabe esa reserva; no es un límite de facturación impuesto al proveedor. Si el gasto observado supera la reserva, se registra completo y se bloquean nuevos despachos. Uso faltante, un error de proveedor sin costo verificable o un fallo al escribir la bitácora también bloquean la continuación. El tiempo máximo controla admisión; no cancela una petición ya enviada. Cada petición y cada comando tienen su propio timeout.

La integración cuenta cada entrada real al transporte del SDK, incluidas reparaciones internas. El costo del intento barato fallido se suma al fuerte. Se restringe esta versión a chat síncrono sin streaming, una conversación por proceso y herramientas locales sin servicios/LLMs auxiliares. Se rechazan rutas o límites que se intenten sobrescribir mediante `extra_body`. No se confunde el máximo de iteraciones del agente con el contador de despachos.

`manifest.json` conserva hashes del original, identidad del verificador y presupuesto. `transport.jsonl` tiene registros anteriores y posteriores a cada despacho. `attempt-N/attempt.json` conserva verificación, estado, tiempo e IDs de llamadas. `result.json` resume ambas ramas, costo conocido/incompleto, ruta final y `accepted=null`. La persistencia normal de OpenHands sí contiene la conversación; la bitácora de Relay contiene metadatos. El costo local queda desconocido salvo que se declare `cpu_hourly_usd`: ese parámetro produce un escenario aplicado al tiempo de pared total, no una medición de CPU ni ahorro facturado.

## Evidencia y alcance científico

El [análisis de desarrollo](../reports/openhands-cascade-development-v3.md) reconstruye la política sobre 24 tareas históricas con ramas reales ya adquiridas: 23/24 aceptadas, cero pérdidas pareadas frente a 120B y 12,68% menos costo tarifario por tarea aceptada. Se cuentan ambos intentos cuando escala. La política usa 120B directamente en siete casos cuyas pruebas no reproducen el defecto inicial.

Es una hipótesis elegida después de inspeccionar esos resultados. El intervalo descriptivo del 95% para el ahorro va de −11,40% a 33,35%; el límite exacto unilateral de daño sigue en 11,73%. No demuestra ahorro neto, no habilita automáticamente la cascada y no establece superioridad frente a RouteLLM o Not Diamond. El [protocolo científico](openhands_cascade_v1.md) separa desarrollo, comprobaciones operativas y evaluación final, fija comparación pareada, contabilidad completa, bootstrap y planificación de precisión.

La [validación prospectiva posterior](../reports/relay-validacion-final-20261118.md), congelada antes de adquirir respuestas, ya terminó en 43 tareas nuevas: Relay aceptó 42/43 y 120B fijo 43/43. Un HTTP 400 de 20B dejó consumo desconocido y el presupuesto bloqueó el escalamiento. No se confirmó ahorro sin pérdida de calidad. Los resultados originales permanecen intactos y el [contraste ampliado](../reports/relay-contraste-integral-20261118.md) distingue las pruebas de software de la evidencia económica. Pasar pruebas locales no activa la política ni convierte en conocidos los costos pendientes.

Se puede reproducir el análisis sin modelos ni red, conservando el informe existente:

```powershell
.\.venv-harness\Scripts\python.exe -I -B tools\analyze_cascade_development.py --output outputs\cascade-analysis-reproduction
```

No requiere BERT ni entrenar otro clasificador para ejecutar esta política. El experimento científico original y la adaptación dev3 permanecen disponibles; sus fuentes congeladas y resultados no se reescribieron.
