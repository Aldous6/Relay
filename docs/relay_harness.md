# Relay para harnesses existentes

La entrega actual es una librería local que conecta la selección de modelos con los resultados del harness. El desarrollador conserva su agente, herramientas, clientes y verificador. Relay escoge una vez al iniciar cada tarea, conserva ese modelo durante la conversación y registra metadatos de costo y aceptación. La versión 0.1.0.dev3 añade [adaptación automática, validación y carga en OpenHands](openhands_adaptation_usage.md), manteniendo el selector experimental y las políticas anteriores por actividad. El núcleo también funciona con un cliente Python síncrono mediante un callback.

La nueva comparación y su recuperación de infraestructura están cerradas: [primario](../reports/openhands-adaptation-v1.md) y [secundario](../reports/openhands-adaptation-recovery-v1.md). El secundario obtuvo 6.99% menos costo LLM tarifario por aceptada con 23/24 aceptaciones, igual que 120B, pero empató con BERT 0.58 y no alcanzó el mínimo del 10%. La carga automática conserva baseline; no se afirma ahorro neto ni ventaja competitiva propia.

El beneficio buscado es reducir el costo de tareas aceptadas conservando calidad. El [piloto anterior con llamadas reales](../reports/harness-live-public-v2.md) encontró 0 % de ahorro para Relay por actividad frente a 120B fijo: ambos aceptaron 11/12 tareas reservadas. BERT transferido y calibrado obtuvo 11/12 con 11.51 % menos costo LLM tarifario por aceptación. La nueva versión incorpora ese motor a la ejecución por prompt, sin convertir el resultado previo en una nueva validación ni en una ventaja propia sobre RouteLLM. Todavía no demuestra ahorro neto o generalización a tareas arbitrarias. Los resultados científicos anteriores permanecen intactos.

## Instalar

Desde este repositorio, el siguiente comando construye el wheel local sin descargas:

```powershell
python tools/build_harness_runtime.py
python -m pip install --no-deps dist/relay-harness/relay_harness-0.1.0.dev3-py3-none-any.whl
relay-harness --help
```

El paquete `relay_harness` requiere Python 3.11 o posterior y no instala dependencias científicas. Se puede copiar el wheel al proyecto del usuario. Incluye un manifiesto SHA-256 de sus fuentes. Es una distribución local preliminar, aún no publicada; la licencia para distribución pública sigue pendiente. El paquete científico `routerbench-icd` y su API `relay_router` se conservan.

## Selección por prompt disponible

Para usar el cliente de consultas independientes de este repositorio con el entorno BERT ya preparado:

```powershell
.\.venv-routellm\Scripts\python.exe tools\relay_prompt_chat.py
```

El cliente mantiene el clasificador cargado, muestra el modelo elegido y llama una vez a Groq con la consulta. Acepta `GROQ_API_KEY` del entorno o solicita una clave oculta; no la guarda. `--route-only` selecciona sin generar ni pedir clave. `--prompt-file consulta.txt` procesa una consulta UTF-8; `--log llamadas.jsonl` guarda únicamente metadatos. Cada consulta es independiente; no simula un historial de chat. Los errores se conservan sin reintentos automáticos. Las tarifas del cliente son un escenario sin caché, no facturación de Groq Free ni costo neto del router.

El perfil utiliza exclusivamente `groq/openai/gpt-oss-20b` y `groq/openai/gpt-oss-120b`, con corte 0.58 fijado sólo en desarrollo. El score representa preferencia transferida desde GPT-4/Mixtral, no probabilidad de resolver la tarea. Cuando el texto excede los 512 tokens que puede leer BERT, el selector conserva el modelo fuerte; esto no verifica por sí solo el contexto o las capacidades del generador.

En un harness existente, conecta el motor a tu cliente:

```python
from relay_harness import Relay
from relay_harness.integrations.routellm import load_groq_pilot_selector

selector, startup = load_groq_pilot_selector("artifacts/routellm-bert")
relay = Relay(selector.strong_model, prompt_selector=selector,
              mode="apply", allow_experimental=True, log_path="relay.jsonl")
task = relay.task("coding", routing_text=prompt,
                  eligible_models=[selector.weak_model, selector.strong_model])
response = task.call(generate, prompt=prompt, measure=measure)
task.finish(accepted=accepted, cost_complete=all_model_costs_accounted_for)
```

`generate`, `measure` y el veredicto siguen perteneciendo a tu aplicación. Para OpenHands, el mismo `task` se entrega a `make_openhands_router`, o se pasa `routing_text` a `attach_relay` del ejemplo. Los mensajes y herramientas reales del agente no se modifican. En `mode="observe"` se registra la sugerencia y se ejecuta el baseline; aplicar este perfil exige `allow_experimental=True` porque no cumple todavía el contrato de una política productiva aprobada. No se combina con una política por actividad ni con `model_override`.

La implementación del selector está en el wheel ligero; BERT es opcional y requiere `torch==2.9.0` y `transformers==4.57.1` (extra `routellm-bert`). Los pesos no se distribuyen en el wheel ni se descargan automáticamente: se verifica la revisión local `86237e3df400762178ea98379477b8296e66d5e4`, sus seis archivos y la ausencia de overrides auxiliares. El modelo se carga una vez por proceso. `startup` conserva tiempo de carga y procedencia; la decisión registra tiempo por solicitud. El costo monetario de ese cómputo permanece desconocido y se excluye explícitamente del costo conocido de las llamadas.

La [comprobación real del motor](../reports/prompt-runtime-v2.json) reprodujo las 24 decisiones congeladas sin generar nuevas respuestas. En la muestra anterior, las tres tareas reservadas enviadas a 20B conservaron aceptación; el ahorro total de llamadas fue sólo USD 0.00039465 en doce tareas, antes de valorar cómputo local. No se ajustó el corte con esa reserva ni se renombraron modelos para trasladar el perfil a Ollama. El adaptador implementa la [regla BERT publicada de RouteLLM](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/routers.py); se mantiene esa atribución.

## Conectar el cliente actual

```python
from relay_harness import Relay

relay = Relay(baseline_model="mi-modelo-actual", log_path="relay.jsonl")
task = relay.task("python-repair", task_id="issue-143", max_calls=20)

# generate(model, ...) llama al cliente que ya utiliza tu harness.
# measure(response) devuelve solamente costo y uso conocidos.
response = task.call(generate, messages=messages, measure=measure)

# Reutiliza task en todos los turnos de ESTA tarea.
# El verificador existente aporta True, False o None (desconocido).
observation = task.finish(accepted=accepted, cost_complete=all_costs_accounted_for)
```

`measure` puede devolver `cost_usd`, `cost_basis`, `input_tokens`, `output_tokens` y `cached_input_tokens`. Las bases permitidas son `provider_reported`, `tariff_estimate` y `unknown`; un costo desconocido se omite o vale `None`. La función devuelve la respuesta original sin modificarla y no añade llamadas, reintentos ni consultas a otro LLM. Este helper es síncrono; para un cliente asíncrono se usan `begin_call()` y `record_call()` alrededor del `await`.

`finish(cost_complete=True)` es una declaración del harness de que contabilizó todas las llamadas, reintentos y modelos auxiliares. Los costos adicionales se declaran con `additional_cost_usd`, en la misma base que los costos registrados y sin duplicarlos. Si falta información, se deja `cost_complete=False`; el total queda desconocido y no justifica ahorro. `max_calls` limita despachos lógicos, no solicitudes HTTP ocultas del proveedor ni gasto monetario. Usa IDs opacos y etiquetas de actividad; no pongas prompts, secretos ni mensajes de error en esos campos.

## OpenHands local

La integración está verificada contra `openhands-sdk==1.62.0`, Python 3.12 y LiteLLM 1.105.7. Usa un entorno separado del extra científico `api`, que fija otra versión de LiteLLM. En ese entorno:

```powershell
python -m pip install "dist/relay-harness/relay_harness-0.1.0.dev3-py3-none-any.whl[openhands]"
python examples/relay_openhands.py
```

El ejemplo comprueba la conexión con las clases reales del SDK sin pedir claves ni llamar modelos. En tu harness, configura tus LLM con `num_retries=0` y sin fallback ni hooks de renovación de credenciales. Crea instancias nuevas por tarea y conecta el router al agente:

```python
from relay_harness import Relay
from relay_harness.integrations.openhands import make_openhands_router

relay = Relay(baseline_model=baseline, log_path="relay.jsonl")
task = relay.task("python-repair", eligible_models=list(configured_llms))
router = make_openhands_router(task, configured_llms)
routed_agent = agent.model_copy(update={"llm": router})
# Usa routed_agent en tu Conversation local existente y ejecútala normalmente.
# Al terminar, registra el veredicto de tus pruebas/revisor y el costo completo.
task.finish(accepted=accepted, cost_complete=all_costs_accounted_for)
```

`configured_llms` es un diccionario `{nombre_exacto_del_modelo: instancia_LLM}` con las capacidades que necesita tu agente. El adaptador conserva mensajes, herramientas y callbacks de streaming, e intercepta Chat Completions y Responses, síncronos y asíncronos. No admite serializar la sesión viva para un agente remoto. Usa conversaciones locales y secuenciales por tarea. Si un condensador o subagente tiene su propio LLM, incluye su costo auxiliar; si usa el mismo router, ya queda registrado. Relay no modifica el logging del SDK. El SDK puede hacer reparaciones internas de caché: si no puedes confirmar el costo de todos los intentos, deja el costo completo como desconocido.

La API se contrastó con el SDK instalado; la guía oficial explica [routing](https://docs.openhands.dev/sdk/guides/llm-routing) y [métricas](https://docs.openhands.dev/sdk/guides/metrics). La clase utilizada por esta versión es `RouterLLM`.

Para GPT-OSS en Groq, el piloto verificó esta configuración del cliente OpenHands. `groq_key` debe proceder del mecanismo de secretos de tu aplicación, nunca de un literal en el código:

```python
from openhands.sdk.llm import LLM

configured_llms = {
    name: LLM(
        model=name, api_key=groq_key, api_mode="chat", num_retries=0,
        caching_prompt=False,
        capability_overrides={
            "supports_prompt_cache_key": False,
            "supports_prompt_cache_retention": False,
        },
    )
    for name in ("groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b")
}
```

Sin esos overrides, el SDK 1.62.0 añadió `prompt_cache_key` y Groq rechazó la petición en el diagnóstico conservado. Desactivar `caching_prompt` por sí solo no evitó ese rechazo. Los errores de formato de herramientas de 20B permanecieron contabilizados como fallos; la configuración anterior no garantiza eliminarlos.

## Preparar la integración con Ollama

El laboratorio es el candidato para un uso real con Ollama. Todavía faltan su tarea habitual, aplicación, catálogo exacto y hardware. Relay ya dispone de una interfaz de selección por prompt, pero su perfil Groq no está calibrado para los modelos locales del laboratorio. La política anterior por actividad valida costos en dólares y tampoco sustituye una política de tiempo/recursos locales. Los bundles científicos anteriores no se pueden transferir renombrando modelos.

El [diagnóstico portátil](../tools/relay_ollama_inventory.py) permite obtener el catálogo sin instalar Relay ni dependencias científicas. Requiere Python 3.11 o posterior y un servidor Ollama ya disponible. Copia únicamente ese archivo al equipo del laboratorio y, desde la carpeta donde lo guardes, ejecuta:

```powershell
python relay_ollama_inventory.py --output laboratorio-ollama.json
```

Por defecto consulta `http://127.0.0.1:11434`. Si Ollama está en otro equipo, el responsable puede pasar su URL base mediante `--url`; el diagnóstico no cambia ninguna configuración de red ni inicia servidores. El archivo de salida debe ser nuevo. No hay inferencia, descargas, exploración de archivos del laboratorio ni llamadas a proveedores de generación.

El informe recoge versión de Ollama, nombres y digests de modelos, tamaño, cuantización, capacidades anunciadas y contexto declarado cuando están disponibles. Consulta por separado cuáles están cargados, su contexto activo y la memoria VRAM asignada que informa Ollama. Si una consulta opcional falla, conserva el inventario parcial con una advertencia; un dato desconocido no se transforma en cero. No guarda plantillas, mensajes de sistema, Modelfiles ni la dirección del servidor. Sí incluye nombres de modelos, que el responsable puede revisar antes de compartir.

Estos campos proceden de [listar modelos](https://docs.ollama.com/api/tags), [listar modelos cargados](https://docs.ollama.com/api/ps), [detalles del modelo](https://docs.ollama.com/api-reference/show-model-details.md) y [versión](https://docs.ollama.com/api-reference/get-version.md). La VRAM asignada no es la capacidad total de la GPU; los límites declarados no garantizan contexto activo; aparecer en el catálogo no certifica calidad ni ejecución exclusivamente local. El diagnóstico no propone rutas.

El responsable todavía debe aportar una tarea frecuente, la aplicación desde la que la realiza, cómo reconoce una respuesta correcta, el modelo que usa habitualmente y su GPU/RAM. Esa información define la integración y el verificador. El objetivo local propuesto es reducir tiempo por tarea aceptada, incluyendo carga, decisión y reintentos; dólares y energía requieren medición específica. La selección por prompt necesita evidencia del catálogo concreto y evaluación separada; el inventario prepara ese trabajo, no demuestra ahorro.

## Pasar de observar a aplicar

```powershell
relay-harness report relay.jsonl
relay-harness propose development.jsonl --baseline mi-modelo-actual --output proposal.json
relay-harness validate proposal.json validation.jsonl --output policy.json
```

Observar producción muestra costo y aceptación del modelo ejecutado; no revela qué habría hecho otro modelo. Para comparar, ejecuta explícitamente los candidatos sobre las mismas tareas de desarrollo con el mismo harness y verificador. Usa `task(..., split="development", evaluation_id="harness-v1", task_id=..., group_id=..., model_override=...)`; comparte task_id y group_id entre los modelos de una tarea. Reserva grupos nuevos para `split="validation"`, con el mismo evaluation_id. Nunca reutilices una tarea reservada para elegir otra política después de consultar su resultado. Los reintentos de una tarea pertenecen a su costo total, no a nuevas observaciones exitosas.

La propuesta selecciona candidatos únicamente en desarrollo y congela esa selección. La validación evalúa esa elección sobre grupos distintos. Por defecto exige al menos 20 grupos en cada conjunto, ahorro observado de al menos 10 % y ninguna caída observada en aceptación. Son criterios empíricos, no garantías estadísticas ni de producción. Costos incompletos, pares ausentes o evidencia insuficiente conservan el baseline. No mezcles tarifas estimadas con facturación; el reporte muestra ambas por separado. Los datos de ejemplo de las pruebas son sintéticos y no demuestran beneficio.

Para activar una política aprobada al iniciar las tareas siguientes:

```python
relay = Relay(baseline_model=baseline, policy="policy.json", mode="apply",
              log_path="relay.jsonl")
```

Con `mode="observe"`, Relay sigue ejecutando el baseline y registra qué sugería la política. Volver a ese modo revierte el cambio en tareas nuevas. Las tareas en curso conservan su selección. Las actividades nuevas o sin ruta aprobada usan el baseline. `eligible_models` permite excluir modelos que no cumplen capacidades; no infiere compatibilidad por nombre. Si cambia el catálogo, el harness o su verificador, recoge evidencia nueva y genera otra política.

El ledger es local y solo contiene los campos permitidos. Cada proceso debe usar su propio archivo; reúne observaciones terminadas después. No elimines tareas fallidas o pendientes de una comparación. `report` muestra pendientes y `propose`/`validate` rechazan un ledger inconcluso. Los identificadores duplicados se rechazan al preparar la política. El hash de la política detecta alteraciones accidentales, no autentica al autor ni convierte registros declarados en evidencia confiable.

El siguiente criterio de producto es una instalación en un harness de uso real, con tareas repetidas, modelos capaces y aceptación externa. Ahí se medirá costo por tarea aceptada junto con tasa de aceptación, costo total y latencia; la comparación competitiva deberá ejecutar las alternativas bajo el mismo contrato. Los ocho casos históricos de mantenimiento siguen siendo diagnóstico, no validación de esta nueva política.

La [verificación inicial del SDK](../reports/harness-runtime-delivery.json) registra 76 pruebas aprobadas, instalación del wheel y una conversación local completa de OpenHands con transporte simulado. Esa comprobación no llamó modelos. El [piloto posterior](harness_public_pilot_v1.md) sí ejecutó OpenHands con Groq: conserva cada despacho HTTP, los fallos, el plan congelado antes de validación y la verificación reservada fuera del contexto del modelo. El runner es infraestructura de investigación del repositorio; no forma parte del wheel ligero ni convierte su política exploratoria en una política aprobada para producción.
