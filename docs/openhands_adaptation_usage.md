Relay 0.1.0.dev3 adapta la elección entre dos modelos a resultados de tareas de un agente OpenHands existente. Aprende y calibra con desarrollo, congela un archivo de política y sólo aplica la alternativa cuando un informe independiente cumple el criterio declarado. Si falta validación o falla el criterio, mantiene el modelo base automáticamente. El informe de esta entrega está en [openhands-adaptation-v1](../reports/openhands-adaptation-v1.md); no debe confundirse una instalación funcional con ahorro demostrado.

Resultado de esta entrega: el primario conserva un fallo local anterior al primer despacho. El [análisis secundario con recuperación aislada](../reports/openhands-adaptation-recovery-v1.md) obtuvo 23/24 aceptadas tanto para Relay como para 120B fijo, con 6.99% menos costo tarifario por aceptada. Las decisiones coincidieron con BERT 0.58. El mínimo declarado era 10%, por lo que ninguno de los dos informes habilita la alternativa; cargar cualquiera conserva 120B sin cargar el encoder. No se modificó ese criterio después de observar resultados. El ahorro neto y una ventaja competitiva propia siguen sin demostrarse.

Se requiere Python 3.12 o posterior para OpenHands. El wheel local se construye con `python tools/build_harness_runtime.py`. En el entorno de tu agente, instala la integración y, si vas a usar la representación BERT, su motor opcional:

```powershell
python -m pip install "./dist/relay-harness/relay_harness-0.1.0.dev3-py3-none-any.whl[openhands,adapt,routellm-bert]"
```

En este equipo ya se instalaron, con autorización explícita, torch 2.9.0, transformers 4.57.1 y scikit-learn 1.8.0 en `.venv-harness`, junto con OpenHands 1.62.0 y Relay dev3. `pip check` no detectó conflictos y pasaron 58 pruebas de adaptación e integración. La [verificación del entorno unificado](../reports/unified-harness-environment-v1.json) cargó los pesos BERT reales, reprodujo el ajuste histórico y las 24 rutas congeladas, y conectó ambas rutas a Agent/RouterLLM con transporte simulado y red bloqueada. No recoge nueva evidencia de calidad ni modifica la validación anterior. Para usar este entorno desde PowerShell: `.\.venv-harness\Scripts\Activate.ps1`.

Los pesos no se distribuyen en el wheel. En este proyecto ya están en `artifacts/routellm-bert`; Relay verifica sus hashes antes de cargarlos y no los descarga automáticamente. La variante estructural y las políticas que mantienen baseline no requieren cargar BERT. scikit-learn se usa al adaptar, no al ejecutar una política exportada. BERT es un componente preentrenado de RouteLLM; la capa aprendida, calibración, integración y contabilidad son de Relay. No se afirma una arquitectura original ni una superioridad general sobre ese proyecto.

Para conectar el resultado al agente local que ya tienes, usa el objeto `agent` configurado pero todavía sin iniciar su conversación, un diccionario `configured_llms` con instancias LLM nuevas por tarea y el texto `request` que realmente enviarás a la conversación. No reutilices un Agent ya inicializado: OpenHands conserva estado y herramientas internas ligados a su conversación previa. No cambia las herramientas configuradas del agente. Cada LLM debe tener `num_retries=0` y `fallback_strategy=None`; las capacidades y el contexto admitidos se verifican en tu aplicación. En el piloto de Groq también se desactivan `supports_prompt_cache_key` y `supports_prompt_cache_retention` mediante capability_overrides, igual que en el runner comprobado.

```python
from relay_harness import load_adapted_relay
from relay_harness.integrations.openhands import attach_openhands_agent

relay = load_adapted_relay(
    "outputs/harness-adaptive/v1/policy-auto.json",
    "outputs/harness-adaptive/v1/run/validation.json",
    scope="openhands-python-v1",
    checkpoint_dir="artifacts/routellm-bert",
    log_path="outputs/my-agent/relay-events.jsonl",
)
task, routed_agent = attach_openhands_agent(
    agent, relay, workflow="function-implementation",
    models=configured_llms, routing_text=request, max_calls=3,
)
# Usa routed_agent en tu Conversation existente y ejecuta tus pruebas.
# accepted viene de esas pruebas, nunca de que el modelo declare éxito.
task.finish(
    accepted=accepted,
    additional_cost_usd=auxiliary_llm_cost,
    cost_complete=every_attempt_accounted_for,
)
```

Los valores `accepted`, `auxiliary_llm_cost` y `every_attempt_accounted_for` pertenecen a tu verificador/contabilidad. Si no conoces algún gasto o reintento, deja cost_complete=False; Relay no lo convierte en ahorro. Las herramientas externas, subagentes y condensadores también pueden consumir modelos: incluye su gasto si no pasó ya por la misma sesión, sin duplicarlo. Las conversaciones remotas o restauradas desde un perfil serializado no están incluidas en esta integración.

En Windows se reprodujo un bloqueo transitorio de la sustitución atómica del archivo de estado del SDK antes de llamar al modelo. Para conversaciones locales puedes envolver su ejecución con `retry_openhands_writes`: repite sólo esa escritura, como máximo cinco intentos, sin volver a llamar al modelo ni reiniciar la tarea. Usa la carpeta de persistencia propia de la conversación; el parche está activo temporalmente en el proceso, restringido a ese directorio, y rechaza contextos solapados. No se conoce qué proceso retuvo el archivo en el fallo original del piloto.

```python
from relay_harness.integrations.openhands_io import retry_openhands_writes

with retry_openhands_writes(conversation_persistence_dir):
    conversation.send_message(request)
    conversation.run()
```

La tarea conserva el modelo elegido durante todas sus llamadas. El registro contiene decisiones y uso, nunca prompts, respuestas ni claves. La adaptación necesita un archivo local distinto que contenga el texto autorizado y ambos resultados completos por tarea. Formato por registro: task_id, group_id, prompt_text, model, accepted booleano, cost_usd numérico o null y cost_basis (`tariff_estimate`, `provider_reported` o `unknown`). Puedes entregar una lista JSON, JSONL o un objeto con `records`. Los dos candidatos deben haber ejecutado el mismo texto; los intentos y fallos se agregan dentro del costo de cada episodio. Una sola rama elegida en producción no permite saber qué habría hecho el otro modelo.

Para adaptar a datos nuevos de TU integración, el siguiente comando decide entre las dos representaciones disponibles sin que tengas que fijar pesos ni umbrales. Guarda también `policy.json.selection.json` con todos los candidatos y su selección. Sin --checkpoint sólo evalúa la variante estructural. El alcance debe describir el harness, sus verificadores, catálogo y configuración; otro entorno necesita su propia evidencia.

```powershell
relay-harness adapt development.json --baseline groq/openai/gpt-oss-120b --candidate groq/openai/gpt-oss-20b --scope openhands-python-v1 --checkpoint artifacts/routellm-bert --output policy.json
relay-harness validate-adaptation policy.json fresh-validation.json --checkpoint artifacts/routellm-bert --output validation.json
```

Nunca ajustes otra vez esa versión mirando los resultados de fresh-validation.json. Una futura adaptación convierte resultados ya vistos en desarrollo y requiere otra reserva. El artefacto y el informe están vinculados por hashes de integridad; no son una certificación de que los datos suministrados por terceros sean auténticos. El criterio predeterminado del piloto es 20 grupos nuevos, al menos 10% menos costo por aceptada y cero pérdidas pareadas observadas. Superarlo no demuestra no inferioridad estadística ni asegura ahorro futuro.

La entrega local conserva 24 tareas históricas como desarrollo y 24 nuevas para la comparación acotada. `tools/adapt_openhands.py --output otra-policy.json` reproduce su adaptación offline con los scores BERT previamente verificados. `tools/run_adaptive_pilot.py --suite data/imported/harness-adaptive-v1/fresh_suite.json --policy outputs/harness-adaptive/v1/policy-auto.json --output outputs/harness-adaptive/v1/run --report-only` reconstruye el informe a partir de los episodios ya registrados, sin APIs. El mismo runner sin --report-only adquiere los episodios que falten usando la cuenta Groq Free autorizada, entrada oculta y límites explícitos; no repite episodios completados ni descarta interrupciones. No reutilices esta reserva para validar otra selección.

El ahorro principal del informe es un escenario de tarifas LLM que incluye fallos y todos los despachos auditados. Groq Free no equivale a esa factura. El precio del CPU, carga del encoder, entrenamiento y adquisición de evidencia se presentan aparte; sin monetizarlos no se declara ahorro neto. El reporte compara también 20B fijo y las dos referencias BERT congeladas, con idénticas tareas y modelos.
