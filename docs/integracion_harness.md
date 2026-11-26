Relay 0.8.1 se integra como librería Python y conserva los bundles v6/v7. El harness conserva clientes de LLM, herramientas, historial, control de contexto y reintentos; Relay recibe el texto serializado completo y devuelve el modelo elegido. Streamlit es una demo separada y no una dependencia de esta API. La nueva [guía de evaluación de agentes](evaluacion_agentes.md) describe `AgentRequest`, `ModelReply`, políticas por paso y episodios con herramientas. Es infraestructura de evaluación; no hay un router v8 entrenado ni ahorro productivo acreditado.

Instalación local desde la raíz: `python -m pip install .`. El nombre de distribución existente sigue siendo routerbench-icd, versión 0.8.1; el import público es relay_router. No hay publicación en PyPI ni licencia elegida para distribución pública. Un entorno sin setuptools necesita resolver las dependencias de construcción declaradas en pyproject.toml. Las dependencias científicas permanecen; reducirlas será trabajo de empaquetado posterior.

```python
from relay_router import load_bundle

router = load_bundle("reports/relay-v7-development-v1")
decision = router.route("Instrucciones, historial y solicitud serializados por el harness")
print(decision.to_dict())

# Tu función debe aceptar el nombre exacto y el mismo prompt, y devolver la respuesta.
# result = router.run(prompt, generate=mi_cliente_de_modelos)
```

`route()` no llama proveedores. `run()` decide primero y llama una vez a `generate(model_name, prompt)`. No reintenta ni cambia de modelo después de observar una respuesta. El callback recibe el prompt completo sin recorte. `available_models` permite que el harness restrinja las opciones según disponibilidad o contexto; si solo queda una, la decisión registra availability_fallback y no conserva implícitamente el alcance de calidad de la política completa. Si ninguna es viable, falla antes de generar.

La política `pairwise_utility` de v7 estima `strong_only`, `weak_only` y `equal`: solo el fuerte cumple el objetivo, solo el débil lo cumple o ambos tienen el mismo resultado binario. El objetivo histórico es score de RouterBench ≥ 0.75. Calcula `expected_quality_gain = P(strong_only) - P(weak_only)` y `operating_score = expected_quality_gain / (costo_fuerte_previsto - costo_débil_previsto)`. El costo adicional debe ser estrictamente positivo. Elige el fuerte cuando `operating_score > tradeoff`; el empate elige el débil. `tradeoff` es un umbral del cociente, puede ser negativo y no es una probabilidad ni una garantía de calidad. Se carga del bundle; cambiarlo altera la política evaluada.

`decision.to_dict()` conserva `selected_model`, `reason`, `policy_mode`, `router_seconds`, `predicted_cost`, `cost_unit` y `evaluation_scope`. V7 añade `expected_quality_gain` y `operating_score`; los motivos normales son `utility_strong` y `utility_weak`. Los campos `harm_probability` y `weak_success_probability` son `None` con esta política. En `fixed_strong` o `availability_fallback` no se ejecuta el predictor y las estimaciones quedan en `None`.

La versión 0.8.1 conserva compatibilidad con bundles v6 de esquema 1 y política `guarded`, incluidos sus campos `max_harm`, `min_weak_success` y probabilidades de auditoría. Esos dos umbrales no gobiernan `pairwise_utility`. El modo `fixed_strong` también sigue disponible; un bundle que lo declare conserva ese modo al cargarlo.

Los costos del bundle de RouterBench son históricos (`cost_unit="historical_cost"`) y no precios de API actuales. El harness debe medir sus propios tokens, gasto, fallos y tiempo de tarea. Ni las estimaciones ni los umbrales garantizan un resultado individual; el alcance del bundle sigue siendo desarrollo histórico reutilizado, sin validación independiente de productividad o ahorro neto en agentes.

El bundle contiene un manifiesto `bundle.json` de esquema 1 y el artefacto indicado por `artifact`; v7 usa `neighborhood_primary.joblib`. Para `pairwise_utility` el manifiesto incluye `tradeoff` finito, además del par exacto y el hash. `load_bundle` acepta la carpeta o el manifiesto. Solo cargar bundles de autores confiables: joblib puede ejecutar código al deserializar. Se verifica SHA-256 y confinamiento de la ruta antes de cargar los mismos bytes. El hash comprueba integridad respecto al manifiesto, no la identidad o seguridad de su autor. El artefacto v7 conserva una matriz de vecinos derivada del conjunto de ajuste; sus derechos y procedencia requieren revisión antes de distribuirlo.

El candidato v7 está entrenado para el par histórico GPT-4-1106-preview/Mixtral. No se puede cambiar su catálogo por Qwen, Claude u otro modelo y atribuirle las mismas métricas. Para un catálogo propio se necesita evidencia de sus versiones concretas, ajuste/calibración disjuntos y prueba independiente; el constructor RelayRouter admite un decider con predict_tables y el par exacto para ese trabajo. La API actual selecciona entre dos candidatos. `RelayPolicy` permite invocarla por paso dentro del nuevo evaluador y congela un fingerprint del predictor; mantiene el catálogo exacto. La validación productiva por episodio, catálogos arbitrarios y streaming siguen pendientes.

El ejemplo `examples/relay_harness.py` usa una respuesta simulada explícita y funciona sin credenciales: `python examples/relay_harness.py reports/relay-v7-development-v1 "Resume este texto"`. Requiere que la ejecución offline haya generado el bundle local. Sirve para comprobar integración, no para medir respuestas de LLM. Antes de publicar en GitHub quedan pendientes la licencia del autor, revisar derechos de artefactos de terceros y una evaluación independiente sobre tareas representativas de harnesses. El dataset original y los checkpoints grandes no se incluyen en el paquete.

La extensión 0.8.1 permite ejecutar el [piloto local real](agent_local_competition_protocolo.md), con modelos exactos de Ollama, metadatos por llamada y controles de transferencia separados. La prueba de seis tareas no produjo una mejora competitiva: cascada y adaptador estructural 0/6, Qwen 1.5B fijo 1/6 y BERT transferido 2/6. El [informe](../reports/agent-local-competition-v2/competition_report.md) conserva fallos y cobertura; no acredita uso productivo. Los pesos Qwen/BERT y el adaptador previo no se incluyen en el wheel de código.
