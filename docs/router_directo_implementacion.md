Primer hito del router directo, 8 de noviembre de 2026. El objetivo es construir un candidato propio y compararlo de forma verificable con referencias. No se declara superioridad sobre RouteLLM ni validación productiva.

Se implementó un predictor de éxito por modelo, calibración sigmoide con datos disjuntos y un estimador de costo histórico basado en longitudes disponibles antes de responder. El costo usa regresión Ridge de log-costo con corrección residual ajustada solo en fit. Sigue siendo una estimación y debe examinarse su error. Cuando una salida carece de ambas clases en ajuste o calibración, se registra una predicción constante suavizada en lugar de fingir que se aprendió una frontera.

La selección busca el menor costo previsto entre los modelos que superan la probabilidad mínima declarada. Si ninguno la supera, usa mayor probabilidad prevista con desempate por costo y nombre. El umbral absoluto de calidad que define el objetivo y la probabilidad mínima para seleccionar son conceptos distintos. La política admite una máscara de compatibilidad por modelo/contexto y se abstiene si no existe candidato permitido. El archivo histórico no contiene un registro productivo de capacidades, por lo que este hito no valida ese filtro contra proveedores reales.

La representación estructural usa las 14 variables y normalización aprendida solo en ajuste. La híbrida añade TF-IDF de palabras y caracteres, sin recortar el texto. La selección de vocabulario sí descarta patrones no representados, por lo que no equivale a comprender el documento completo. La representación contextual semántica queda pendiente de un encoder y evaluación específicos. `context.py` prepara ventanas con offsets y cobertura completa sobre IDs de tokens proporcionados por el caller; todavía no incorpora un tokenizer ni un agregador semántico.

El runner verifica los hashes de la preparación anterior y utiliza exclusivamente sus filas de train para la nueva comparación. Asigna fit/calibración/validación por hash, conserva sus IDs y genera estimaciones, decisiones, resultados observados y manifiesto. Las decisiones reciben probabilidades y costos previstos, nunca los scores y costos reales de esa fila. El test exterior no se evalúa. Los resultados de validación se identifican como exploratorios y no pueden convertirse en una evaluación confirmatoria después de escoger configuraciones mirando esa validación.

Desde PowerShell, sin instalar nuevos paquetes en el entorno actual:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m routerbench.development --max-fit-rows 6000 --output reports/competitor-development-pilot-v1
```

Ese piloto limita solo el número de filas de ajuste, sin truncar prompts. Usa todos los casos de calibración y validación internos. El mismo subconjunto de ajuste se usa para ambas variantes. Si ya existe esa salida, elegir otro nombre para conservar trazabilidad. Omitir `--max-fit-rows` entrena con todo el fit interior; no ejecutar cambios de configuración sin registrar nueva versión.

Una vez completo el entrenamiento, el manifiesto y los artefactos permiten una recomendación para un prompt nuevo:

```powershell
python -m routerbench.recommend --run-dir reports/competitor-development-pilot-v1 --variant lexical_hybrid --prompt "Compare the assumptions in these two documents."
```

Este comando devuelve el nombre elegido, éxito estimado, costo histórico previsto y si se activó respaldo. No llama al LLM, no genera una respuesta y no utiliza precios actuales. Los artefactos joblib deben proceder de una ejecución local de confianza; el hash detecta modificaciones pero no autentica a un tercero. La demo conectada a modelos actuales requiere seleccionar catálogo, evaluar transferencia y configurar acceso y presupuesto.

Las diferencias pareadas se calculan sin reentrenar con `python -m routerbench.comparison --run-dir reports/competitor-development-pilot-v1 --replicates 2000`. El resumen de la primera ejecución está en `reports/competitor_piloto.md`. Los resultados corresponden únicamente al desarrollo y no se ha medido RouteLLM.

El runner reporta cada modelo fijo como referencia, además del mejor fijo y el más barato elegidos solo en fit. Guarda la curva costo-calidad para umbrales preespecificados y diagnósticos Brier/log-loss por modelo junto con intervalos de probabilidad agrupados. Un Brier menor no acredita por sí solo mejor calibración: combina varios componentes. La latencia piloto incluye representación desde el texto, predicciones y selección con batch=1, en proceso caliente y sin caché de embeddings. El overhead no se suma a dólares históricos sin un modelo económico explícito.

La referencia oficial MF de RouteLLM queda pendiente porque el código revisado calcula embeddings mediante API. No se ejecutó, no se sustituyó por una imitación y no se adjudicaron resultados ficticios. La comparación futura restringirá ambos routers al mismo par, separando ese resultado del catálogo de once modelos. Los siguientes hitos son referencia oficial o adaptación identificada, representación semántica con cobertura de contexto largo, validación por familias, criterio confirmatorio preespecificado y demostración con un catálogo contemporáneo.
