Fase semántica de Relay, 8 de noviembre de 2026. El usuario eligió el nombre Relay y autorizó continuar la comparación. Se conserva el paquete técnico `routerbench` para no romper los artefactos existentes. Esta fase añade variantes semántica e híbrida semántica; las anteriores estructural y léxica permanecen como referencias. Los resultados siguen siendo de desarrollo, con el test exterior reservado y RouteLLM oficial pendiente.

El encoder local congelado es `intfloat/multilingual-e5-small`, revisión `6a0d452a575215f80b8f66276dd4ee5d504942c6`, exportación oficial ONNX qint8. Se verificó el SHA-256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88` antes de cargar sus 118,346,824 bytes. Fuentes: [ficha del autor](https://huggingface.co/intfloat/multilingual-e5-small) y [revisión fijada](https://huggingface.co/intfloat/multilingual-e5-small/commit/6a0d452a575215f80b8f66276dd4ee5d504942c6). La exportación cuantizada se identifica expresamente como tal: no se presume equivalencia con fp32 ni superioridad de rendimiento en cualquier CPU.

Se instaló el backend opcional ONNX Runtime 1.31.1 y tokenizers 0.23.3 dentro de `.venv-relay`, reutilizando los paquetes del entorno científico base sin modificarlos. El setup descarga únicamente archivos del modelo público fijado. El cálculo de embeddings posterior se realiza localmente, sin APIs pagadas ni envío de prompts. Las dependencias adicionales observadas están en `requirements-semantic-lock.txt`. Los archivos originales del modelo y sus hashes están en un manifiesto dentro de `artifacts/encoders/`, excluido de Git.

El tokenizer procesa `query: ` y el texto conjuntamente, sin truncación ni padding global. Después se separan los IDs del prefijo estable. Así se conserva el comportamiento del prefijo incluso si el texto contiene tokens especiales literales. Se reservan tres IDs de prefijo y dos especiales: cada ventana contiene hasta 507 tokens de cuerpo dentro del máximo 512 del encoder. Se mantiene solapamiento de 64 tokens. La última ventana cubre la cola completa y la cobertura se verifica en IDs, sin decodificar y volver a tokenizar fragmentos. La forma real del ONNX se inspeccionó: input_ids, attention_mask y token_type_ids int64; salida last_hidden_state. Se resuelven CLS/SEP/PAD con el tokenizer local, no con supuestos de arquitectura.

Cada ventana produce media de estados ocultos usando attention_mask y normalización L2. La representación del prompt es la media ponderada de esos vectores por tokens nuevos cubiertos, seguida de L2. El promedio es una heurística que puede diluir una restricción breve o perder relaciones entre ventanas; cubrir todos los tokens no acredita comprensión del documento. La variante semántica tiene 384 dimensiones. La híbrida concatena esas dimensiones con las 14 variables escaladas solo en fit, normalizando los bloques. La agregación no recibe scores, costos, identificadores ni nombres de benchmark.

Una comprobación de integración detectó que la exportación cuantizada producía vectores ligeramente distintos cuando un prompt compartía lote con otros: en tres ejemplos los cosenos entre modos estuvieron aproximadamente entre 0.9955 y 0.9970. Con una ventana por llamada se obtuvo identidad exacta en esos ejemplos. Por ello este primer experimento fija batch_size=1. No se interpreta aquella comprobación pequeña como garantía universal de determinismo. La política, ensamblado de tokens, batch, hilos, revisión, hashes de modelo/tokenizer y versiones forman parte de la identidad de caché. Las pruebas de aislamiento son importantes porque [la cuantización de ORT](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) depende de decisiones numéricas y del hardware.

La caché SQLite se indexa por SHA-256 del texto exacto y namespace del encoder. No utiliza la normalización de duplicados: espacios o mayúsculas distintos pueden cambiar un embedding. Guarda vectores float32 y estadísticas con controles de integridad, sin una columna con el texto del prompt. No conserva sesiones ONNX dentro de joblib. Tras cargar el router se abre una sesión de forma perezosa y se comprueban los archivos locales; no hay descargas automáticas al recomendar.

Para comparar exactamente con el piloto anterior se verifica el hash de `development_rows.parquet` y se reutilizan las mismas 6,000 filas de ajuste, 4,383 de calibración y 4,406 de validación. También se comprueban hashes de datos preparados y huellas de prompt. No se recompone la partición mirando resultados. Los costos previstos, targets y clasificador son los mismos del router directo; solo cambia la representación. El encoder no se ajusta con los datos de RouterBench, pero su preentrenamiento externo puede contener fuentes relacionadas con benchmarks, una limitación no resuelta por esta partición.

Entrenar y evaluar desarrollo con el encoder ya preparado:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv-relay\Scripts\python.exe -m routerbench.development --config configs/semantic_competitor.json --output reports/relay-semantic-development-v1
```

El runner guarda las estadísticas por prompt de validación, el número de ventanas, tokens cubiertos y tokens truncados, y el manifiesto del encoder. La inferencia por lotes sobre vectores cacheados se etiqueta como tal. Para latencia de una solicitud se desactiva la caché y se incluyen tokenización, todas las ventanas, agregación, atributos, predicciones y selección. El arranque de una sesión y la carga del modelo se consideran trabajo adicional y no se confunden con el proceso caliente del piloto.

Comparar los resultados pareados con las variantes anteriores:

```powershell
.\.venv-relay\Scripts\python.exe -m routerbench.comparison --run-dir reports/relay-semantic-development-v1 --reference-run reports/competitor-development-pilot-v1 --replicates 2000
```

La comparación confirma igualdad de datos, objetivo y orden de prompts, y calcula deltas de éxito y costo histórico sobre los mismos casos. El desarrollo ya se ha usado para explorar y no sirve para declarar a posteriori una victoria confirmatoria. Los intervalos remuestrean prompts y no sustituyen evaluación por familia. El objetivo de calidad y la probabilidad mínima de selección siguen siendo parámetros distintos. El margen confirmatorio y el catálogo actual de la demo aún requieren definición antes de su evaluación correspondiente.

Para recomendar un modelo de este catálogo histórico a partir de un prompt nuevo:

```powershell
.\.venv-relay\Scripts\python.exe -m routerbench.recommend --run-dir reports/relay-semantic-development-v1 --variant semantic_hybrid --prompt-file ruta/al/prompt.txt
```

La recomendación desactiva la caché de embeddings de entrenamiento y procesa el texto completo. Todavía no ejecuta el LLM seleccionado ni genera su respuesta. La capacidad de manejar bloques largos en el router no demuestra calidad de conversaciones o documentos en producción, y no amplía el contexto admitido por cada LLM final. Esa evaluación sigue `produccion_contexto_largo.md`.

La fase inicial ya se ejecutó. `reports/relay_semantico.md` resume los resultados y su incertidumbre. `tools/verify_semantic_run.py` verifica hashes, igualdad exacta de particiones, cobertura en validación, serialización y una integración nativa con texto sintético superior a 8,192 tokens, sin ejecutar LLMs ni consultar el test final.
