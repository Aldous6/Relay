Relay · evaluación de adaptación local, cohorte nueva de 64 casos. La política primaria quedó congelada antes de generar esta cohorte. No superó a usar siempre Qwen 1.5B: acertó 38/64 frente a 46/64. BERT adaptado acertó 39/64. No se modifica la demo ni se elige retrospectivamente otra arquitectura.

El desarrollo utilizó los 64 casos sintéticos ya observados en la auditoría local, con 48 filas de ajuste y 16 de calibración por hash independiente de resultados. Relay usa los 14 atributos estructurales; el control usa 768 atributos CLS del BERT congelado. Comparten filas, hiperparámetros del árbol y grilla de utilidad, y cada política eligió su corte con calibración: 0.1 y 0.2. Calibrar y escoger el corte en esas mismas 16 filas produce diagnósticos de desarrollo optimistas; no se confunden con esta cohorte nueva.

Se fijaron artefactos, digests de Qwen, instrucciones, parámetros y snapshot del código antes de muestrear la nueva semilla. Se verificó ausencia de coincidencias exactas normalizadas con los casos de desarrollo y RouterBench. Los 64 casos son otras instancias de las mismas ocho plantillas; la prueba no evalúa familias desconocidas, conversaciones reales ni documentos extensos. No acredita no inferioridad o superioridad general.

| Política | Exactas | Qwen 1.5B / 0.5B | Tokens entrada / salida | Segundos de generación seleccionados |
| --- | ---: | ---: | ---: | ---: |
| Relay · 14 atributos adaptados | 38/64 (59.38 %) | 9 / 55 | 15680 / 270 | 95.838 |
| BERT adaptado local | 39/64 (60.94 %) | 7 / 57 | 15680 / 273 | 95.098 |
| RouteLLM BERT publicado · transferencia | 46/64 (71.88 %) | 64 / 0 | 15680 / 277 | 137.105 |
| Siempre Qwen 1.5B | 46/64 (71.88 %) | 64 / 0 | 15680 / 277 | 137.105 |
| Siempre Qwen 0.5B | 33/64 (51.56 %) | 0 / 64 | 15680 / 265 | 91.964 |

Las 128 respuestas únicas se comparten entre las cinco políticas; no son cinco ejecuciones de servicio. No hubo fallos de generación. Relay redujo 30.10 % la suma de duraciones de generación seleccionadas frente a Qwen 1.5B, pero perdió ocho aciertos, equivalentes a 12.50 puntos porcentuales. Esa pérdida impide presentar el ahorro como calidad conservada. Los tokens totales fueron 15,950 frente a 15,957: esta prueba no respalda un ahorro sustancial de tokens.

RouteLLM BERT publicado conserva su checkpoint de preferencias y umbral 0.5, proyectados al nuevo par. Eligió Qwen 1.5B en todos los casos. Es distinto del control BERT adaptado localmente: este último usa el backbone congelado y un nuevo árbol entrenado con las respuestas locales. Las preferencias publicadas se calcularon en el mismo forward que extrae CLS, sin entrar en los atributos del nuevo árbol. Todas las rutas, incluidas las shadow, quedaron persistidas antes de la primera generación.

| Plantilla | Relay | BERT adaptado | Qwen 1.5B fijo | Qwen 0.5B fijo |
| --- | ---: | ---: | ---: | ---: |
| Consulta de un registro | 5/8 | 5/8 | 8/8 | 5/8 |
| Suma de lotes | 0/8 | 0/8 | 0/8 | 0/8 |
| Filtro y suma | 1/8 | 1/8 | 0/8 | 1/8 |
| Actualización hasta una fecha | 2/8 | 2/8 | 6/8 | 2/8 |
| Relación de dos pasos | 7/8 | 7/8 | 8/8 | 7/8 |
| Conflicto de fuentes | 8/8 | 8/8 | 8/8 | 8/8 |
| Dato ausente | 7/8 | 8/8 | 8/8 | 7/8 |
| Corrección en historial | 8/8 | 8/8 | 8/8 | 3/8 |

Complementariedad observada: ambos modelos acertaron 32 casos, solo Qwen 1.5B acertó 14, solo Qwen 0.5B acertó 1 y ambos fallaron 17. El oráculo que conociera las respuestas habría acertado 47/64; es un límite retrospectivo, no una política ejecutable. Los modelos fallaron los ocho ejercicios de suma de lotes. Un router que solo escoge entre estas respuestas no puede corregir esos fallos compartidos.

La exactitud exige igualdad normalizada de la respuesta completa. Puede penalizar explicaciones aunque incluyan el dato correcto, y no es una evaluación semántica amplia. La colección fija temperatura 0, seed 42, contexto 8192 y salida máxima 96; conserva el texto completo del prompt y recoge una sola respuesta por caso/modelo. No estima variabilidad entre generaciones ni usa APIs pagadas.

La medición de routing usa CPU residente, cuatro hilos, calentamiento con dos textos neutrales y ejecución secuencial previa a generación. Los milisegundos no incluyen carga inicial ni calentamiento. El tiempo nativo del BERT publicado se comparte con la extracción del control, por lo que no es una ejecución independiente alternada. La suma de generación seleccionada excluye router y conserva efectos de caché, carga y orden. No representa dinero, ahorro energético, latencia completa de producción ni SLA. El costo monetario permanece desconocido.

La siguiente mejora requiere evidencia local representativa y una selección más conservadora: identificar dónde el modelo pequeño conserva calidad, reservar nuevas familias y medir errores de enrutado. Los casos nuevos de este informe quedan consumidos para evaluar esta política; si se usan para adaptar, pasan a desarrollo. Para tareas que ambos modelos fallan, habrá que ampliar capacidades o candidatos y evaluar de nuevo antes de atribuir una mejora al router. Esta es la interpretación de estos resultados, no un nuevo experimento ejecutado.

Evidencia: [comparación verificable](comparison.json), [resumen original](evaluation/summary.json), [cohorte](cohort.jsonl), [políticas](policy_lock.json), [rutas publicadas](published_bert_routes.json), [barrera previa a generación](all_routes_before_generation.json), [respuestas](evaluation/responses.json) y [snapshot ejecutado](source_bundle.zip). Parámetros y contadores: [chat de Ollama](https://docs.ollama.com/api/chat) y [parámetros de Ollama](https://docs.ollama.com/modelfile).
