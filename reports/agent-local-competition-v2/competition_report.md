Relay · competencia local entre políticas de agentes. Desarrollo exploratorio.

La primaria predeclarada fue la cascada barato→fuerte tras error. Completó 0/6 tareas (0.00 %) con 40 intentos de generación registrados y 6 reintentos. Las cifras siguientes conservan todos los controles; ningún resultado cambia la primaria ni acredita adopción o ROI.

La comparación está afectada por 21 fallos LiveBackendError repartidos en 6 episodios. Esa clase agrupa problemas de transporte, HTTP, respuesta del proveedor y contrato/identidad; el journal no identifica la causa concreta. No es evidencia de incapacidad del modelo. Los fallos y su consumo desconocido se mantienen en tasas, tiempo y cobertura, pero impiden aislar la calidad de routing o de los LLMs a partir de esta ejecución. Las acciones JSON inválidas se registran por separado.

Esta ejecución repite las mismas tareas tras una interrupción técnica de reports/agent-local-competition-v1: PermissionError al reemplazar el journal; no se identificó la causa concreta del bloqueo. Se verificó que tareas, modelos, parámetros de generación, políticas y cortes permanecieron iguales. Las fuentes que cambiaron fueron routerbench/agent_evaluation.py. La cohorte es desarrollo reutilizado tras esa interrupción, no una nueva observación independiente.

La ejecución previa conserva 16 llamadas completadas en el journal canónico, 3 episodios cerrados y 1 intento pendiente. Sus llamadas completadas registran 9,635 tokens de entrada, 258 de salida y 45.003 s de generación. Son recursos adicionales separados de las comparaciones siguientes; no representan todo el consumo previo porque el intento pendiente y otros componentes siguen desconocidos. El archivo temporal se preservó y verificó, sin incorporarlo como resultado confirmado ni reanudar esa ejecución.

La ejecución contiene 6 tareas y 30 episodios tarea–política con los modelos locales qwen2.5:1.5b y qwen2.5:0.5b. Fuerte y barato nombran los roles y no presuponen un orden de aciertos. Las familias son aggregate_records, file_lookup, repair_function. Son instancias de familias ya conocidas, no una prueba de generalización a familias nuevas. Con 6 tareas, un solo resultado cambia 16.67 puntos porcentuales de éxito. La muestra es demasiado pequeña para una conclusión competitiva general.

La cascada es una regla sobre errores observados del harness, no un router aprendido nuevo. El adaptador estructural conserva entrenamiento local previo y se transfiere ahora a entradas con estado e historial de episodios. BERT conserva el checkpoint publicado y proyecta su decisión fuerte/barato al catálogo Qwen: no fue calibrado con preferencias de ese par. Su resultado aquí no representa el rendimiento oficial con GPT-4/Mixtral ni evalúa MF o SW. Los regímenes de supervisión y las representaciones difieren; comparar resultados de episodios no los vuelve equivalentes.

El control estructural conserva 48 casos de ajuste, 16 de calibración y corte 0.1 sobre utilidad con segundos previstos; ese corte no es una probabilidad ni un precio. BERT conserva routellm/bert_gpt4_augmented, revisión 86237e3df400762178ea98379477b8296e66d5e4, corte 0.5 y máximo 512 tokens de entrada. Los scores internos tienen objetivos distintos y no se interpretan como probabilidades comparables de éxito del episodio.

Los parámetros, identidades de modelos y procedencia exacta de cada política están preservados en el manifiesto y en comparisons.json. No se reentrenó ni se eligieron nuevos umbrales para redactar este informe. El verificador final cierra el episodio tanto al acertar como al fallar: su resultado privado no alimenta una nueva decisión de escalado.

| Política | Éxitos / tareas | Éxito | Intentos registrados | Reintentos | Respuestas con metadatos Ollama |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cascada · primaria | 0/6 | 0.00 % | 40 | 6 | 38 |
| Siempre fuerte | 1/6 | 16.67 % | 34 | 4 | 34 |
| Siempre barato | 0/6 | 0.00 % | 42 | 36 | 35 |
| Adaptador estructural local · transferencia | 0/6 | 0.00 % | 32 | 26 | 20 |
| RouteLLM BERT publicado · transferencia a Qwen | 2/6 | 33.33 % | 34 | 5 | 34 |

| Política | Llamadas con LiveBackendError | Episodios afectados por backend/contrato |
| --- | ---: | ---: |
| Cascada · primaria | 2 | 2 |
| Siempre fuerte | 0 | 0 |
| Siempre barato | 7 | 2 |
| Adaptador estructural local · transferencia | 12 | 2 |
| RouteLLM BERT publicado · transferencia a Qwen | 0 | 0 |

BERT eligió el fuerte en todas sus decisiones. Su selección de candidatos coincide con siempre fuerte; cualquier diferencia observada entre sus aciertos no puede atribuirse a escoger modelos distintos. Permanecen variaciones de respuestas, backend, orden y tiempo de ejecución que esta muestra no separa causalmente.


Se identificaron 48 grupos entre políticas con el mismo modelo solicitado y request_sha256 exacto. En 6 grupos aparecieron textos de respuesta distintos y en 3 estados de error distintos. BERT y siempre fuerte compartieron 29 solicitudes exactas; en 3 difirieron respuesta o error. Todos los grupos, no solo los discrepantes, quedan en comparisons.json. Esto registra variabilidad observada bajo parámetros congelados; no identifica su causa ni permite atribuirla al routing.


Un intento del harness puede fallar antes de obtener respuesta, incluso antes de invocar el modelo por un límite local. Las respuestas con metadatos no incluyen intentos interrumpidos o fallidos sin respuesta utilizable. Los reintentos ya están dentro del total y no se suman dos veces. No se eliminan tareas fallidas para mejorar tasas o recursos medios.

BERT truncó su entrada en 34/34 decisiones. Retuvo 17,408 de 53,363 tokens acumulados por decisión (32.62 %); la cobertura individual fue de 19.49 % a 78.65 %. Las entradas completas tenían entre 651 y 2,627 tokens. La tarea precede al historial, pero esta cobertura no demuestra retención de las observaciones relevantes: BERT puede perder historial que la cascada sí utiliza. Son tokens del router, no del generador.

| Política | Respuestas que alcanzaron el límite de salida | Motivos de terminación del proveedor | Errores observados del harness |
| --- | ---: | --- | --- |
| Cascada · primaria | 0 | stop: 38 | InvalidAction:ValueError: 3; LiveBackendError: 2; ToolError:InvalidArguments: 1 |
| Siempre fuerte | 0 | stop: 34 | ToolError:InvalidArguments: 6 |
| Siempre barato | 6 | length: 6; stop: 29 | InvalidAction:ValueError: 32; LiveBackendError: 7; ToolError:InvalidArguments: 2 |
| Adaptador estructural local · transferencia | 0 | stop: 20 | InvalidAction:ValueError: 17; LiveBackendError: 12; ToolError:InvalidArguments: 1 |
| RouteLLM BERT publicado · transferencia a Qwen | 0 | stop: 34 | ToolError:InvalidArguments: 6 |

El límite de salida se contabiliza solo cuando Ollama informa finish_reason=length. No se confunde ese truncamiento de una respuesta con la truncación de entrada de BERT. Una salida limitada o inválida permanece en los recursos y resultados; no se genera otra respuesta para corregir este informe.

| Política | Intentos con fuerte | Intentos con barato | Tokens de entrada totales | Tokens de salida totales | Cobertura entrada | Cobertura salida |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cascada · primaria | 34 | 6 | desconocido | desconocido | 38/40 | 38/40 |
| Siempre fuerte | 34 | 0 | 32,392 | 1,243 | 34/34 | 34/34 |
| Siempre barato | 0 | 42 | desconocido | desconocido | 35/42 | 35/42 |
| Adaptador estructural local · transferencia | 0 | 32 | desconocido | desconocido | 20/32 | 20/32 |
| RouteLLM BERT publicado · transferencia a Qwen | 34 | 0 | 32,200 | 1,131 | 34/34 | 34/34 |

Los tokens se acumulan por llamada, incluido el contexto repetido en reintentos. Son contadores de recursos, no unidades monetarias. Un contador faltante deja desconocido el total; los subtotales y la cobertura permanecen en summary.json. Tokens cacheados desconocidos no se convierten en cero ni se infieren del total de entrada.

| Política | Routing p50 (ms) | Routing p95 (ms) | Routing medido | Episodio p50 (s) | Episodio p95 (s) | Episodios medidos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cascada · primaria | 0.199 | 2.119 | 40/40 | 31.897 | 47.640 | 6/6 |
| Siempre fuerte | 0.222 | 0.510 | 34/34 | 10.136 | 50.655 | 6/6 |
| Siempre barato | 0.184 | 0.345 | 42/42 | 23.042 | 59.586 | 6/6 |
| Adaptador estructural local · transferencia | 20.276 | 27.554 | 32/32 | 11.203 | 23.785 | 6/6 |
| RouteLLM BERT publicado · transferencia a Qwen | 638.158 | 819.468 | 34/34 | 40.197 | 55.393 | 6/6 |

El tiempo por episodio es el observado durante ejecución del entorno y sus llamadas; los cuantiles excluyen mediciones desconocidas y muestran su cobertura. Se comprobó la rotación contra la secuencia congelada: always_strong, always_weak, cheap_then_escalate, local_structural_transfer, routellm_bert_transfer. Rotar políticas reduce parte del efecto del orden, pero no elimina diferencias por carga inicial, caché, calentamiento o competencia de recursos. Con 6 tareas y 5 políticas, los resultados no equivalen a un ensayo de latencia aleatorizado. No se interpreta menor duración como menor costo USD ni como un SLA.

| Política | Generación acumulada (s) | Carga del proveedor acumulada (s) | Costo LLM (USD) | Costo total (USD) | Episodios con costo completo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cascada · primaria | 186.211 | desconocido | desconocido | desconocido | 0/6 |
| Siempre fuerte | 110.850 | 0.226 | desconocido | desconocido | 0/6 |
| Siempre barato | 158.137 | desconocido | desconocido | desconocido | 0/6 |
| Adaptador estructural local · transferencia | 72.335 | desconocido | desconocido | desconocido | 0/6 |
| RouteLLM BERT publicado · transferencia a Qwen | 175.416 | 0.225 | desconocido | desconocido | 0/6 |

La carga informada por el proveedor es un componente diagnóstico de sus llamadas, no tiempo adicional que se sume otra vez al episodio. Los USD permanecen desconocidos cuando no hay gasto medido ni tarifa declarada; ejecutar localmente no convierte el costo en cero. Faltan, entre otros, energía y valoración del hardware/cómputo. No se convierten segundos o tokens en dinero mediante precios inventados.

| Cascada primaria frente a | Diferencia de éxito | IC 95 % por grupos | Reducción del costo total | Estado ROI/adopción |
| --- | ---: | ---: | ---: | --- |
| Siempre fuerte | -16.67 pp | [-50.00, +0.00] pp | desconocido | indeterminate |
| Siempre barato | +0.00 pp | [+0.00, +0.00] pp | desconocido | indeterminate |
| RouteLLM BERT publicado · transferencia a Qwen | -33.33 pp | [-66.67, +0.00] pp | desconocido | indeterminate |
| Adaptador estructural local · transferencia | +0.00 pp | [+0.00, +0.00] pp | desconocido | indeterminate |

| Cascada primaria menos referencia | Diferencia de intentos | Diferencia de tokens entrada | Diferencia de tokens salida | Diferencia de tiempo medio por episodio (s) |
| --- | ---: | ---: | ---: | ---: |
| Siempre fuerte | +6 | desconocido | desconocido | +12.740 |
| Siempre barato | -2 | desconocido | desconocido | +4.891 |
| RouteLLM BERT publicado · transferencia a Qwen | +6 | desconocido | desconocido | -1.907 |
| Adaptador estructural local · transferencia | +8 | desconocido | desconocido | +19.330 |

Las diferencias son primaria menos referencia: una diferencia temporal negativa indica menos segundos observados. Los intervalos de éxito usan 2000 réplicas, semilla 20261112, emparejadas por episodio y remuestreadas por source_group. Son diagnósticos descriptivos con pocos grupos, plantillas conocidas y desarrollo; no permiten confirmar superioridad, no inferioridad o transferencia general. Las diferencias de tokens y tiempo son descriptivas y no se convierten a ahorro económico.

| Familia conocida | Tareas por política | Cascada | Siempre fuerte | Siempre barato | BERT transferido | Estructural transferido |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate_records | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| file_lookup | 2 | 0/2 | 1/2 | 0/2 | 2/2 | 0/2 |
| repair_function | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

| Política | Finales incorrectos | Presupuesto agotado | Acciones inválidas | Errores de herramienta | Pruebas públicas fallidas | Repeticiones exactas de herramientas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cascada · primaria | 2 | 4 | 3 | 1 | 0 | 24 |
| Siempre fuerte | 1 | 4 | 0 | 6 | 0 | 20 |
| Siempre barato | 1 | 5 | 32 | 2 | 0 | 0 |
| Adaptador estructural local · transferencia | 2 | 4 | 17 | 1 | 0 | 0 |
| RouteLLM BERT publicado · transferencia a Qwen | 1 | 3 | 0 | 6 | 0 | 19 |

Finales incorrectos y presupuesto agotado cuentan episodios; las otras columnas cuentan acciones. Una repetición exacta puede ser legítima: el conteo por sí solo no demuestra un bucle. Los errores se solapan con las acciones, no son fallos adicionales que deban sumarse. Las trayectorias ordenadas por episodio están en comparisons.json y describen hechos observados, sin volver a puntuar respuestas privadas.

Se observaron 6 episodios de agregación con lecturas repetidas, ninguna acción write_json y presupuesto agotado; 10 episodios de reparación sin inspect_function; y 7 llamadas donde se pasó un objeto schema como arguments. Son patrones registrados, no una causa inferida de los errores de backend. La hipótesis siguiente es que un contrato de acciones más explícito y un estado compacto de progreso observable podrían reducir errores de formato y repetición. Habría que probarlo como intervención nueva, común a todos los controles, después de instrumentar los fallos de backend y comprobar la variabilidad; no se cambiaron estos resultados ni cortes.

Las filas por familia muestran los conteos completos y no se usan para seleccionar otra primaria. La implementación de reparación cambia campos de una plantilla aritmética restringida; no representa programación libre de agentes. La siguiente evidencia útil exige una cohorte mayor y nuevas familias, catálogo y presupuesto predeclarados, y costos completos antes de evaluar ROI.

Evidencia: [manifiesto y políticas](manifest.json), [journal](journal.json), [resumen original](summary.json), [comparaciones extendidas](comparisons.json), [snapshot de fuentes](source_bundle.zip) y [procedencia del informe](competition_report_provenance.json). Se verificó la cadena completa y se reprodujo summarize con las referencias fuerte, barato, BERT y estructural sin nuevos callbacks ni acceso a pesos del modelo.
