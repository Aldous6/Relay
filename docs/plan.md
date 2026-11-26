Dirección vigente al 17 de noviembre de 2026: concentrar Relay en optimizar harnesses existentes. El estudio científico y los pilotos previos quedan conservados. La entrega activa es el [SDK ligero para harnesses](relay_harness.md), con integración OpenHands local, selección estable por tarea, observación de costos y aceptación, propuesta por actividad y validación sobre grupos separados antes de aplicar. El paquete local es `relay-harness 0.1.0.dev1`; no incluye datasets ni depende del paquete científico.

La siguiente prueba de valor debe ocurrir en un harness de uso real: mantener su verificador, contabilizar todos los intentos y comparar tareas equivalentes con modelos capaces. No se afirma ahorro ni superioridad sobre RouteLLM o Not Diamond a partir de las comprobaciones de software. No se reabre el test histórico ni se usa el piloto de ocho casos como validación nueva. El agente propio de mantenimiento deja de ser la prioridad.

Los planes y resultados que siguen son antecedentes de las etapas anteriores y no sustituyen esta dirección vigente.

Dirección vigente al 17 de noviembre de 2026: demostrar una ventaja medida frente a un competidor. La [comparación de ocho casos ya ejecutada](../outputs/maintenance-comparison/20261117-live-v3-complete/delivery_report.md) no respalda promover la cascada barata: empata con BERT a umbral fijo y cuesta 65.14 % más que BERT calibrado/120B primero, con un éxito menos. La aceptación reservada detectó defectos que los tests visibles no mostraban. La corrección Windows LF/CRLF ya elimina cinco rechazos de aplicación, con [diagnóstico separado](../reports/maintenance-line-ending-diagnostic-v1/report.md), sin convertirlo en una nueva comparación ganada.

La decisión de producto es mantener edición directa, usar 120B primero como referencia operativa y conservar la cascada como modo explícito. El próximo trabajo debe resolver los fallos semánticos de edición y validación detectados —registros entrecomillados, ventanas, copia de configuración y precisión decimal— antes de añadir otro predictor o ampliar indiscriminadamente llamadas. Pasar tests públicos no autoriza afirmar tarea correcta. Cuando el ejecutor tenga calidad suficiente en problemas nuevos, la comparación del router deberá usar una política congelada y una cohorte representativa no reutilizada para seleccionarla. Estos ocho casos siguen siendo desarrollo diagnóstico; no se convierten en fit ni holdout. La meta competitiva permanece pendiente; BFCL y los resultados históricos no se reabren.

El plan del 12 de noviembre se conserva a continuación como antecedente. La entrega de producto solicitada es una librería descargable desde GitHub para integrarse en harnesses de agentes y reducir recursos manteniendo calidad. La licencia del autor queda pendiente. Streamlit conserva su papel de demo previa y deja de ser el eje del desarrollo.

El [piloto local de episodios 0.8.1](../reports/agent-local-competition-v2/competition_report.md) está terminado y no muestra mejora: cascada y adaptador estructural 0/6, Qwen 1.5B fijo 1/6 y BERT transferido 2/6. La prioridad siguiente es distinguir errores de transporte/contrato, acciones inválidas y errores semánticos; verificar capacidad del modelo de respaldo y estabilidad de solicitudes idénticas; y solo después recoger una cohorte de desarrollo mayor con ajuste y calibración separados. Llama 3 de 8B aparece instalado y puede ser un candidato a comprobar, pero su superioridad en estas tareas no se ha medido. No se debe remapear el artefacto Qwen existente a otro par. La cascada barata seguida de escalado no se promueve con estos resultados.

MF/SW siguen pendientes: no hay sus pesos y conjuntos auxiliares locales. La revisión oficial fijada requiere embeddings text-embedding-3-small, y el espacio E5 local no los sustituye. Antes de ejecutarlos deben concretarse revisiones, procedencia, entorno compatible y presupuesto/autorización para embeddings externos, o disponer de embeddings precomputados verificables. No se activó ese servicio. Fuentes: [MF](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/matrix_factorization/model.py), [SW](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/routers.py).

Como antecedente, Relay 0.8.0 incorporó el [evaluador de episodios](evaluacion_agentes.md): primero se verifica su integración con controles programados y contabilidad explícita; después se necesita fijar catálogo/versiones reales, presupuesto, tareas representativas y familias reservadas antes de recoger evidencia. El fixture configura 12 tareas y tres controles, con tokens y tarifas ficticios; no entrena v8 ni prueba ROI. Los siguientes párrafos conservan el diagnóstico de los routers históricos y los pendientes científicos y de publicación.

La auditoría, clasificación original, evaluación histórica final y pilotos v3–v5 ya están ejecutados. Sus artefactos y conclusiones se conservan. La comparación histórica final no acreditó no inferioridad dentro del margen fijado y la adaptación Qwen perdió calidad. No se vuelve a seleccionar sobre el test consumido.

Las iteraciones v6 y v7 están ejecutadas y exportan bundles para relay_router. V7 probó predicción global con vecinos de ajuste y un control sin vecinos, siguiendo relay_v7_protocolo.md. La primaria conserva 78.82 % de éxito en desarrollo reutilizado, pero cuesta 4.46 % más que BERT101; no alcanzó el objetivo competitivo. El control sin vecinos y el ensamble v4 mantuvieron mejores medias que la primaria. El resumen en reports/relay-v7-development-v1/summary.json conserva los resultados sin sustituir a posteriori el candidato predeclarado. No procede presentar más complejidad o un número de versión mayor como mejora.

La siguiente hipótesis debe explicar qué información útil falta y contrastarse con el ensamble v4 además de BERT, conservando las variantes fallidas. Una representación semántica con mayor cobertura de ajuste puede estudiarse, pero este experimento no demuestra que vaya a mejorar. Su costo de inferencia y la evidencia de discriminación entre éxitos exclusivos del fuerte y del barato deben comprobarse antes de aumentar complejidad. No se cambia el umbral de éxito para obtener un porcentaje atractivo ni se adjudica al router un éxito que los candidatos no pueden producir.

El siguiente hito de evidencia es una cohorte nueva y representativa de tareas de harnesses, con familias, documentos y conversaciones separados de desarrollo. Fijar catálogo/versiones, métricas de tarea completa, regla de éxito y presupuesto antes de medir; adaptar con ajuste y calibración separados, congelar políticas y comparar contra cada modelo fijo y referencias comparables. Incluir tiempo del router, generación y reintentos. La nueva preparación por familias ya existe; faltan los casos y una rúbrica validada para resultados abiertos. Los casos locales ya observados permanecen como desarrollo si se vuelven a usar.

Antes de la publicación final: decidir licencia de código propio, revisar procedencia y derechos de pesos/datos, preparar instrucciones reproducibles y una instalación mínima, cerrar pruebas de integración y documentar resultados positivos o negativos. Esta carpeta todavía no tiene historial Git. No confundir construir un wheel local con publicar GitHub o acreditar utilidad productiva. La presentación académica debe incorporar únicamente resultados ejecutados y distinguir v1 de las ampliaciones.

El calendario siguiente se conserva como plan inicial histórico, no como inventario actual de pendientes. La fecha del 26 de noviembre proviene de la propuesta y no se ha reconfirmado en esta revisión.

# Plan inicial de trabajo (histórico)

Entrega de referencia: 26 de noviembre de 2026, según la propuesta adjunta. El calendario puede revisarse con el autor.

## 7–12 de noviembre: datos y contrato

Auditoría reproducible del original, trazabilidad, prompt reconstruido, deduplicación, 14 atributos y particiones persistidas. Comparar las cifras de la propuesta con la auditoría nueva y explicar diferencias. Confirmar con el autor cualquier cambio en el protocolo antes de presentar resultados como definitivos. Verificar la disponibilidad de las skills al iniciar el siguiente turno.

Entregables: `reports/audit.json`, `data/processed/manifest.json`, documentación de atributos y pruebas. Base implementada. La selección de las skills está razonada en `docs/skills.md`.

## 13–19 de noviembre: comparación en entrenamiento

Implementar un comando de CV de 5 folds que produzca predicciones fuera de muestra y resultados por fold para las cuatro técnicas. Registrar semilla, versiones y parámetros. Reportar F1-macro y métricas de Tier 2. Elegir un modelo y congelar configuración. Estimar importancia por permutación sin usar desempeño del test para seleccionar señales. Revisar si una búsqueda de hiperparámetros pequeña aporta valor.

Entregables pendientes: `src/routerbench/evaluation.py`, `src/routerbench/policy.py`, `reports/cv_results.csv` y figuras. No crear resultados ficticios ni rellenar tablas con valores de las notas.

## 20–26 de noviembre: evaluación y redacción

Fijar representantes con train y evaluar una vez el test. Separar métricas de clasificación y política. Evaluar política sobre el test completo. Sensibilidad de umbrales, evaluación por grupos e incertidumbre pareada. Redactar resultados positivos o negativos, limitaciones y bibliografía comprobable. Actualizar la presentación con cifras provenientes de artefactos ejecutados.

Entregables pendientes: reporte final, figuras exportables, tabla costo-calidad, notebook de resultados y presentación revisada. La investigación bibliográfica inicial cubre la fuente de datos y metodología base; ampliar el estado del arte de routing antes de la redacción final.

Ampliación solicitada el 8 de noviembre: incluir una demostración en vivo con selección de LLM y respuesta observada. Su alcance y relación con la propuesta están en `alcance_y_demo.md`. Primero cerrar la evaluación científica de referencia. La demo requiere elegir catálogo, acceso y presupuesto, y no acredita por sí sola transferencia a modelos distintos de RouterBench. El orden y calendario de esta ampliación quedan sujetos a la disponibilidad de esos elementos.
