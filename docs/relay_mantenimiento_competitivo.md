# Relay: ventaja verificable en mantenimiento Python

Dirección corregida por el usuario el 15 de noviembre de 2026: la entrega debe demostrar ahorro o una ventaja frente a una referencia ejecutada. Que el agente prepare un parche correcto es necesario, pero no satisface ese objetivo. Hoy no hay una ventaja demostrada ni un predictor entrenado para GPT-OSS en mantenimiento. Los resultados históricos siguen sin modificarse.

La hipótesis es escoger la primera edición que minimice el costo hasta terminar correctamente: empezar con 20B cuando basta y con 120B cuando evita un intento barato fallido. El perfil directo incorporado el 17 de noviembre omite la inspección LLM y usa el alcance autorizado del config; permite como máximo una reparación con 120B para ambos candidatos. El perfil anterior de inspección común con 20B sigue disponible. Se compara exactamente la primera edición y su continuación fijada. No se promete selección óptima de todas las actividades ni routing general.

La implementación recoge las dos trayectorias reales desde el mismo mensaje inicial. En el perfil con inspección, esta se ejecuta una vez y su consumo se cuenta en cada trayectoria lógica; el perfil directo no hace esa llamada. El gasto de adquirir ambas ramas se registra aparte. Se conservan mensajes, hashes, resultados y uso informado; las generaciones rechazadas por el proveedor pueden carecer de contenido o consumo observado, y se identifican como tales. Las verificaciones iniciales se ejecutan en ambas copias; comparten el diagnóstico público observado primero. Las rutas absolutas de ejecución y las duraciones de unittest se sustituyen en los mensajes públicos; los logs originales se conservan. El orden de los candidatos depende del hash de la tarea, antes de conocer resultados.

Los controles son la primera edición barata con reparación fuerte, la primera edición fuerte con reparación fuerte y el checkpoint RouteLLM-BERT publicado, transferido explícitamente a GPT-OSS. El perfil `verified` de Relay coincide con la cascada barata; el perfil anterior `activity` empieza con el fuerte. Ninguno es aprendido. El código de aprendizaje reutiliza `TrajectorySuccessPredictor` para estimar éxito final y costo de cada continuación, pero todavía no hay un predictor entrenado de mantenimiento. Texto y estado público entran al predictor; resultados, consumos, grupo y partición son objetivos o procedencia y quedan fuera de sus variables.

La referencia reproduce el scoring del checkpoint fijado con un wrapper local; no ejecuta el SDK oficial, MF ni Not Diamond. El [controller oficial](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/controller.py) puntúa el último mensaje de usuario. Se conserva ese texto y la truncación original de BERT a 512 tokens, registrada por caso. El [router BERT](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/routers.py) tiene supervisión externa de preferencias; compararlo con Relay ajustado localmente mide adaptación frente a transferencia, no igualdad de supervisión. MF usa embeddings de una API según su [implementación oficial](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/matrix_factorization/model.py). No se activó otro servicio.

## Uso inmediato

Desde la raíz, comprueba configuración sin APIs:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_compare_maintenance.py check
```

El siguiente comando recoge un par para la demo ya conocida, puntúa BERT localmente y genera `comparison.md` y `comparison.json`. Pide la clave Groq de forma oculta y mantiene la cuenta Free y presupuesto declarado US$0. Reserva como máximo cinco llamadas reales, sin reenvíos automáticos. La carga inicial de BERT puede tardar. No aplica ningún cambio al original.

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_compare_maintenance.py run
```

**Ese ejemplo es un diagnóstico del comparador, no evidencia competitiva.** Su suite tiene `scope=diagnostic`; la CLI prohíbe usarla para entrenar o como holdout. Los proveedores simulados también quedan identificados y no entran al entrenamiento de la CLI.

Para continuar una adquisición con tareas pendientes, repite `run` con la misma `--suite`, `--partition`, modos y `--output`. Solo continúa después de pares terminados y si manifiesto, código y archivos siguen iguales. Una llamada incierta o un par incompleto bloquea el reenvío; hay que revisar el registro. `--max-calls` acepta hasta 30 por invocación y reserva cinco llamadas por caso con inspección, o cuatro en modo directo. `--collect-only` omite el cálculo local BERT y el reporte comparativo. La [guía Relay Fix](relay_fix.md) documenta el perfil directo y la suite de ocho casos.

La puntuación se puede ejecutar después sin credenciales:

```powershell
.\.venv-routellm\Scripts\python.exe tools\relay_compare_maintenance.py score "CARPETA_ADQUISICION"
.\.venv-groq\Scripts\python.exe tools\relay_compare_maintenance.py compare "CARPETA_ADQUISICION"
```

## Cohorte y condición para seguir

El 17 de noviembre se añadió `configs/maintenance_cases_v1.json`: ocho problemas autocreados de intervalos, dependencias, registros entrecomillados, ventanas temporales, configuración inmutable, caché, reparto decimal y rutas. Sus tests públicos y reservados se congelaron antes de la generación. Las soluciones de referencia pasan 58 tests y los ocho puntos de partida fallan al menos un test público. Esto comprueba la batería, pero no la convierte en una muestra industrial representativa. Todos son desarrollo diagnóstico; se prohíbe usarlos como ajuste o holdout. La aceptación tiene autoría compartida con las tareas y sus comprobaciones son finitas. No se declara independiente el BFCL ni el test histórico ya inspeccionado.

El manifiesto enumera todas las tareas y sus particiones `fit`, `calibration`, `development` y `holdout` antes de las llamadas. Tareas del mismo repositorio/problema pertenecen al mismo `group_id`; los grupos no cruzan particiones. Se rechazan duplicados por petición y archivos, sin usar resultados. El alcance de esa suite será `development_and_reserved_holdout`. Cada tarea referencia un config de mantenimiento con archivos explícitos, catálogo 20B/120B y una reparación permitida. No se descargan tareas ni se ejecutan repositorios externos automáticamente.

Una adquisición pequeña de desarrollo debe resolver primero si existe margen para routing. Si la cascada barata ya resuelve los casos y cuesta menos, ese resultado cuenta contra añadir complejidad. No se entrenan predictores de éxito cuando faltan ambas clases en ajuste o calibración; tampoco se inventan fallos para habilitarlos. Costos distintos entre trayectorias pueden justificar estudiar un predictor de consumo, pero ese camino no está implementado ni validado aquí.

Cuando hay datos suficientes, la CLI `train CARPETA_FIT CARPETA_CALIBRACION CARPETA_DESARROLLO --output CARPETA_POLITICA` usa únicamente esas tres particiones completas y compatibles. Ajusta vocabulario/predictores solo en fit, calibra en calibration y selecciona umbrales en development. Ambos selectores exploran doce puntos; BERT incluye los extremos siempre fuerte y siempre débil. El piso empírico de calidad en desarrollo es el éxito del control fuerte, sin margen de pérdida. Si no hay un punto elegible no se promueve una política. Se guardan hashes del predictor, política, fuentes y contrato de adquisición.

La adquisición de holdout exige `--policy CARPETA_POLITICA/policy.json --policy-sha256 HASH_DEVUELTO_AL_ENTRENAR` antes de la primera llamada; debe conservar la misma suite y contrato. La evaluación exige ese mismo digest y audita grupos y tareas. No se emite comparación de holdout si faltan pares o hubo fallos operativos sin resolver. La procedencia externa del SHA debe conservarse fuera del artefacto: un hash autocambiado no autentica al autor.

## Qué significarán las cifras

En las suites con `acceptance_files`, el éxito conjunto exige pasar las pruebas públicas y las reservadas. Las segundas se ejecutan sobre el resultado final en una copia separada: nunca entran al mensaje del modelo ni provocan reparaciones. Los errores operativos de aceptación detienen la comparación; no se descartan casos para aumentar la tasa. Un fallo público conocido basta para determinar fracaso conjunto, aunque el veredicto privado no se haya podido obtener; ese veredicto permanece desconocido. Las suites antiguas sin aceptación siguen midiendo únicamente tests visibles. Ninguna batería finita acredita corrección completa ni calidad productiva general.

El costo se calcula sobre tokens informados, incluyendo reparaciones, bajo un escenario común de tarifa pública **sin caché**. Las [tarifas de 20B](https://console.groq.com/docs/model/openai/gpt-oss-20b) y [120B](https://console.groq.com/docs/model/openai/gpt-oss-120b), verificadas el 15 de noviembre, coinciden con el archivo de tarifas del 12 de noviembre. No se imputa caché observada ni se presenta el escenario como factura. En Free no se demuestra ahorro monetario pagado. Un consumo desconocido conserva costo desconocido, nunca cero. Cómputo del router, pruebas, adquisición y entrenamiento no están monetizados; no se afirma reducción del costo operativo total.

Se reportan éxito, costo por tarea resuelta, consumo y diferencias pareadas. Los intervalos bootstrap agrupan por `group_id`, con 2000 réplicas y semilla fijada. Pocos grupos, datos usados para selección y distribuciones degeneradas no sostienen una conclusión confirmatoria; el reporte nunca proclama superioridad automáticamente. Las latencias de selección de BERT y Relay se miden separadas de carga; no se presentan como latencia real de extremo a extremo del replay.

El siguiente hito aceptable será una comparación nueva en la que Relay conserve calidad y reduzca costo frente a **esta referencia RouteLLM-BERT** y frente a la cascada barata, o mejore calidad a costo comparable, con incertidumbre y alcance explícitos. Una sola reparación correcta, menos tokens que el fuerte o una suite de tests internos aprobada no cierran ese hito. Si los resultados no apoyan la hipótesis, se registra y se detiene esa línea antes de construir más capas.

La revisión metodológica utilizó la skill local [scientific-critical-thinking](../.agents/skills/scientific-critical-thinking/SKILL.md) como guía; el protocolo del proyecto y las instrucciones del usuario tienen prioridad. Procedencia: Kassis, T., Agarwal, V., He, Y., Patel, D. y Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065); metadatos de arXiv verificados el 17 de noviembre. Es una referencia de procedimiento, no evidencia de eficacia de Relay.
