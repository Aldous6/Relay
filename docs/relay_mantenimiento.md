Relay: entrega parcial de mantenimiento Python. Dirección elegida con el usuario el 15 de noviembre de 2026. El objetivo de esta entrega es un flujo utilizable de petición, inspección, edición, verificación y aplicación revisada. El agente usa la librería de Relay; los estudios históricos conservan sus resultados y dejan de gobernar el siguiente entregable.

Desde la raíz del proyecto, para ejecutar el ejemplo con la cuenta Groq Free ya autorizada:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_maintain.py run
```

La entrada usa `.venv-groq`, solicita la clave de Groq de forma oculta si falta `GROQ_API_KEY`, y la conserva únicamente en memoria. No la pegues en el chat ni en configuración. La ejecución llama a los modelos y consume cuota de esa cuenta. No activa facturación ni otro servicio. La declaración Free proviene de la cuenta autorizada por el usuario; el cliente no puede consultar ni garantizar su configuración de facturación. No se deben utilizar credenciales de una cuenta pagada con esta configuración. El lanzador alternativo `Start-RelayMaintenance.ps1` ofrece el mismo flujo; este equipo bloqueó los scripts PowerShell durante la revisión, por eso la instrucción principal invoca Python directamente sin cambiar políticas del sistema.

El ejemplo contiene una función Python con un defecto intencional al normalizar etiquetas y cinco pruebas existentes. Es un ejemplo de integración pequeño y reproducible, no una medición representativa de capacidad. Primero se ejecutan esas pruebas en una copia de los archivos declarados. El modelo de inspección recibe la petición, fuentes y tests; propone los archivos que necesitan cambiar. El modelo de edición devuelve sustituciones exactas. El agente aplica esas sustituciones en la copia y vuelve a ejecutar las mismas pruebas. Si fallan, permite una reparación adicional. Un timeout, problemas de importación o falta de pruebas bloquean el flujo; un error de sintaxis introducido por la edición puede repararse dentro del límite.

Para validar rutas y configuración sin usar modelos:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_maintain.py check
```

Cada ejecución crea una carpeta nueva en `outputs/maintenance/`. `report.md` resume el resultado; `report.json` conserva etapas, decisiones, consumo disponible y verificaciones; `events.jsonl` registra cada llamada antes y después de enviarla; `changes.diff` muestra el cambio cuando existe una propuesta. `responses/call-NNN.json` conserva la respuesta entregada por el adaptador antes de validar sus campos, con hash y referencia desde la llamada. Los rechazos internos del proveedor, JSON malformado o respuestas truncadas pueden terminar antes de esa captura. `worktree/` contiene la propuesta y `originals/` la copia previa. Los logs completos de cada verificación quedan en archivos `test-*.stdout` y `test-*.stderr`, enlazados desde el informe. La consola muestra la carpeta concreta.

La inspección prioriza archivos; los permisos los fija la configuración. El formato solicitado usa JSON Schema estricto con las rutas editables reales como valores permitidos, tanto para inspección como para edición. La compatibilidad de este formato con ambos GPT-OSS está documentada en [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs). El presupuesto local también reserva espacio para el esquema. La selección se normaliza solo para barras Windows y prefijos `./`, se eliminan duplicados y se descartan rutas ajenas o de tests. Si no queda ninguna ruta utilizable, el flujo conserva el conjunto original de `source_files`, anuncia el ajuste y registra la sugerencia y el resultado en `inspection_selection`. No se resuelven `..`, rutas absolutas o nombres incompletos hacia archivos autorizados. La edición vuelve a comprobar permisos y las pruebas siguen siendo inmutables.

Cuando el estado sea `ready_for_review`, revisa el diff. Para aplicar esa propuesta:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_maintain.py apply 'outputs\maintenance\CARPETA_DE_LA_EJECUCION'
```

La aplicación comprueba que fuentes y pruebas originales no hayan cambiado desde el inicio y que la propuesta sea exactamente la que pasó verificación. Si hay conflicto, se detiene. Los tests son inmutables para el modelo. No se permite que un cambio apruebe silenciosamente ejecutando menos pruebas u omitiendo más casos. El estado aprobado significa que pasan esas pruebas; requiere revisión para aceptar que se cumplió toda la petición.

La política inicial `activity_policy_v1` asigna inspección a `groq/openai/gpt-oss-20b` y edición/reparación a `groq/openai/gpt-oss-120b`. Es una política configurable por actividad, no un predictor aprendido del mejor modelo. No usa los bundles históricos de GPT-4/Mixtral, no comunica probabilidades inventadas ni acredita superioridad sobre RouteLLM o Not Diamond. La integración deja un punto explícito de sustitución en `router.route(activity, attempt=...)`. El objetivo posterior es mejorar esa selección conservando el mismo flujo de producto.

Se presupuestan hasta tres llamadas, 2048 tokens máximos de salida y 8000 unidades estimadas por solicitud, con espera mínima de 60 segundos entre llamadas al mismo modelo. Si están disponibles, se reutilizan únicamente los assets locales verificados del tokenizador GPT-OSS; no se cargan tareas ni resultados BFCL. Sin esos assets se usa una estimación conservadora por bytes y se declara en el informe. La estimación no garantiza la plantilla del servidor. Un rechazo de cuota o una llamada incierta termina la ejecución sin reenvío automático. Consumo ausente significa desconocido. Los importes no se inventan a partir de la cuenta Free.

Para usar otro repositorio Python, copia `configs/maintenance_demo.json` a un archivo local, cambia `repository` (relativo a la configuración), y declara cada archivo editable en `source_files` y cada prueba existente en `test_files`. Incluye los módulos locales y `__init__.py` que necesiten las importaciones. Esta entrega soporta unittest, archivos UTF-8 existentes y cambios acotados; no instala dependencias ni inventa comandos de prueba. Los archivos deben ser explícitos: no se recorren datasets, secretos o carpetas completas. Los límites por archivo y conjunto se verifican antes de pedir la clave; el contexto disponible puede requerir un alcance más pequeño.

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_maintain.py run --config 'configs\mi_mantenimiento.local.json' --task 'Describe aquí el cambio concreto'
```

Las pruebas ejecutan Python con los permisos del usuario. La copia aísla los cambios y retira credenciales del entorno de los procesos de prueba; no es un sandbox de seguridad. Utiliza repositorios de confianza y revisa los archivos autorizados antes de enviarlos a Groq. Los comandos de verificación son fijos y nunca se toman de las respuestas del modelo.

La entrada independiente del lanzador es `python tools/relay_maintain.py run --config configs/maintenance_demo.json --task "Petición"`. Después de instalar el paquete se expone también `relay-maintain`. Se puede utilizar `--mode always_strong` para ejecutar el mismo flujo con el modelo de edición en todas las etapas; requiere otra ejecución y su propia cuota, y no constituye una comparación científica automática. La API de biblioteca es `relay_router.maintenance.run_maintenance`, con proveedor y router inyectables.

La [verificación ejecutada](../reports/maintenance-delivery-verification.json) registra 102 pruebas aprobadas, cero fallos y una omitida porque Windows no permitió crear un symlink. La protección contra junctions sí se comprobó. Incluye regresiones del SDK existente y el flujo completo: pruebas inicialmente fallidas, propuesta con cinco pruebas aprobadas y aplicación sobre una copia del ejemplo. El [informe del flujo](../outputs/maintenance-delivery-check/20261116T034844029030Z/run/report.md) y su [diff](../outputs/maintenance-delivery-check/20261116T034844029030Z/run/changes.diff) son inspeccionables.

La verificación offline usa respuestas programadas explícitas, solicitudes HTTP simuladas, archivos reales y procesos reales de unittest. Acredita integración, manejo de errores y conservación de archivos. No acredita que Groq haya generado correctamente una solución real. No se levantó ninguna aplicación para revisión visual.

La primera ejecución real reportada por el usuario, `20261116T041148101846Z` (UTC), completó la inspección de Groq y registró 2262 tokens. Se bloqueó al validar `files`; esa versión no guardó la selección recibida, por lo que su valor exacto es desconocido. Se conservaron la corrida y sus archivos sin reetiquetar el resultado. La corrección elimina rutas ficticias en prompts, usa el esquema estricto, conserva las respuestas y trata la inspección como sugerencia dentro del alcance autorizado. La [verificación de la corrección](../reports/maintenance-inspection-fix-verification.json) registra 66 pruebas aprobadas y una omitida por permisos de symlink, más un flujo completo offline con reservas 5065 y 5376 sobre el máximo 8000. Esa comprobación offline por sí sola no acredita generación real; ninguna corrida se reenvía automáticamente.

La ejecución posterior `20261116T042531066594Z` (UTC) sí completó generación y verificación reales: 20B inspeccionó, 120B editó y las mismas cinco pruebas pasaron después de tres fallos iniciales. No hizo falta reparación adicional ni ajuste de la selección de archivos. El consumo informado fue 5219 tokens; el costo monetario sigue desconocido. Se revisó el [diff real](../outputs/maintenance/20261116T042531066594Z/changes.diff): elimina espacios y vacíos, usa casefold, preserva orden al quitar duplicados y no modifica la entrada. La [revisión con hashes](../reports/maintenance-first-live-review.json) confirma que la propuesta coincide con la verificada y que los originales siguen intactos. Está lista para aplicar mediante el comando `apply` de esta guía. Este caso confirma el funcionamiento del ejemplo con modelos reales, no calidad general del routing ni ahorro frente a otra política.

Contratos del proveedor consultados el 15 de noviembre de 2026: [API de Groq](https://console.groq.com/docs/api-reference), [salidas JSON](https://console.groq.com/docs/structured-outputs) y [cuotas](https://console.groq.com/docs/rate-limits). Respaldan el formato del adaptador; no resultados de Relay.
