# Selección de skills

Selección específica para este proyecto, no un ranking universal. Se compararon el catálogo oficial OpenAI, su sucesor de plugins y Scientific Agent Skills de K-Dense. Se priorizaron contenido inspeccionable, tareas concretas, ejecución local, licencia identificable y pocas dependencias. Se fijaron snapshots inmutables consultados en GitHub; no se afirma que sean los últimos commits disponibles.

## Instalación por proyecto

Destino `.agents/skills/`, sin alterar skills personales. `configs/skills-lock.json` registra repositorios, commits y rutas. `tools/install_project_skills.py` usa el instalador oficial local, descarga solo los paquetes elegidos y conserva hashes de cada archivo en `reports/skills-installed.json`. No ejecuta sus demos ni instala sus dependencias. Las licencias se conservan en `references/skill-licenses/` y dentro de las skills cuando el origen las incluye.

`scikit-learn`, de [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/515742b7ed522c78f19bb9b61f346a0435785302/skills/scikit-learn), cubre pipelines, clasificación, evaluación, desbalance y separación por grupos. Aporta referencias para las cuatro técnicas del estudio. La cabecera declara BSD-3-Clause mientras el repositorio publica MIT; se conserva esa declaración sin decidir que una sustituye a la otra. El snapshot instalado declara Python 3.11+ y sklearn 1.7+; el entorno verificado usa sklearn 1.8.0. Verificar APIs contra esa versión, sin actualizar automáticamente el entorno.

`scientific-critical-thinking`, de [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/515742b7ed522c78f19bb9b61f346a0435785302/skills/scientific-critical-thinking), sirve para revisar hipótesis, sesgos, comparaciones y alcance de las conclusiones. Se usará al revisar el protocolo y redactar limitaciones. No reemplaza la búsqueda de evidencia primaria. Licencia MIT declarada.

`scientific-visualization`, de [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/tree/515742b7ed522c78f19bb9b61f346a0435785302/skills/scientific-visualization), orienta matrices de confusión, distribuciones, incertidumbre y curvas costo-calidad con datos trazables y exportación. Matplotlib y Seaborn son opcionales para la fase de figuras. Licencia MIT declarada.

`jupyter-notebook`, de [OpenAI](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431/skills/.curated/jupyter-notebook), aporta un generador local de notebooks y una estructura para experimentos reproducibles. Su scaffold usa la biblioteca estándar. Se selecciona esta copia con Apache-2.0 explícita aun cuando [el repositorio está deprecado](https://github.com/openai/skills/blob/49f948faa9258a0c61caceaf225e179651397431/README.md). La alternativa de plugins actual contiene referencias a conectores que no necesitamos para archivos locales. Al invocarla aquí, usar su ruta local `.agents/skills/jupyter-notebook`, no la ruta personal de los ejemplos.

## Opciones descartadas inicialmente

No se instala toda la colección científica. `statistical-analysis` introduce un stack más amplio que el requerido para estas primeras comparaciones. Se puede reconsiderar al definir el análisis estadístico final. `research-lookup` depende de búsqueda externa con API y un flujo de revisión extensa; por ahora bastan búsquedas web verificadas y la bibliografía del proyecto. Skills de bioinformática, entrenamiento profundo, despliegue, GPU y agentes productivos no corresponden al alcance offline tabular.

Las skills de presentaciones, documentos y PDF ya disponibles en la sesión se pueden invocar cuando haya que producir entregables. No hace falta duplicarlas en el proyecto.

## Límites de uso

Una skill es una guía para el agente. No agrega por sí sola paquetes Python, calidad estadística ni permisos para subir datos. `AGENTS.md` y el protocolo prevalecen sobre recomendaciones comerciales, instalaciones o servicios de terceros. No se activan cuentas, APIs, telemetría ni promociones. Se conserva el contenido original de las skills con su procedencia y se inspeccionan los scripts antes de ejecutarlos en futuras tareas.

Las skills instaladas estarán disponibles en el siguiente turno de Codex. Su instalación se confirma exclusivamente mediante `reports/skills-installed.json`; la presencia de este documento de selección por sí sola no acredita que estén instaladas.
