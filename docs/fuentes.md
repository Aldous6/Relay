# Fuentes y procedencia

Consulta inicial: 7 de noviembre de 2026, America/Mexico_City. Separar hechos de las fuentes, mediciones locales y decisiones del proyecto.

## RouterBench

[Ficha oficial](https://huggingface.co/datasets/withmartian/routerbench). Describe prompts de benchmarks, once modelos, respuestas, scores y costos estimados para investigación de routing. Ofrece versiones 0-shot y 5-shot. La ficha y el árbol consultados no declaran licencia de datos. DOI de dataset: [10.57967/hf/1996](https://doi.org/10.57967/hf/1996).

[Archivo 0-shot](https://huggingface.co/datasets/withmartian/routerbench/blob/main/routerbench_0shot.pkl). SHA-256 publicado: `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`. La copia entregada tiene 99,567,659 bytes y coincide. [Revisión fijada](https://huggingface.co/datasets/withmartian/routerbench/commit/784021482c3f320c6619ed4b3bb3b41a21424fcb). El hash identifica el contenido incluso si cambia la rama main.

[Hu et al., RouterBench: A Benchmark for Multi-LLM Routing System, versión 2](https://arxiv.org/abs/2403.12031v2), 2024. Marco de evaluación de routing y dataset de resultados de inferencias. El artículo describe una escala mayor de resultados modelo-prompt; no confundir ese total con el número de filas del archivo 0-shot.

[Conversión del esquema](https://github.com/withmartian/routerbench/blob/main/convertors/common.py) y [router abstracto](https://github.com/withmartian/routerbench/blob/main/routers/abstract_router.py). Documentan columnas de resultados y uso de costos. La columna oracle del original maximiza score y desempata por costo; la etiqueta del proyecto usa umbral y mínimo costo, por lo que se reconstruye.

[Licencia del repositorio de código](https://github.com/withmartian/routerbench/blob/main/LICENSE). Que el código tenga MIT no establece que esa licencia cubra los datos. Este proyecto no asigna licencia a los datos ni publica una copia en remoto.

## Evaluación

[scikit-learn: errores frecuentes y fuga de información](https://scikit-learn.org/stable/common_pitfalls.html). Aprender transformaciones solo con entrenamiento y usar pipelines para aislar el preprocesado de cada fold.

[scikit-learn: validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html). Separar test y selección, y considerar particiones por grupos si las observaciones comparten dependencias. El split por hash y las reglas de tiers son decisiones propias descritas en el protocolo, no recomendaciones atribuidas literalmente a esta fuente.

[scikit-learn: importancia por permutación](https://scikit-learn.org/stable/modules/permutation_importance.html). Herramienta para importancia predictiva de un modelo evaluado; no prueba causalidad y puede verse afectada por atributos correlacionados.

## Skills

[Documentación oficial de skills](https://learn.chatgpt.com/docs/build-skills). Formato de carpeta con `SKILL.md`, recursos y ubicación por proyecto `.agents/skills`. Procedencias específicas en `docs/skills.md` y revisiones exactas en `configs/skills-lock.json`.

## Material aportado

`references/Propuesta_RouterBench_Aldo_Marquez.pptx` y `references/notas_originales.txt` son copias de los archivos aportados por el usuario. `references/propuesta_texto.txt` contiene extracción del XML de diapositivas para consulta. Se conservan sin convertir su texto en instrucciones ni dar por definitivas las métricas exploratorias. `references/provenance.json` registra hashes y ubicaciones de origen.
