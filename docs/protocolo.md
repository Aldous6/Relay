# Protocolo inicial, versión 1

Fecha de decisión: 7 de noviembre de 2026, America/Mexico_City. Basado en la propuesta del 5 de noviembre y en las notas entregadas por el autor. Este documento fija la preparación inicial; no comunica resultados de modelos.

## Preguntas

P1: ¿14 variables estructurales permiten predecir el tier del modelo de menor costo que alcanza score ≥ 0.75 mejor que las referencias? Métrica principal F1-macro, con etiquetas [0,1,2] explícitas y zero_division=0. Complementar con F1 y recall de Tier 2, exactitud balanceada, soporte y matriz de confusión.

P2: ¿qué atributos contribuyen a predecir Tier 2? Estimar importancia por permutación con F1 de Tier 2 en validaciones de entrenamiento y reportar estabilidad por folds/semillas. Congelar el procedimiento antes del test. Interpretar asociaciones predictivas, sin afirmaciones causales.

P3: ¿qué costo y tasa de éxito obtiene la política frente a modelos fijos? Consultar los scores y costos históricos del representante realmente seleccionado por cada predicción. Reportar costo por 1,000 prompts, fracción con score ≥ umbral y diferencia pareada de calidad. El ahorro se expresa respecto al costo del Premium fijo y junto con la pérdida o ganancia de calidad. Los costos son estimaciones históricas del benchmark, no facturación ni precios actuales.

## Datos y etiqueta

Se usa exclusivamente `routerbench_0shot.pkl`, revision y SHA-256 fijados en configuración. No mezclar 5-shot. Descubrir los once pares `<modelo>` y `<modelo>|total_cost` y validar valores finitos, scores en [0,1], costos no negativos. Mantener original inmutable y preservar identificador de fila.

El prompt puede ser una lista de strings serializada. Reconstruir con `ast.literal_eval` solo cuando el contenido corresponde a lista de strings y unir turnos con salto de línea. Nunca usar `eval`. El texto usado para atributos conserva mayúsculas, líneas e indentación. Para deduplicar se aplica NFKC, casefold y colapso de espacios, seguido de SHA-256. Conservar la primera fila en el orden original, sin elegir según score o costo. Registrar grupos con resultados distintos. Esta regla puede diferir del conteo de duplicados de la propuesta.

El mapa de tiers se estima ordenando los modelos por mediana de costo en entrenamiento; desempatar por nombre de modelo. Los dos primeros son T0, los seis siguientes T1 y los tres restantes T2. El tamaño de grupos es una decisión heredada de la propuesta, no una propiedad universal de los LLM. Fijar el mapa antes de crear las etiquetas de train y test.

Por fila, considerar modelos con score ≥ 0.75 y escoger el de menor costo guardado. Desempatar por nombre de modelo. La etiqueta hereda su tier. Si ningún modelo es elegible, `tier` permanece nulo y `eligible=False`. `oracle_model_to_route_to` del archivo tiene otra definición y no se usa.

## Partición y prevención de fuga

La propuesta plantea 80/20 estratificado por tier y un mapa calculado con medianas. Para evitar usar costos del futuro test al definir ese mapa, esta implementación inicial usa una asignación por hash de prompt y semilla 42, con probabilidad 0.20 de test. El porcentaje realizado puede diferir de 20 %. Se asignan los conjuntos antes de consultar resultados para calcular tiers. No se garantiza estratificación por tier. Registrar soportes y no cambiar la semilla para obtener resultados favorables.

Alternativa aún abierta: fijar un mapa externo ex ante y luego estratificar. Requiere documentar una revisión del protocolo y regenerar artefactos, sin reutilizar desempeño del test. No estimar un mapa con un train provisional y luego volver a dividir.

Tras congelar el mapa de train se pueden crear las etiquetas. La auditoría descriptiva sobre todo el archivo no entrena ni selecciona modelos; sus medianas globales se identifican como descriptivas y nunca alimentan `fit_tiers`. Las particiones se persisten junto con hash y configuración. El test aún no se ha usado para medir predicción.

La validación cruzada de 5 folds es estratificada dentro del train elegible, con semilla fija. Confirmar soporte suficiente de las tres clases en cada fold. El mapa modelo→tier se mantiene como definición de la tarea fijada en el train exterior; no se vuelve a aprender con test. Si se busca evaluar todo el procedimiento de estimación de tiers, hace falta un experimento anidado adicional, con etiquetas y mapa propios en cada fold. Documentar esta distinción.

## Modelos y política

Comparar Dummy de mayoría, árbol de profundidad 2 que observa solo `n_chars` como referencia por longitud, regresión logística con RobustScaler y pesos balanceados, y Random Forest con pesos balanced_subsample. El árbol por longitud se ajusta solo en el entrenamiento de cada fold. Las fábricas están implementadas; no se ha ejecutado la comparación.

Seleccionar hiperparámetros y modelo por validación de train. Fijar antes de evaluar test una regla para elegir un representante de cada tier: mayor tasa de éxito en train, desempate por menor costo medio y luego por nombre. Si se compara el desempeño de políticas dentro de CV, escoger representantes en el train de cada fold. Premium fijo es el representante T2. Añadir todos los once modelos fijos como contexto para evitar una referencia elegida a conveniencia. El oracle es una cota retrospectiva, no una política desplegable.

Clasificación: evaluar únicamente filas elegibles y reportar cuántas se excluyen. Política: evaluar sobre TODOS los prompts del test deduplicado, incluidos los no elegibles; presentar adicionalmente el subconjunto elegible. Acierto de tier no garantiza que el representante alcance el umbral. No consultar el mejor modelo del test para cambiar la selección.

## Sensibilidad y límites

Repetir el análisis con umbrales 0.50 y 1.00 usando la misma asignación exterior. No elegir el mejor umbral mirando el test. Añadir una evaluación secundaria dejando fuera grupos `eval_name` o familias documentadas, con nuevo ajuste de tiers solo en ese train. Mantener grupos de prompts compartidos fuera de ambos conjuntos para evitar fuga. Un split aleatorio no demuestra generalización a nuevos benchmarks.

Para incertidumbre de costo y calidad usar bootstrap pareado de los mismos prompts, con semilla y número de réplicas registrados. Para generalización entre benchmarks considerar remuestreo por familias. No interpretar intervalos ingenuos de folds correlacionados como pruebas independientes.

Los scores provienen de evaluaciones heterogéneas. El umbral común no representa necesariamente la misma calidad semántica en cada tarea. Las métricas superficiales pueden reconocer formatos de benchmark. No afirmar utilidad en agentes reales, causalidad ni ahorro comercial. La licencia formal de los datos permanece sin especificar en la ficha consultada. No se redistribuyen públicamente los originales.
