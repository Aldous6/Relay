# Diccionario de los 14 atributos

Definiciones operacionales v1, implementadas en `src/routerbench/features.py`. El texto se reconstruye antes de extraerlos. No se aplica la normalización usada para duplicados. Todos los resultados son numéricos y el denominador de cada proporción es la cantidad total de caracteres, incluidos espacios.

`n_chars`: longitud de Python `len(text)`, medida en puntos de código Unicode, no tokens del LLM ni bytes. `n_words`: coincidencias Unicode de `\b\w+\b`, incluidos dígitos y guion bajo; no es una segmentación lingüística fiable para todos los idiomas. `n_lines`: cantidad de elementos de `splitlines()`; vacío tiene cero líneas y un salto final no agrega una línea vacía.

`avg_word_length`: suma de las longitudes de las palabras detectadas dividida por su cantidad. `lexical_diversity`: número de palabras distintas tras casefold dividido por cantidad de palabras. Si no hay palabras, ambas variables son cero.

`digit_ratio`: fracción de caracteres con `isdigit()`. `punctuation_ratio`: fracción con categoría Unicode que empieza con P. `uppercase_ratio`: fracción con `isupper()`. Estas operaciones siguen la base Unicode de la versión de Python registrada.

`n_questions`: cantidad de caracteres `?`, `？` y `¿`, por lo que una pregunta española puede aportar dos. `n_brackets`: cantidad de caracteres de `()[]{}`. `n_code_markers`: coincidencias no superpuestas de la expresión regular `CODE_PATTERN`, que incluye triple backtick, `=>`, `::`, `#include` y palabras clave fijas sensibles a mayúsculas. Palabras como “class” o “from” también pueden aparecer en prosa: es una señal superficial.

`n_math_symbols`: cantidad de caracteres del conjunto fijo `MATH_SYMBOLS` del módulo. Incluye operadores ASCII y símbolos Unicode, y puede contar puntuación usada fuera de matemáticas. `non_ascii_ratio`: fracción de caracteres con punto de código mayor a 127. `cjk_ratio`: fracción dentro de los rangos fijos `CJK_RANGES`, que aproximan Han, hiragana, katakana y hangul. No se interpreta como detección del idioma ni como cobertura exhaustiva de toda escritura asiática.

El conjunto de columnas se fija mediante `FEATURE_NAMES`. Las limitaciones de estas heurísticas forman parte del experimento: la pregunta es si señales superficiales y baratas aportan predicción, no si miden dificultad semántica directamente.
