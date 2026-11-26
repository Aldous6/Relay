# Auditoría inicial

Archivo local verificado contra el SHA-256 oficial antes de abrirlo. Los resultados reproducibles están en `reports/audit.json`. No se entrenaron modelos ni se estimó desempeño final.

Se confirmaron 36,497 filas, 37 columnas, once modelos y 86 evaluaciones. No hay prompts vacíos ni nulos en los scores y costos. Hay 7,905 filas con alguna puntuación parcial. Al umbral 0.75, 1,505 filas originales no tienen ningún modelo elegible.

La normalización documentada detecta 16 filas adicionales duplicadas. Once grupos de prompts duplicados tienen resultados diferentes entre sus filas. Conservar la primera fila original independientemente de resultados deja 36,481 prompts únicos en total, de los cuales 34,977 son elegibles. La propuesta informaba 34,976: no se logró reproducir ese valor con la regla inicial implementada. No modificar las cifras para que coincidan sin reconstruir primero la regla anterior.

La cuenta `36,497 − 1,505 − 16` no puede aplicarse mecánicamente: las exclusiones por falta de elegibilidad y por duplicación se superponen. Con la regla adoptada se eliminan 15 filas elegibles duplicadas y una fila ineligible duplicada. El orden de deduplicar también importa cuando los resultados dentro de un grupo difieren.

La preparación reservó 29,157 prompts para train y 7,324 para test mediante hash. Los elegibles son 27,964 en train y 7,013 en test. Los casos sin etiqueta se conservan para futura simulación de política sobre la población completa. Estas cuentas describen soporte, no desempeño predictivo.

El mapa aprendido en train asigna T0 a mistralai/mistral-7b-chat y WizardLM/WizardLM-13B-V1.2; T1 a seis modelos intermedios; T2 a claude-v1, claude-v2 y gpt-4-1106-preview. El mapa exacto y la configuración se guardan en el manifiesto. No se reutiliza la columna oracle original.
