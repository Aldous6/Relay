# Relay Fix: mantenimiento directo

Versión experimental disponible desde el 17 de noviembre de 2026. Recibe una petición y archivos Python autorizados, prepara el cambio con GPT-OSS 120B, ejecuta las pruebas públicas y permite una reparación con 120B cuando hace falta. El perfil predeterminado se eligió después del piloto: la cascada barata tuvo peor costo y menor éxito que empezar con 120B. Esto es una decisión de producto basada en desarrollo, no una validación independiente del nuevo perfil. Una credencial inválida, una cuota agotada o una petición de resultado incierto detiene el flujo. Los originales solo cambian al aplicar expresamente el diff revisado.

La selección predeterminada es una referencia fija, no un predictor aprendido ni una garantía de elegir el mejor modelo. La cascada 20B→120B sigue disponible con `--mode verified` para comparación explícita. El [piloto ejecutado](../outputs/maintenance-comparison/20261117-live-v3-complete/delivery_report.md) evaluó esa cascada antes de la corrección de saltos de línea: obtuvo 2/8 éxitos y el mismo costo que BERT a umbral fijo, mientras que 120B primero obtuvo 3/8 y costó menos. No se ha ganado una comparación competitiva. Las pruebas de aceptación reservadas pertenecen al comparador y nunca se envían como feedback para reparar.

Desde PowerShell en la raíz del proyecto:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_fix.py
```

El ejemplo predeterminado usa `maintenance_demo`. Para un cambio propio:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_fix.py --config configs\mi_repositorio.json --task "Describe aquí el cambio requerido"
```

El config conserva el contrato de [mantenimiento](relay_mantenimiento.md): repositorio de confianza, archivos fuente y tests explícitos, cuenta Groq Free y límites acotados. No se busca ni se persiste la clave; la CLI la solicita de forma oculta si no está en el proceso. Las pruebas ejecutan código con los permisos del usuario.

El perfil omite la inspección LLM redundante y utiliza el alcance de archivos declarado. Solicita JSON por instrucciones, sin validación de formato en el servidor; Relay valida localmente el objeto, los campos, las rutas autorizadas y las sustituciones antes de aplicar un cambio en la copia. Se admite únicamente equivalencia LF/CRLF en archivos y fragmentos de estilo uniforme: la indentación, los demás caracteres y la unicidad de la coincidencia siguen siendo exactos. Se conserva el estilo de saltos del archivo; no se adivina una convención para saltos mezclados. La respuesta inválida consume un intento y su uso informado se conserva. El razonamiento no se solicita en el contenido de la respuesta.

La [verificación de la corrección Windows](../reports/maintenance-line-ending-diagnostic-v1/report.md) reaplicó localmente las cinco respuestas rechazadas por saltos de línea: todas pasaron los tests públicos y dos pasaron también los privados. No hizo llamadas nuevas, no cambió los resultados originales y no constituye una medición competitiva de esta versión corregida.

Después de revisar `report.md` y `changes.diff`, se aplica con:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_fix.py apply "CARPETA_DE_LA_PROPUESTA"
```

Para repetir el comparativo de ocho problemas propios, conservando pruebas privadas y controles comunes:

```powershell
.\.venv-groq\Scripts\python.exe tools\relay_compare_maintenance.py run --suite configs\maintenance_cases_v1.json --mode verified --reference-threshold 0.5 --inspection-mode direct --output-mode prompt_json --max-calls 26
```

La salida incluye la comparación con BERT a umbral fijo, además del BERT cuyo umbral se selecciona con esos mismos casos de desarrollo. La segunda cifra no es holdout. Los ocho casos son autocreados y no representan toda la actividad de mantenimiento. Repetirlos para corregir la integración no convierte los resultados en una evaluación independiente.

Los costos monetarios son un escenario de tarifa sin caché sobre tokens informados, no una factura ni ahorro pagado en Free. El consumo desconocido conserva costo desconocido. El informe registra también las adquisiciones técnicas previas; no borra errores para mejorar la comparación. Pasar tests finitos no demuestra corrección completa del requisito.
