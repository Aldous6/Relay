# Host para tareas de repositorio

`tools/repository_snapshot_host.py` implementa `prepare`, `build_agent` y `private_verdict` para `tools/evaluate_repository_cascade.py`. Usa el editor local de OpenHands y un contenedor Linux nuevo para cada verificación. El operador prepara una imagen por digest, sin credenciales, con Python y las dependencias de prueba instaladas. Cada check corre sin red, con montaje de entradas de solo lectura, usuario sin privilegios y un directorio temporal de trabajo. La CLI de Docker usa el motor local y exige que la imagen ya esté disponible.

Define `RELAY_REPOSITORY_CASES` con la carpeta de los casos preparados. Dentro de ella cada subdirectorio se llama exactamente como su `task_id` y contiene lo siguiente.

```text
<task_id>/
  case.json
  snapshot/
    relay_public_check.py
    src/...
  private_check.py
```

En `case.json` copia `task_id`, `source` y `revision` de la cohorte. Indica la imagen por digest y la lista de archivos editables. Este ejemplo ilustra el formato y requiere los valores del caso real.

```json
{
  "task_id": "example-project-123",
  "source": "https://github.com/example/project",
  "revision": "<commit completo>",
  "image": "example/runner@sha256:<64 dígitos hexadecimales>",
  "base_path": "/project",
  "editable_paths": ["src/package/module.py"]
}
```

Para un proyecto pequeño, omite `base_path` e incluye los archivos necesarios en `snapshot`. Para un repositorio grande, la imagen contiene su checkout en `base_path`. La verificación comprueba el commit, copia el árbol sin historial Git al directorio temporal y aplica allí los archivos de la instantánea. El agente ve únicamente los archivos públicos preparados y puede modificar los archivos existentes declarados en `editable_paths`. La instantánea admite hasta 1000 archivos y 20 MB. La selección de archivos debe hacerse a partir del issue y del árbol original, antes de consultar el parche o las pruebas privadas. El expediente y el sistema de archivos quedan fuera del alcance del código candidato en el host.

Los dos verificadores son programas Python confiados por el operador. Se ejecutan por separado, con el directorio temporal del repositorio como directorio actual y con `python -I -B`. Cada uno debe localizar el proyecto en esa copia, verificar que se ejecutaron pruebas pertinentes y devolver 0 si pasan o 10 si una aserción falla. Otros códigos, una excepción o un timeout indican resultado inconcluso. Las dependencias del verificador se congelan en el digest de la imagen. El verificador privado se monta solo al cerrar las tres ramas y su salida no vuelve al agente. Para clasificación reproducible, comprueba ambos verificadores con un control que deba pasar y otro que deba fallar antes de congelar la campaña.

El preflight estático enumera todos los casos faltantes y termina con código 2 si falta alguno. La opción `--inspect-images` verifica además su presencia en el motor local sin generar respuestas ni iniciar contenedores. El paso `freeze` también exige esa inspección. Para adquisición, el host lee la credencial de Groq del entorno del proceso de OpenHands.

```powershell
python tools/repository_snapshot_host.py preflight --cohort data/imported/relay-repository-v1/cohort.json
```

La cohorte seleccionada contiene veinte issues de SWE-bench-Live Lite. Este host permite evaluarlos como una adaptación sobre instantáneas preparadas. El protocolo oficial restringe la información del agente al issue y a la imagen de cada instancia, y prescribe un rollout. Este host compara tres ejecuciones, limita la vista de archivos y usa un verificador privado del operador. Para atribuir resultados al evaluador oficial se necesita conservar los parches elegidos, ejecutarlos con la versión fijada del harness de Microsoft contra las imágenes correspondientes y contrastar sus veredictos. Las pruebas unitarias de este host ejercitan las rutas de archivos, permisos del editor, clasificación y construcción del comando mediante dobles. La ejecución de las imágenes requiere prepararlas en el host del estudio. [Protocolo de evaluación SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live/blob/main/evaluation/README.md).

## Expedientes oficiales de aceptación

La selección completa de veinte IDs tiene sus entradas de evaluación upstream en `data/imported/relay-repository-acceptance-v1/official_cases.jsonl.gz` (446899 bytes). Su manifiesto enlaza cada issue y commit con las huellas de su fila y de su parche de **pruebas**, su imagen publicada y el número de pruebas de cada tipo. Hay 66 entradas `FAIL_TO_PASS` y 63839 `PASS_TO_PASS` en el conjunto. El materializador valida el SHA-256 del Parquet antes de proyectar diez columnas explícitas. Conserva literalmente los comandos de prueba y reconstrucción y el parser como datos privados. El preflight verifica esa huella sin importar el expediente en el runtime ni ejecutar su contenido.

```powershell
python -I -B tools/prepare_repository_acceptance.py verify
python -I -B tools/repository_snapshot_host.py preflight --cohort data/imported/relay-repository-v1/cohort.json --official-acceptance data/imported/relay-repository-acceptance-v1
```

Para reproducir desde el Parquet local, usa `prepare --private <directorio-nuevo>` en el primer comando. Para entregar las veinte filas al harness oficial, usa `export --jsonl <archivo-nuevo.jsonl>` en una carpeta privada existente. La exportación verifica archivo comprimido, huellas de cada fila, cobertura y commits. Su SHA-256 descomprimido es `2dc64d49b4e4939bb9fb8b3f34198f3341f8b5d90574f639b8d9622bf76a366a`.

El JSONL sirve como `--dataset` para `evaluation.evaluation` con predicciones externas. El evaluador consume un archivo JSON por brazo con cada `instance_id` asociado a su `model_patch`; una tarea sin parche conserva una entrada vacía y cuenta como tal. Las veinte referencias a imágenes publicadas carecen de digest en el Parquet y deben resolverse y fijarse antes de ejecutar el estudio. También faltan una revisión fijada del harness y sus dependencias, los snapshots y checks operacionales, y los parches producidos por los agentes. Los tests privados upstream ya están preparados. El preflight informa su cobertura aparte de la preparación de los casos de trabajo. [Implementación oficial del evaluador](https://github.com/microsoft/SWE-bench-Live/blob/main/evaluation/evaluation.py).
