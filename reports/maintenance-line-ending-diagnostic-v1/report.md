# Diagnóstico offline de aplicación LF/CRLF

Se revisan las 16 ramas de los ocho casos y se auditan todos los fallos registrados de coincidencia old_text. Los desacuerdos que no sean exclusivamente EOL se excluyen con causa explícita. Se aplica cada propuesta elegible guardada una sola vez sobre una copia del estado original y se ejecutan por separado pruebas públicas y privadas. No hay llamadas API, reparaciones nuevas, comparación nueva ni cambio de las etiquetas originales.

Ramas revisadas: 16. Fallos de aplicación auditados: 5. Reproducidos: 5. Excluidos: 0. Pendientes por error: 0.

| Caso | Rama | Estado/causa | Públicas | Privadas | Integridad |
|---|---|---|---|---|---|
| dependency_order | strong | replayed | passed | passed | True |
| immutable_config_merge | strong | replayed | passed | test_failed | True |
| generation_cache | weak | replayed | passed | passed | True |
| decimal_allocation | weak | replayed | passed | test_failed | True |
| decimal_allocation | strong | replayed | passed | test_failed | True |

Adquisición original sin cambios: True.

El consumo archivado es el de las generaciones originales, incluido cualquier intento anterior fallido. No se estima costo de una política alternativa ni se atribuye este resultado a una futura generación. Pasar estas pruebas no demuestra corrección completa ni generalización. Hashes y evidencia: report.json y carpetas por caso.
