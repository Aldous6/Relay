"""Calcula el ingreso de las ventas de la tienda ficticia."""
import csv
from pathlib import Path

ventas = Path(__file__).with_name("ventas.csv")
with ventas.open(encoding="utf-8", newline="") as archivo:
    filas = csv.DictReader(archivo)
    ingreso = sum(int(fila["precio_unitario"]) for fila in filas)

print(f"Ingreso total: {ingreso}")
