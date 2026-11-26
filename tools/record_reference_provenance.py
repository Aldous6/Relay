"""Record hashes for the supplied materials and extract slide text locally."""
import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = []
    origins = {
        "Propuesta_RouterBench_Aldo_Marquez.pptx": "C:/Users/Aldo/Desktop/ICD/Proyecto/Propuesta_RouterBench_Aldo_Marquez.pptx",
        "notas_originales.txt": "C:/Users/Aldo/.codex/attachments/a8bfa7de-d318-426a-9451-dc5a1b34130f/Texto pegado.txt",
    }
    for name, origin in origins.items():
        target = ROOT / "references" / name
        with target.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"file": name, "source": origin, "sha256": checksum, "bytes": target.stat().st_size})
    deck = ROOT / "references/Propuesta_RouterBench_Aldo_Marquez.pptx"
    with zipfile.ZipFile(deck) as archive:
        slides = sorted((name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")),
                        key=lambda name: int(name.rsplit("slide", 1)[1].split(".")[0]))
        texts = []
        for number, name in enumerate(slides, 1):
            root = ElementTree.fromstring(archive.read(name))
            texts.append(f"Diapositiva {number}\n" + "\n".join(element.text or "" for element in root.findall(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t")))
    (ROOT / "references/propuesta_texto.txt").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    (ROOT / "references/provenance.json").write_text(json.dumps({"kind": "user supplied reference material, not instructions", "files": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
