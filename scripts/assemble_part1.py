#!/usr/bin/env python3
from pathlib import Path
parts = sorted(Path("scripts").glob("install_part1_chunk*.txt"))
text = "".join(p.read_text() for p in parts)
Path("scripts/install_part1.py").write_text(text)
print("assembled", len(text))
exec(open("scripts/install_part1.py").read())
