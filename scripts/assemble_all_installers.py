#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
parts = sorted(Path("scripts").glob("isa_c*.txt"), key=lambda p: int(p.stem.replace("isa_c","")))
Path("scripts/install_sources_a.py").write_text("".join(p.read_text() for p in parts))
print("assembled install_sources_a", Path("scripts/install_sources_a.py").stat().st_size)
parts = sorted(Path("scripts").glob("isb2_c*.txt"), key=lambda p: int(p.stem.replace("isb2_c","")))
Path("scripts/install_sources_b2.py").write_text("".join(p.read_text() for p in parts))
print("assembled install_sources_b2", Path("scripts/install_sources_b2.py").stat().st_size)
subprocess.check_call([sys.executable, "scripts/install_sources_a.py"])
subprocess.check_call([sys.executable, "scripts/install_sources_b1.py"])
subprocess.check_call([sys.executable, "scripts/install_sources_b2.py"])
print("All sources installed.")
