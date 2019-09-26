import os
from pathlib import Path

def workdir():
    curd = Path(os.path.dirname(os.path.abspath(__file__)))
    workdir = curd.parent
    return workdir