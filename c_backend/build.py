"""Build the optional C force library with an available GCC/Clang compiler."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def build(optimization: str = "-O2") -> Path:

    directory = Path(__file__).resolve().parent
    compiler = next((name for name in ("gcc", "clang") if shutil.which(name)), None)
    if compiler is None:
        raise RuntimeError("No GCC or Clang compiler was found on PATH")
    suffix = ".dll" if sys.platform.startswith("win") else ".so"
    output = directory / f"pairforce{suffix}"
    command = [compiler, "-std=c11", "-shared", optimization, str(directory / "pairforce.c"), "-o", str(output)]
    if not sys.platform.startswith("win"):
        command.append("-fPIC")
    # GCC/Clang keep the square-root function in libm on most platforms, including MinGW.
    command.append("-lm")
    subprocess.run(command, check=True, cwd=directory)
    return output


if __name__ == "__main__":
    print(build())
