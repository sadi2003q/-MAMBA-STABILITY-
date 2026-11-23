"""
Look inside the open-source fluid sloshing simulator (Olucha Delgado, Das, Toth, Preda, 2026, arXiv 2604.12505)
so it can be connected to space pilot C. It only PRINTS the simulator's files, README and function names.

python tools/inspect_fluid_simulator.py \
    --folder /kaggle/working/fuel-sloshing-sph-simulator

Copy the printed output and send it to Claude.
"""

import argparse
import ast
import os
import subprocess

parser = argparse.ArgumentParser(description="Print the fluid simulator's files and functions")
parser.add_argument("--folder", default="/kaggle/working/fuel-sloshing-sph-simulator")
SPH_DIR = parser.parse_args().folder


# Step 1: clone it (Kaggle has internet; switch "Internet on" in the notebook settings)
if not os.path.exists(SPH_DIR):
    result = subprocess.run(["git", "clone", "--depth", "1",
                             "https://gitlab.com/Javi-Olucha/fuel-sloshing-sph-simulator.git", SPH_DIR],
                            capture_output=True, text=True)
    print(result.stdout[-2000:], result.stderr[-2000:])

if os.path.exists(SPH_DIR):
    # Step 2: the folder tree (3 levels)
    print("=" * 70)
    print(" FILES")
    print("=" * 70)
    for root, folders, files in os.walk(SPH_DIR):
        if ".git" in root:
            continue
        depth = root[len(SPH_DIR):].count(os.sep)
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for name in sorted(files):
            print("  " * (depth + 1) + name)

    # Step 3: README and installation files
    for name in ["README.md", "README.rst", "README.txt", "requirements.txt", "pyproject.toml", "setup.py"]:
        path = os.path.join(SPH_DIR, name)
        if os.path.exists(path):
            print("=" * 70)
            print(" " + name)
            print("=" * 70)
            with open(path) as handle:
                lines = handle.read().splitlines()
            print("\n".join(lines[:200]))

    # Step 4: every function and class in every Python file, with its arguments and first docstring line
    print("=" * 70)
    print(" FUNCTIONS AND CLASSES")
    print("=" * 70)
    for root, folders, files in os.walk(SPH_DIR):
        if ".git" in root:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            print("--- " + path[len(SPH_DIR) + 1:])
            try:
                with open(path) as handle:
                    tree = ast.parse(handle.read())
            except Exception as error:
                print("    (could not read:", error, ")")
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if isinstance(node, ast.FunctionDef):
                        arguments = []
                        for argument in node.args.args:
                            arguments.append(argument.arg)
                        header = "def " + node.name + "(" + ", ".join(arguments) + ")"
                    else:
                        header = "class " + node.name
                    docstring = ast.get_docstring(node)
                    first_line = docstring.splitlines()[0] if docstring else ""
                    print("    " + header + "   # " + first_line)
    # Step 5: notebooks or example scripts the authors ship
    for root, folders, files in os.walk(SPH_DIR):
        for name in files:
            if name.endswith(".ipynb") or "example" in name.lower() or "main" in name.lower():
                print("example file:", os.path.join(root, name)[len(SPH_DIR) + 1:])
else:
    print("Could not clone the fluid simulator (is Internet switched on?).")
