from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class FuncInfo:
    qualname: str         # modulo:Clase.metodo o modulo:funcion
    file: str
    lineno: int
    kind: str             # "function" | "method"


def iter_py_files(root: Path, exclude_dirs: Set[str]) -> List[Path]:
    out = []
    for p in root.rglob("*.py"):
        if any(part in exclude_dirs for part in p.parts):
            continue
        out.append(p)
    return out


class Indexer(ast.NodeVisitor):
    def __init__(self, module: str, file: Path):
        self.module = module
        self.file = str(file)
        self.class_stack: List[str] = []
        self.functions: Dict[str, FuncInfo] = {}

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.class_stack:
            qual = f"{self.module}:{'.'.join(self.class_stack)}.{node.name}"
            kind = "method"
        else:
            qual = f"{self.module}:{node.name}"
            kind = "function"
        self.functions[qual] = FuncInfo(
            qualname=qual, file=self.file, lineno=node.lineno, kind=kind
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # tratar igual que FunctionDef
        self.visit_FunctionDef(node)


class CallExtractor(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.class_stack: List[str] = []
        self.func_stack: List[str] = []
        self.edges: Set[Tuple[str, str]] = set()

    def _current_qual(self) -> str | None:
        if not self.func_stack:
            return None
        func = self.func_stack[-1]
        if self.class_stack:
            return f"{self.module}:{'.'.join(self.class_stack)}.{func}"
        return f"{self.module}:{func}"

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call):
        src = self._current_qual()
        if src is None:
            self.generic_visit(node)
            return

        # Captura simple: llamadas por nombre (foo()) o atributo (obj.foo())
        dst = None
        if isinstance(node.func, ast.Name):
            dst = f"{self.module}:{node.func.id}"
        elif isinstance(node.func, ast.Attribute):
            # ejemplo: self.bar() -> intentamos tomar "bar"
            dst = f"{self.module}:{node.func.attr}"

        if dst:
            self.edges.add((src, dst))

        self.generic_visit(node)


def module_name_from_path(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def build_maps(root_dir: str, exclude: List[str] | None = None):
    root = Path(root_dir).resolve()
    exclude_dirs = set(exclude or {".venv", "venv", "__pycache__", "build", "dist", ".git"})
    py_files = iter_py_files(root, exclude_dirs)

    all_funcs: Dict[str, FuncInfo] = {}
    all_edges: Set[Tuple[str, str]] = set()

    for file in py_files:
        try:
            src = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # si hay algún archivo raro, lo saltamos
            continue

        try:
            tree = ast.parse(src, filename=str(file))
        except SyntaxError:
            continue

        mod = module_name_from_path(root, file)

        idx = Indexer(mod, file)
        idx.visit(tree)
        all_funcs.update(idx.functions)

        ce = CallExtractor(mod)
        ce.visit(tree)
        all_edges |= ce.edges

    # filtra edges a cosas que existan en el índice (si no, quedan como "externos/indeterminados")
    known = set(all_funcs.keys())
    edges_known = sorted([(a, b) for (a, b) in all_edges if a in known and b in known])
    edges_unknown = sorted([(a, b) for (a, b) in all_edges if a in known and b not in known])

    return all_funcs, edges_known, edges_unknown


def write_dot(edges: List[Tuple[str, str]], out_path: Path):
    lines = ["digraph callgraph {"]
    for a, b in edges:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Mapa de funciones (inventario + callgraph simple por AST).")
    ap.add_argument("root", nargs="?", default=".", help="Directorio raíz del proyecto")
    ap.add_argument("--out", default="mapa_funciones", help="Prefijo de salida")
    ap.add_argument("--exclude", nargs="*", default=None, help="Carpetas a excluir")
    args = ap.parse_args()

    funcs, edges_known, edges_unknown = build_maps(args.root, args.exclude)

    out_prefix = Path(args.out)
    out_json = out_prefix.with_suffix(".json")
    out_dot = out_prefix.with_suffix(".dot")

    payload = {
        "functions": [asdict(v) for v in funcs.values()],
        "edges_known": edges_known,
        "edges_unknown": edges_unknown,
        "counts": {
            "functions": len(funcs),
            "edges_known": len(edges_known),
            "edges_unknown": len(edges_unknown),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dot(edges_known, out_dot)

    print("OK")
    print(f"Funciones: {len(funcs)}")
    print(f"Edges conocidos: {len(edges_known)}")
    print(f"Edges desconocidos (llamadas no resueltas): {len(edges_unknown)}")
    print(f"JSON: {out_json}")
    print(f"DOT:  {out_dot}")
    print("Tip: si tienes graphviz -> dot -Tpng mapa_funciones.dot -o mapa_funciones.png")


if __name__ == "__main__":
    main()
