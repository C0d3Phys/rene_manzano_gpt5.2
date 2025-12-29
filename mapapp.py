from __future__ import annotations

import ast
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


# -----------------------------
# Datos
# -----------------------------
@dataclass
class FuncInfo:
    qualname: str
    file: str
    lineno: int
    kind: str  # function|method


# -----------------------------
# Utilidades de resolución
# -----------------------------
def guess_project_root(entry_file: Path) -> Path:
    # Si tienes /src, úsalo como referencia; si no, usa el padre del entry
    root = entry_file.parent
    # sube hasta que no haya más carpetas padres (en proyecto típico, root = repo root)
    return root.resolve()


def module_name_from_file(root: Path, file: Path) -> str:
    rel = file.resolve().relative_to(root.resolve()).with_suffix("")
    return ".".join(rel.parts)


def resolve_module_to_file(root: Path, module: str) -> Optional[Path]:
    """
    Resuelve "paquete.mod" a:
      - root/paquete/mod.py
      - root/paquete/mod/__init__.py
    """
    mod_path = root.joinpath(*module.split("."))
    cand1 = mod_path.with_suffix(".py")
    if cand1.exists():
        return cand1
    cand2 = mod_path / "__init__.py"
    if cand2.exists():
        return cand2
    return None


def is_local_module(root: Path, module: str) -> bool:
    return resolve_module_to_file(root, module) is not None


# -----------------------------
# Indexador de funciones/métodos
# -----------------------------
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
        self.functions[qual] = FuncInfo(qual, self.file, node.lineno, kind)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)


# -----------------------------
# Extractor simple de llamadas (aprox)
# -----------------------------
class CallExtractor(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.class_stack: List[str] = []
        self.func_stack: List[str] = []
        self.edges: Set[Tuple[str, str]] = set()

    def _current_qual(self) -> Optional[str]:
        if not self.func_stack:
            return None
        fn = self.func_stack[-1]
        if self.class_stack:
            return f"{self.module}:{'.'.join(self.class_stack)}.{fn}"
        return f"{self.module}:{fn}"

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

        dst = None
        if isinstance(node.func, ast.Name):
            # foo()
            dst = f"{self.module}:{node.func.id}"
        elif isinstance(node.func, ast.Attribute):
            # obj.foo() => capturamos "foo" (sin resolver obj)
            dst = f"{self.module}:{node.func.attr}"

        if dst:
            self.edges.add((src, dst))

        self.generic_visit(node)


# -----------------------------
# Parse de imports
# -----------------------------
def extract_imported_modules(tree: ast.AST, current_module: str) -> Set[str]:
    """
    Devuelve módulos importados (strings) sin intentar resolver alias.
    Maneja imports relativos de forma básica.
    """
    out: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            level = getattr(node, "level", 0) or 0

            # Construye nombre absoluto aproximado si es relativo
            if level > 0:
                parts = current_module.split(".")
                base = parts[:-level] if level <= len(parts) else []
                abs_mod = ".".join(base + node.module.split("."))
                out.add(abs_mod)
            else:
                out.add(node.module)

    return out


# -----------------------------
# Recorrido reachable
# -----------------------------
def iter_reachable_files(
    root: Path,
    entry_file: Path,
    exclude_dirs: Set[str],
) -> List[Path]:
    visited: Set[Path] = set()
    queue: List[Path] = [entry_file.resolve()]
    reachable: List[Path] = []

    while queue:
        f = queue.pop()
        if f in visited:
            continue
        visited.add(f)

        # excluir por carpeta
        if any(part in exclude_dirs for part in f.parts):
            continue
        if not f.exists() or f.suffix != ".py":
            continue

        reachable.append(f)

        try:
            src = f.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(f))
        except Exception:
            continue

        cur_mod = module_name_from_file(root, f)
        imported = extract_imported_modules(tree, cur_mod)

        for mod in imported:
            # Solo seguir módulos locales
            if is_local_module(root, mod):
                nf = resolve_module_to_file(root, mod)
                if nf is not None and nf not in visited:
                    queue.append(nf)

    return reachable


def write_dot(edges: List[Tuple[str, str]], out_path: Path) -> None:
    lines = ["digraph callgraph {", "  rankdir=LR;"]
    for a, b in edges:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Mapa estático reachable desde main.py")
    ap.add_argument("--entry", default="main.py", help="Archivo entrypoint de la app")
    ap.add_argument("--root", default=".", help="Directorio raíz del proyecto")
    ap.add_argument("--out", default="mapa_main", help="Prefijo de salida (json/dot)")
    ap.add_argument("--exclude", nargs="*", default=["legacy", ".venv", "venv", "__pycache__", ".git", "build", "dist"],
                    help="Carpetas a excluir")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    entry = (root / args.entry).resolve()

    exclude_dirs = set(args.exclude)

    files = iter_reachable_files(root, entry, exclude_dirs)

    all_funcs: Dict[str, FuncInfo] = {}
    all_edges: Set[Tuple[str, str]] = set()

    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(f))
        except Exception:
            continue

        mod = module_name_from_file(root, f)

        idx = Indexer(mod, f)
        idx.visit(tree)
        all_funcs.update(idx.functions)

        ce = CallExtractor(mod)
        ce.visit(tree)
        all_edges |= ce.edges

    known = set(all_funcs.keys())
    edges_known = sorted([(a, b) for (a, b) in all_edges if a in known and b in known])
    edges_unknown = sorted([(a, b) for (a, b) in all_edges if a in known and b not in known])

    out_prefix = Path(args.out)
    out_json = out_prefix.with_suffix(".json")
    out_dot = out_prefix.with_suffix(".dot")

    payload = {
        "entry": str(entry),
        "files_reachable": [str(p) for p in files],
        "functions": [asdict(v) for v in all_funcs.values()],
        "edges_known": edges_known,
        "edges_unknown": edges_unknown,
        "counts": {
            "files_reachable": len(files),
            "functions": len(all_funcs),
            "edges_known": len(edges_known),
            "edges_unknown": len(edges_unknown),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dot(edges_known, out_dot)

    print("OK")
    print(f"Reachable files: {len(files)}")
    print(f"Funciones: {len(all_funcs)}")
    print(f"Edges conocidos: {len(edges_known)}")
    print(f"Edges desconocidos: {len(edges_unknown)}")
    print(f"JSON: {out_json}")
    print(f"DOT : {out_dot}")
    print("Si tienes graphviz: dot -Tpng mapa_main.dot -o mapa_main.png")


if __name__ == "__main__":
    main()
