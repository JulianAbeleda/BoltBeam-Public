#!/usr/bin/env python3
"""Which tests can observe a change, from first principles: a test can only fail because something it depends on
changed. A pass is a fact about an exact tree, so a tree that already passed on this interpreter is not tested again,
and a new tree is tested only where it differs from the last tree that passed.

  python3 tools/impacted_tests.py [--head REV] [--base REV] [--full]   # select: one line per changed path, then
                                                                       #   "verdict SKIP|NONE|FULL|SELECT", "go yes|no", "test <path>"...
  python3 tools/impacted_tests.py --record [--head REV] [--how WORDS]  # after the selection passed: record the tree

Dependencies are read from the head tree, never guessed: Python imports (ast, relative and package __init__ included),
and every file that names another (a string literal holding its file name, path, dotted module name or "pkg." prefix,
or a directory it globs/walks). Anything the rules below do not map runs the full suite: unknown is not pass. The full
suite still runs at a release candidate (docs/release-runbook.md) and with --full."""
from __future__ import annotations
import argparse, ast, datetime, fnmatch, hashlib, importlib.metadata, json, os, pathlib, platform, re, shutil, subprocess, sys

# Path rules, first match wins (fnmatch: * also crosses "/"). full: the change reaches every test or the gate itself.
# py: follow imports and references. doc: tested only where a file names it. data: as doc, but a file nothing names
# may be read by a computed name, so it runs the full suite. go: the TUI's own `go test ./...` (plus Python readers).
RULES = [("conftest.py", "full"), ("*/conftest.py", "full"), ("pyproject.toml", "full"), ("setup.*", "full"),
         ("pytest.ini", "full"), ("tox.ini", "full"), ("tests/requires.py", "full"), ("sz.py", "full"),
         ("tools/*", "full"), (".githooks/*", "full"), (".gitignore", "full"),
         ("tui/*", "go"), ("boltbeam/*.py", "py"), ("tests/*.py", "py"),
         ("docs/*", "doc"), ("*.md", "doc"), ("LICENSE", "doc"),
         ("boltbeam/*", "data"), ("tests/*", "data"), ("schemas/*", "data"), ("bench/*", "data"), ("gui/*", "data")]
# A conftest configures every test: what it imports (transitively) and the data those modules read reach them all.
# What it only names (skip lists, the apparatus sentinel) matters to it only by existing: counted when added or deleted.
CONFTEST = ("conftest.py", "*/conftest.py")
PY_ROOTS = ("boltbeam/", "tests/", "tools/")
SCAN_CALLS = {"glob", "iglob", "rglob", "iterdir", "listdir", "walk", "scandir"}
PATH_CALLS = SCAN_CALLS | {"join", "Path", "PurePath", "PosixPath", "joinpath", "open", "files"}
# A .py named by a file that runs files by path is run, not read as text: its imports count like the file's own.
RUN_CALLS = {"spec_from_file_location", "run_path", "Popen", "run", "check_output", "check_call", "call", "execv", "execvp"}
MAX_TEXT = 4 << 20
TOKEN = re.compile(r"[A-Za-z0-9_.*/-]+")


def git(*args:str, cwd=None, inp:bytes|None=None) -> bytes:
  return subprocess.run(["git", *args], cwd=cwd, input=inp, capture_output=True, check=True).stdout

def rule(path:str) -> str:
  return next((kind for pat, kind in RULES if fnmatch.fnmatchcase(path, pat)), "full")

def is_test(path:str) -> bool:
  name = path.rsplit("/", 1)[-1]
  return path.startswith("tests/") and name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def read_tree(rev:str, cwd) -> dict[str, bytes|None]:
  """Every path in REV; the text of the ones that can name another file (None for binary or oversized)."""
  rows = [r.split(b"\t", 1) for r in git("ls-tree", "-r", "-l", "-z", "--full-tree", rev, cwd=cwd).split(b"\0") if r]
  meta = [(p.decode(), m.split()) for m, p in rows]
  want = [(p, f[2]) for p, f in meta if f[1] == b"blob" and f[3] != b"-" and int(f[3]) <= MAX_TEXT]
  out:dict[str, bytes|None] = {p: None for p, _ in meta}
  raw = git("cat-file", "--batch", cwd=cwd, inp=b"".join(s + b"\n" for _, s in want))
  i = 0
  for p, _ in want:
    head = raw[i:raw.index(b"\n", i)].split(); i += len(b" ".join(head)) + 1
    body = raw[i:i + int(head[2])]; i += int(head[2]) + 1
    out[p] = None if b"\0" in body[:8192] else body
  return out


def dotted(path:str) -> list[str]:
  """The names a Python file is imported by: its dotted path, and for tests the names pytest's rootdir/prepend import
  mode makes importable (tests/forkkit.py is also `forkkit`)."""
  mod = path[:-3].replace("/", ".")
  mod = mod[:-9] if mod.endswith(".__init__") else mod
  names = [mod]
  if mod.startswith("tests."):
    parts = mod.split(".")[1:]
    names += [".".join(parts[i:]) for i in range(len(parts))]
  return [n for n in names if n]


class PyFacts(ast.NodeVisitor):
  def __init__(self, path:str):
    self.pkg = path[:-3].replace("/", ".").rsplit(".", 1)[0] if not path.endswith("__init__.py") else path[:-12].replace("/", ".")
    self.imports:set[str] = set(); self.literals:set[str] = set(); self.pathlits:set[str] = set(); self.scans = False; self.runs = False; self.walks:set[str] = set()
    self.walkargs:set[str] = set()  # a walk's own pattern: what it reads is in walks (none, when rooted in tmp_path)
    self.tmp:set[str] = {"tmp_path", "tmpdir"}
  def _add(self, mod:str):
    parts = mod.split(".")
    self.imports.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
  def visit_Import(self, node):
    for a in node.names: self._add(a.name)
  def visit_ImportFrom(self, node):
    base = self.pkg.split(".")[:len(self.pkg.split(".")) - node.level + 1] if node.level else []
    mod = ".".join([*base, *([node.module] if node.module else [])])
    if mod: self._add(mod)
    for a in node.names: self._add(f"{mod}.{a.name}" if mod else a.name)
  def visit_Call(self, node):
    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
    self.scans |= name in SCAN_CALLS; self.runs |= name in RUN_CALLS
    arg = node.args[0] if node.args else None
    if name in SCAN_CALLS and isinstance(arg, ast.Constant) and isinstance(arg.value, str): self.walkargs.add(arg.value)
    if name in SCAN_CALLS and not self._in_tmp(node):  # the pattern it walks with; none (iterdir, walk) matches everything
      self.walks.add(arg.value.rsplit("/", 1)[-1] if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and name in ("glob", "iglob", "rglob") else "*")
    if name in PATH_CALLS: self.pathlits.update(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
    if name == "format" and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
      self.literals.add(re.sub(r"\{[^}]*\}", "*", node.func.value.value))  # "dir/{}.json".format(x)
    self.generic_visit(node)
  def _root(self, e):
    while isinstance(e, (ast.Attribute, ast.Call, ast.BinOp, ast.Subscript)):
      e = e.left if isinstance(e, ast.BinOp) else e.func if isinstance(e, ast.Call) else e.value
    return e.id if isinstance(e, ast.Name) else None
  def _in_tmp(self, node) -> bool:
    """A walk rooted in pytest's fresh tmp_path (directly, or via a name this function only ever assigns from it)
    reads nothing of the repo."""
    recv = node.func.value if isinstance(node.func, ast.Attribute) else (node.args[0] if node.args else None)
    return recv is not None and self._root(recv) in self.tmp
  def visit_BinOp(self, node):  # pathlib's `root / "data"`: the operand is a path part
    if isinstance(node.op, ast.Div):
      self.pathlits.update(x.value for x in (node.left, node.right) if isinstance(x, ast.Constant) and isinstance(x.value, str))
    if isinstance(node.op, ast.Add):  # "prefix" + x, x + ".json": a name built from pieces is a glob over its holes
      lit = [x.value if isinstance(x, ast.Constant) and isinstance(x.value, str) else None for x in (node.left, node.right)]
      if any(lit): self.literals.add("".join(v if v is not None else "*" for v in lit))
    if isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):  # "d/%s.json" % x
      self.literals.add(re.sub(r"%[-#0 +]*\d*(?:\.\d+)?[a-z]", "*", node.left.value))
    self.generic_visit(node)
  def visit_JoinedStr(self, node):  # f"{stem}.schema.json" -> "*.schema.json"
    self.literals.add("".join(v.value if isinstance(v, ast.Constant) else "*" for v in node.values))
    self.generic_visit(node)
  def visit_Constant(self, node):
    if isinstance(node.value, str) and len(node.value) < 4096:
      self.literals.add(node.value); self.literals.update(node.value.replace(":", " ").replace(",", " ").split())
  def visit_body(self, node):  # docstrings name files for the reader, not for the program
    saved = self.tmp
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      src:dict[str, set] = {}
      for a in ast.walk(node):
        for t in (a.targets if isinstance(a, ast.Assign) else [a.target] if isinstance(a, (ast.For, ast.AnnAssign, ast.NamedExpr)) else []):
          for n in ast.walk(t):
            if isinstance(n, ast.Name): src.setdefault(n.id, set()).add(self._root(a.iter if isinstance(a, ast.For) else a.value))
      self.tmp = {"tmp_path", "tmpdir"}
      while True:
        more = {n for n, roots in src.items() if roots <= self.tmp and n not in self.tmp}
        if not more: break
        self.tmp |= more
      params = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg]
      self.tmp -= {a.arg for a in params if a is not None} - {"tmp_path", "tmpdir"}
    self._visit_body(node); self.tmp = saved
  def _visit_body(self, node):
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant): body = body[1:]
    for n in body: self.visit(n)
    for f in node._fields:
      v = getattr(node, f) if f != "body" else []
      for x in v if isinstance(v, list) else [v]:
        if isinstance(x, ast.AST): self.visit(x)
  visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_body


class Graph:
  """Edges point from a file to what it depends on, built once over the head tree, in two kinds:
  depends[f]: f runs it. Imports, modules loaded by name, and the data files f reads. Followed transitively: whatever
    imports f runs it too.
  checks[f]: f reads it as source text (a test opening tools/x.py), or matches it by a pattern (a walk, a name built
    from pieces: f"{stem}.json"). Followed one hop: the check and the files importing it directly observe it; a module
    further out imports the checker for another name and need not run the scan."""
  def __init__(self, tree:dict[str, bytes|None], extra:list[str]):
    self.paths = sorted(set(tree) | set(extra))
    self.modules:dict[str, set[str]] = {}
    self.base:dict[str, set[str]] = {}; self.dirs:dict[str, set[str]] = {}; self.stems:dict[str, set[str]] = {}; self.aliased:set[str] = set()
    for p in self.paths:
      if p.endswith(".py") and p.startswith(PY_ROOTS) or ("/" not in p and p.endswith(".py")):
        for n in dotted(p): self.modules.setdefault(n, set()).add(p)
        if p.startswith("tests/"): self.aliased.add(p)
      self.base.setdefault(p.rsplit("/", 1)[-1], set()).add(p)
      if p.endswith(".py"):
        self.stems.setdefault(p.rsplit("/", 1)[-1][:-3], set()).add(p)
        if p.endswith("/__init__.py"): self.stems.setdefault(p.split("/")[-2], set()).add(p)
      seg = p.split("/")[:-1]
      for i in range(len(seg)):
        for j in range(i + 1, len(seg) + 1): self.dirs.setdefault("/".join(seg[i:j]), set()).add(p)
    self.conf_names:set[str] = set(); self.conf_paths:set[str] = set(); self.imported:dict[str, set[str]] = {}
    self.depends:dict[str, set[str]] = {}; self.checks:dict[str, set[str]] = {}; self.globs:dict[str, set[str]] = {}
    for p, body in tree.items():
      if body is not None: self._edges(p, body)
    self.configures = self._conftest_reach()
    self.readers:dict[str, set[str]] = {}; self.checkers:dict[str, set[str]] = {}
    for edges, rev in ((self.depends, self.readers), (self.checks, self.checkers)):
      for f, deps in edges.items():
        for d in deps - {f}: rev.setdefault(d, set()).add(f)

  def _edges(self, path:str, body:bytes):
    text, code = body.decode("utf-8", "replace"), path.endswith(".py")
    if code:
      facts = PyFacts(path)
      try: facts.visit(ast.parse(text))
      except (SyntaxError, ValueError, RecursionError): self.depends[path] = set(self.paths); return  # unreadable: reads all
      imports, literals, walkargs, runs_py, walked = facts.imports, facts.literals, facts.walkargs, facts.runs, facts.walks
    elif rule(path) == "doc": return  # nothing executes a doc: a test that reads one reads its text, not what it names
    else:  # data and Go: a module name in it is one a program loads by that name (an adapter row, `-m boltbeam`)
      imports, literals, walkargs, runs_py, walked = set(), {t for t in TOKEN.findall(text) if "." in t or t == "-m"}, set(), False, set()
    # an import that is no dotted module of the repo may be a sibling put on sys.path at run time (`import build_jobs`
    # beside a runtime script): it reaches every repo file of that name
    unknown = {n.split(".")[0] for n in imports if n.split(".")[0] not in self.modules}
    runs = self.imported[path] = {f for n in imports for f in self.modules.get(n, ())} | {f for n in unknown for f in self.stems.get(n, ())}
    runs = set(runs)
    has_m, named, holes, dirnamed = "-m" in literals, set(), set(), set()
    for t in literals:
      t = t.strip().strip("/").removeprefix("./")
      if not t or len(t) > 512: continue
      mod = t.split(":")[0]
      # a dotted name in a string is a module loaded by that name; a bare word is one only if it is a real top-level
      # module (`-m boltbeam`), not the import alias pytest gives a test helper ("requires" is also a dict key)
      if mod in self.modules and ("." in mod or not self.modules[mod] <= self.aliased):
        runs |= self.modules[mod]
        if has_m: runs |= self.modules.get(mod + ".__main__", set())
      if mod.endswith(".") and mod.count(".") > 1: runs |= {f for n, fs in self.modules.items() if n.startswith(mod) for f in fs}
      if not code: continue
      named |= self.base.get(t, set())
      if "/" in t: named |= {f for f in self.base.get(t.rsplit("/", 1)[-1], ()) if f.endswith("/" + t) or f == t}
      if "*" in t and t not in walkargs and len(t.replace("*", "")) >= 3 and ("." in t or "/" in t):  # a name with holes
        holes |= self._glob(t)
      if "/" in t and t in self.dirs: dirnamed |= self.dirs[t]  # a directory it names by path ("support/offline")
    if walked:  # a walk whose root is a variable may walk anything: every file its patterns match is read
      for w in walked: holes |= self._glob(w)
    # exact names are followed transitively; a pattern (a walk, a name with holes) is followed one hop, like a source check
    self.checks[path] = holes | dirnamed
    if any(fnmatch.fnmatchcase(path, c) for c in CONFTEST):
      self.conf_paths |= named | dirnamed | holes; self.conf_names |= named | (runs - self.imported[path])
      runs = self.imported[path]; self.checks.pop(path, None)
    else:
      runs |= {f for f in named if not f.endswith(".py") or runs_py}
      self.checks.setdefault(path, set()).update(f for f in named if f.endswith(".py") and not runs_py)
    self.depends[path] = runs

  def _glob(self, pat:str) -> set[str]:
    if pat not in self.globs:
      self.globs[pat] = {f for f in self.paths if fnmatch.fnmatchcase(f, "*" + pat) or fnmatch.fnmatchcase(f.rsplit("/", 1)[-1], pat)}
    return self.globs[pat]

  def _conftest_reach(self) -> set[str]:
    """What every test runs through: the conftests, what they import (transitively), and the data those read. A module
    one of them only loads by name when a test calls it is that test's dependency, followed like any other."""
    todo = [p for p in self.depends if any(fnmatch.fnmatchcase(p, c) for c in CONFTEST)]
    seen = set(todo) | {p for p in self.conf_paths if not is_test(p)}  # a path it sets up (PYTHONPATH, a sentinel): the
                                                                       # file itself, not what it imports
    while todo:
      f = todo.pop()
      for d in self.imported.get(f, set()) | {d for d in self.depends.get(f, ()) if not d.endswith(".py")}:
        if d not in seen: seen.add(d); todo.append(d)
    return seen

  def reach(self, path:str) -> set[str]:
    """Every file that runs PATH, plus each source check that reads it and the files importing that check directly."""
    seen, todo = {path}, [path]
    while todo:
      for r in self.readers.get(todo.pop(), ()):
        if r not in seen: seen.add(r); todo.append(r)
    for c in self.checkers.get(path, ()):
      seen.add(c); seen |= {r for r in self.readers.get(c, ()) if c in self.imported.get(r, ())}
    return seen - {path}


def select(changed:list[tuple[str, str]], tree:dict[str, bytes|None]) -> tuple[str, bool, list[str], list[str]]:
  """changed: (git status letter, path). Returns (verdict NONE|FULL|SELECT, go, tests, one reason line per path)."""
  graph, tests, go, full, lines = Graph(tree, [p for _, p in changed]), set(), False, False, []
  for status, p in changed:
    kind, why, reached = rule(p), "", set()
    if kind != "full":
      reached = graph.reach(p)
      hit = sorted(r for r in reached | {p} if is_test(r) and r in tree)
      conf = p in graph.configures
      go |= kind == "go" or any(r.startswith("tui/") for r in reached)
    if kind == "full": full, why = True, "FULL: not mapped to a narrower set (it configures or gates the suite)"
    elif conf: full, why = True, "FULL: a conftest imports or reads it, so it reaches every test"
    elif status in "AD" and p in graph.conf_names: full, why = True, "FULL: a conftest names it, and it was added or deleted"
    elif kind in ("data", "py") and not reached and status != "D" and not is_test(p):
      full, why = True, "FULL: no file imports or names it, so it may be loaded by a computed name"
    elif hit: tests.update(hit); why = f"{len(hit)} tests: " + ", ".join(h.removeprefix("tests/") for h in hit[:4]) + (" ..." if len(hit) > 4 else "")
    elif kind == "go": why = "go test ./... in tui"
    else: why = f"none: {'deleted; ' if status == 'D' else ''}" + (f"read by {len(reached)} files, no test reaches them" if reached else "no file names it")
    lines.append(f"changed {p}: {why}")
  if full: return "FULL", True, [], lines
  return ("SELECT" if tests else "NONE"), go, sorted(tests), lines


def _out(*cmd:str) -> str:
  try: return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or "none"
  except (OSError, subprocess.SubprocessError): return "none"

def env_key(repo:str=".") -> str:
  """A pass is a fact about a tree on an interpreter, its installed packages, and the machine state that decides which
  tests skip (tests/requires.py): the GPUs the driver sees now, the engine builds, Go, the tools probed, BOLTBEAM_*."""
  dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"])
  builds = pathlib.Path(os.environ.get("BOLTBEAM_TEST_MACHINE_BUILDS") or "~/.boltbeam/builds.json").expanduser()
  try: builds_sha = hashlib.sha256(builds.read_bytes()).hexdigest()
  except OSError: builds_sha = "none"
  # every program a test could find on PATH, by literal or computed name (shutil.which(f"python{v}"), an engine finder):
  # PATH itself and each executable on it, with its mtime so an upgrade in place counts
  progs:list[str] = [os.environ.get("PATH", "")]
  for d in os.environ.get("PATH", "").split(os.pathsep):
    try:
      for e in sorted(os.scandir(d), key=lambda e: e.name):
        try:
          if e.is_file() and os.access(e.path, os.X_OK): progs.append(f"{d}/{e.name}:{e.stat().st_mtime_ns}")
        except OSError: pass
    except OSError: pass
  blob = "\n".join([os.path.realpath(sys.executable), sys.version, platform.node(), *dists, _out("nvidia-smi", "-L"),
                    builds_sha, _out("go", "version"), _out("uv", "python", "list", "--only-installed"), *progs,
                    *sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith("BOLTBEAM_") and k != "BOLTBEAM_SUITE_CACHE")])
  return hashlib.sha256(blob.encode()).hexdigest()[:16]

def cache_path() -> pathlib.Path:
  return pathlib.Path(os.environ.get("BOLTBEAM_SUITE_CACHE") or "~/.cache/boltbeam/suite-pass.json").expanduser()

def load_cache() -> dict:
  try: return json.loads(cache_path().read_text())
  except (OSError, ValueError): return {}

def tree_of(rev:str, cwd) -> str:
  return git("rev-parse", f"{rev}^{{tree}}", cwd=cwd).decode().strip()

def passed_base(head:str, cwd, passed:dict) -> tuple[str, str]|None:
  """The nearest ancestor (first parent first, then any) whose tree passed here, as (commit, tree)."""
  for args in (("--first-parent", "-n200"), ("-n400",)):
    for line in git("log", *args, "--format=%H %T", head, cwd=cwd).decode().splitlines():
      if (r := line.split())[1] in passed: return r[0], r[1]
  return None


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--head", default="HEAD"); ap.add_argument("--base", help="compare with this revision instead of the last passed tree")
  ap.add_argument("--full", action="store_true"); ap.add_argument("--record", action="store_true")
  ap.add_argument("--how", default="selected", help="--record: how the tree passed (full, selected, none)")
  ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[1]))
  a = ap.parse_args(argv)
  cwd, key, cache = a.repo, env_key(a.repo), load_cache()
  head_tree, passed = tree_of(a.head, a.repo), cache.get(key, {})
  if a.record:
    passed[head_tree] = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "how": a.how,
                         "commit": git("rev-parse", a.head, cwd=cwd).decode().strip(), "python": sys.version.split()[0]}
    cache[key] = passed
    cache_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path().with_suffix(".tmp"); tmp.write_text(json.dumps(cache, indent=1, sort_keys=True)); tmp.replace(cache_path())
    print(f"recorded tree {head_tree[:12]} passed ({a.how}) for interpreter {key}"); return 0
  if a.full: print("verdict FULL\ngo yes\nreason --full"); return 0
  if not a.base and head_tree in passed:
    print(f"verdict SKIP\ngo no\nreason tree {head_tree[:12]} already passed here ({passed[head_tree]['how']}, {passed[head_tree]['at']})"); return 0
  base = (a.base, tree_of(a.base, cwd)) if a.base else passed_base(a.head, cwd, passed)
  if base is None: print("verdict FULL\ngo yes\nreason no tree among this branch's last 400 ancestors passed on this interpreter"); return 0
  raw = git("diff", "--no-renames", "--name-status", "-z", base[1], head_tree, cwd=cwd).decode().split("\0")[:-1]
  verdict, go, tests, lines = select(list(zip(raw[::2], raw[1::2])), read_tree(a.head, cwd))
  print(f"base {base[0][:12]} tree {base[1][:12]}", *lines, f"verdict {verdict}", f"go {'yes' if go else 'no'}",
        *(f"test {t}" for t in tests), sep="\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
