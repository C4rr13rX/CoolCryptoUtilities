from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, timeout: int = 900) -> tuple[bool, str]:
    try:
        resolved = shutil.which(command[0])
        if resolved:
            command = [resolved, *command[1:]]
        env = os.environ.copy()
        # The host Django site exports its own settings module in some worker
        # processes. Generated manage.py files use setdefault(), so inheriting
        # that value silently validates the wrong project.
        env.pop("DJANGO_SETTINGS_MODULE", None)
        if not env.get("CHROME_BIN"):
            browser_candidates = [
                shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("msedge"),
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
            for candidate in browser_candidates:
                if candidate and Path(candidate).is_file():
                    env["CHROME_BIN"] = str(candidate)
                    break
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        text = (result.stdout + "\n" + result.stderr)[-12000:]
        return result.returncode == 0, text
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def require(root: Path, patterns: list[str]) -> list[str]:
    missing = [pattern for pattern in patterns if not list(root.glob(pattern))]
    return [f"missing required path matching {pattern}" for pattern in missing]


def project_root(root: Path, marker: str) -> Path:
    """Accept the assigned directory or one conventional nested project root."""
    direct = root / marker
    if direct.exists():
        return root
    matches = [path for path in root.glob(f"*/{marker}") if path.is_file()]
    return matches[0].parent if len(matches) == 1 else root


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def csv_contract(path: Path, required: set[str], minimum: int, errors: list[str]) -> list[dict[str, str]]:
    fields, rows = read_csv_rows(path)
    missing = sorted(required - set(fields))
    if missing:
        errors.append(f"{path.name} missing columns: {', '.join(missing)}")
    if len(rows) < minimum:
        errors.append(f"{path.name} has {len(rows)} rows; requires at least {minimum}")
    return rows


def validate(case: str, root: Path) -> dict:
    errors: list[str] = []
    evidence: list[dict] = []
    if case == "django-spectrum-instrument":
        root = project_root(root, "manage.py")
        errors += require(root, ["manage.py", "README*", "**/tests.py", "**/*.html"])
        manage_source = (root / "manage.py").read_text(encoding="utf-8", errors="replace") if (root / "manage.py").exists() else ""
        settings_match = re.search(
            r"DJANGO_SETTINGS_MODULE['\"]?\s*,\s*['\"]([^'\"]+)", manage_source,
        )
        settings_module = settings_match.group(1) if settings_match else ""
        config_dir = root / settings_module.split(".")[0] if settings_module else root
        settings_path = root.joinpath(*settings_module.split(".")).with_suffix(".py") if settings_module else root / "settings.py"
        settings_source = settings_path.read_text(encoding="utf-8", errors="replace") if settings_path.exists() else ""
        active_roots = [config_dir] if config_dir.exists() else []
        for child in root.iterdir() if root.exists() else []:
            if child.is_dir() and re.search(rf"['\"]{re.escape(child.name)}(?:\.[^'\"]+)?['\"]", settings_source):
                active_roots.append(child)
        production_sources: list[str] = []
        for active_root in dict.fromkeys(active_roots):
            for path in active_root.rglob("*.py"):
                if not path.name.startswith("test") and "tests" not in path.parts:
                    production_sources.append(path.read_text(encoding="utf-8", errors="replace"))
        production_text = "\n".join(production_sources)
        if re.search(r"(?:^|\n)\s*(?:import\s+numpy|from\s+numpy\s+import)", production_text):
            errors.append("active numerical implementation imports NumPy; DFT must be dependency-free")
        route_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in root.rglob("urls.py")
        ).lower()
        if not (re.search(r"acqui(?:re|sition)", route_text) and "spectrum" in route_text):
            errors.append("URL configuration lacks acquisition and spectrum endpoints")
        if "dashboard" not in route_text:
            errors.append("URL configuration lacks a dashboard route")
        test_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in root.rglob("*.py") if path.name.startswith("test") or "tests" in path.parts
        )
        for marker, label in (("assertRaises", "invalid input"), ("client", "HTTP endpoints"),
                              ("dashboard", "dashboard rendering")):
            if marker.lower() not in test_text.lower():
                errors.append(f"tests lack explicit {label} coverage")
        for command in ([sys.executable, "manage.py", "check"], [sys.executable, "manage.py", "test"]):
            ok, output = run(command, root)
            if command[-1] == "test":
                discovered = [int(value) for value in re.findall(r"(?:Found|Ran)\s+(\d+)\s+test", output, re.IGNORECASE)]
                if not discovered or max(discovered) < 5:
                    ok = False
                    output += "\nBenchmark rejection: fewer than five acceptance tests were discovered."
            evidence.append({"command": command, "ok": ok, "output": output})
            if not ok:
                errors.append(f"command failed: {' '.join(command)}\n{output[-3000:]}")
    elif case in {"dearpygui-impedance-instrument", "tkinter-pid-instrument"}:
        if not (root / "tests").exists():
            candidates = [path.parent for path in root.glob("*/tests") if path.is_dir()]
            if len(candidates) == 1:
                root = candidates[0]
        errors += require(root, ["README*", "tests/test*.py", "**/*.py"])
        if case == "dearpygui-impedance-instrument":
            errors += require(root, ["requirements.txt"])
            requirements_path = root / "requirements.txt"
            requirements = (
                requirements_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if requirements_path.exists() else []
            )
            dependencies = [line.strip() for line in requirements if line.strip() and not line.lstrip().startswith("#")]
            if not any(re.match(r"dearpygui==\d", line, re.IGNORECASE) for line in dependencies):
                errors.append("requirements.txt lacks an exact DearPyGui version pin")
            unpinned = [line for line in dependencies if "==" not in line]
            if unpinned:
                errors.append(f"requirements.txt contains non-exact dependency pins: {unpinned}")
            gui_sources = []
            for path in root.rglob("*.py"):
                if "tests" in path.parts or path.name == "impedance_core.py":
                    continue
                source = path.read_text(encoding="utf-8", errors="replace")
                if re.search(r"(?:import|from)\s+dearpygui(?:\.dearpygui)?", source):
                    gui_sources.append(path)
            if not gui_sources:
                errors.append("no DearPyGui application source imports dearpygui")
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        ok, output = run(command, root)
        evidence.append({"command": command, "ok": ok, "output": output})
        if not ok:
            errors.append(f"headless tests failed:\n{output[-4000:]}")
        if case == "tkinter-pid-instrument" and (root / "pid_controller.py").exists():
            hidden = """
from pid_controller import PIDController

def make(**kwargs):
    return PIDController(**kwargs)

p = make(Kp=2.0, Ki=0.0, Kd=0.0, setpoint=10.0, output_limits=(-1000.0, 1000.0))
assert abs(p.update(measured_value=8.0, dt=1.0) - 4.0) < 1e-9
i = make(Kp=0.0, Ki=1.0, Kd=0.0, setpoint=100.0, output_limits=(-1000.0, 1000.0))
i.update(measured_value=0.0, dt=1.0)
assert abs(i.update(measured_value=0.0, dt=1.0) - 200.0) < 1e-9
d = make(Kp=0.0, Ki=0.0, Kd=1.0, setpoint=100.0, output_limits=(-1000.0, 1000.0))
d.update(measured_value=0.0, dt=1.0)
assert abs(d.update(measured_value=50.0, dt=1.0) + 50.0) < 1e-9
s = make(Kp=2.0, Ki=0.0, Kd=0.0, setpoint=100.0, output_limits=(0.0, 100.0))
assert s.update(measured_value=0.0, dt=1.0) == 100.0
"""
            hidden_ok, hidden_output = run([sys.executable, "-c", hidden], root)
            evidence.append({"command": [sys.executable, "-c", "<hidden PID invariants>"],
                             "ok": hidden_ok, "output": hidden_output})
            if not hidden_ok:
                errors.append(f"hidden PID invariants failed:\n{hidden_output[-4000:]}")
        if case == "dearpygui-impedance-instrument":
            hidden = r'''
import csv
import importlib.util
import math
import tempfile
from pathlib import Path

root = Path.cwd()
candidates = [p for p in root.rglob("impedance_core.py") if "tests" not in p.parts]
assert candidates, "no GUI-independent impedance_core.py found"
spec = importlib.util.spec_from_file_location("hidden_impedance_core", candidates[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

R, L, C, frequency = 100.0, 0.1, 1e-6, 1000.0
expected = complex(R, 2 * math.pi * frequency * L - 1 / (2 * math.pi * frequency * C))
actual = module.calculate_impedance(R, L, C, frequency)
assert abs(actual - expected) < 1e-9, (actual, expected)
resonance = module.calculate_resonance(R, L, C)
assert abs(resonance - 1 / (2 * math.pi * math.sqrt(L * C))) < 1e-9
frequencies = module.generate_sweep(10.0, 1000.0, 10)
assert len(frequencies) == 10 and all(a < b for a, b in zip(frequencies, frequencies[1:]))
for bad in ((-1.0, L, C, frequency), (R, -L, C, frequency), (R, L, -C, frequency), (R, L, C, -frequency)):
    try:
        module.calculate_impedance(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"non-positive input accepted: {bad}")
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / "sweep.csv"
    module.export_to_csv(target, [(frequency, abs(actual), math.degrees(math.atan2(actual.imag, actual.real)))])
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Frequency (Hz)", "Magnitude (Ω)", "Phase (°)"], rows[0]
    assert len(rows) == 2
'''
            hidden_ok, hidden_output = run([sys.executable, "-c", hidden], root)
            evidence.append({"command": [sys.executable, "-c", "<hidden impedance invariants>"],
                             "ok": hidden_ok, "output": hidden_output})
            if not hidden_ok:
                errors.append(f"hidden impedance invariants failed:\n{hidden_output[-4000:]}")
    elif case == "journalism-pqc-migration":
        errors += require(root, ["article.md", "sources.csv", "claims.csv", "research_notes.md"])
        article = (root / "article.md").read_text(encoding="utf-8", errors="replace") if (root / "article.md").exists() else ""
        notes = (root / "research_notes.md").read_text(encoding="utf-8", errors="replace") if (root / "research_notes.md").exists() else ""
        sources = csv_contract(root / "sources.csv", {
            "source_id", "url", "title", "publisher", "published_at", "accessed_at", "source_type", "primary",
        }, 8, errors)
        claims = csv_contract(root / "claims.csv", {
            "claim_id", "claim", "source_ids", "status", "confidence", "article_section",
        }, 12, errors)
        words = re.findall(r"\b[\w'-]+\b", re.sub(r"https?://\S+", "", article))
        if len(words) < 900:
            errors.append(f"article has {len(words)} substantive words; requires at least 900")
        for marker in ("# ", "dek", "byline", "dateline", "limitations", "corrections"):
            if marker.casefold() not in article.casefold():
                errors.append(f"article lacks journalism marker: {marker.strip()}")
        ids = {row.get("source_id", "").strip() for row in sources}
        domains = set()
        for row in sources:
            url = row.get("url", "").strip()
            match = re.match(r"https?://([^/]+)", url)
            if not match:
                errors.append(f"source {row.get('source_id')} lacks an HTTP(S) URL")
            else:
                domains.add(match.group(1).lower().removeprefix("www."))
        if len(domains) < 4:
            errors.append(f"sources span only {len(domains)} independent domains")
        primary = sum(row.get("primary", "").strip().lower() in {"1", "true", "yes"} for row in sources)
        if primary < 3:
            errors.append(f"only {primary} sources are marked primary")
        for row in claims:
            refs = {value.strip() for value in re.split(r"[;,|]", row.get("source_ids", "")) if value.strip()}
            if not refs or not refs <= ids:
                errors.append(f"claim {row.get('claim_id')} has missing or invalid source IDs")
            if row.get("status", "").strip().lower() in {"unsupported", "unverified", ""}:
                errors.append(f"claim {row.get('claim_id')} is not verified or qualified")
            if refs and not any(ref in article for ref in refs):
                errors.append(f"claim {row.get('claim_id')} sources do not appear in article")
        for marker in ("scope", "search", "conflict", "uncert", "fact-check"):
            if marker not in notes.casefold():
                errors.append(f"research notes lack {marker} documentation")
    elif case == "market-needs-metacognition":
        errors += require(root, ["sources.csv", "observations.csv", "needs.csv", "solutions.csv", "methodology.md", "market_needs.sqlite3"])
        sources = csv_contract(root / "sources.csv", {
            "source_id", "url", "title", "publisher", "source_type", "accessed_at",
        }, 20, errors)
        observations = csv_contract(root / "observations.csv", {
            "observation_id", "source_ids", "actor", "workflow", "offering", "gap", "evidence_type", "confidence",
        }, 15, errors)
        needs = csv_contract(root / "needs.csv", {
            "need_id", "layer", "parent_need_ids", "statement", "evidence_observation_ids", "reasoning", "confidence", "disconfirming_evidence",
        }, 12, errors)
        solutions = csv_contract(root / "solutions.csv", {
            "solution_id", "need_ids", "proposal", "validation_experiment", "metric", "risk", "falsification_criterion",
        }, 4, errors)
        source_ids = {row.get("source_id", "").strip() for row in sources}
        observation_ids = {row.get("observation_id", "").strip() for row in observations}
        need_ids = {row.get("need_id", "").strip() for row in needs}
        domains, source_types = set(), set()
        for row in sources:
            source_types.add(row.get("source_type", "").strip().lower())
            match = re.match(r"https?://([^/]+)", row.get("url", "").strip())
            if match:
                domains.add(match.group(1).lower().removeprefix("www."))
        if len(domains) < 8:
            errors.append(f"market sources span only {len(domains)} domains")
        if len(source_types) < 6:
            errors.append(f"market sources span only {len(source_types)} source types")
        if not any(kind in " ".join(source_types) for kind in ("forum", "social", "message", "review")):
            errors.append("market sources lack direct-user evidence")
        layers: dict[str, int] = {}
        for row in needs:
            layer = row.get("layer", "").strip()
            layers[row.get("need_id", "").strip()] = int(layer) if layer.isdigit() else 0
        if set(layers.values()) < {1, 2, 3, 4}:
            errors.append("needs do not represent every inference layer 1-4")
        for row in observations:
            refs = {x.strip() for x in re.split(r"[;,|]", row.get("source_ids", "")) if x.strip()}
            if not refs or not refs <= source_ids:
                errors.append(f"observation {row.get('observation_id')} has invalid source traceability")
        for row in needs:
            need_id = row.get("need_id", "").strip()
            evidence_refs = {x.strip() for x in re.split(r"[;,|]", row.get("evidence_observation_ids", "")) if x.strip()}
            if not evidence_refs or not evidence_refs <= observation_ids:
                errors.append(f"need {need_id} has invalid observation traceability")
            if layers.get(need_id, 0) > 1:
                parents = {x.strip() for x in re.split(r"[;,|]", row.get("parent_need_ids", "")) if x.strip()}
                if not parents or not parents <= need_ids or any(layers.get(parent, 99) >= layers[need_id] for parent in parents):
                    errors.append(f"higher-layer need {need_id} has invalid lower-layer parents")
        for row in solutions:
            refs = {x.strip() for x in re.split(r"[;,|]", row.get("need_ids", "")) if x.strip()}
            if not refs or not refs <= need_ids:
                errors.append(f"solution {row.get('solution_id')} has invalid need mapping")
        db_path = root / "market_needs.sqlite3"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as connection:
                    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for table, rows in (("sources", sources), ("observations", observations), ("needs", needs), ("solutions", solutions)):
                        if table not in tables:
                            errors.append(f"SQLite database lacks {table} table")
                        else:
                            count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                            if count != len(rows):
                                errors.append(f"SQLite {table} count {count} differs from CSV count {len(rows)}")
            except sqlite3.Error as exc:
                errors.append(f"invalid market SQLite database: {exc}")
        methodology = (root / "methodology.md").read_text(encoding="utf-8", errors="replace").casefold() if (root / "methodology.md").exists() else ""
        for marker in ("sampling", "inference", "duplicate", "uncert", "cannot conclude"):
            if marker not in methodology:
                errors.append(f"methodology lacks {marker} documentation")
    elif case == "threejs-multiscale-physics":
        root = project_root(root, "package.json")
        errors += require(root, [
            "package.json", "tsconfig.json", "README*", "src/physics/**/*.ts",
            "src/rendering/**/*.ts", "src/**/*.test.ts",
        ])
        try:
            package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        except Exception as exc:
            package = {}
            errors.append(f"invalid package.json: {exc}")
        dependencies = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
        if "three" not in dependencies:
            errors.append("package.json lacks Three.js")
        unpinned = [name for name, version in dependencies.items()
                    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", str(version))]
        if unpinned:
            errors.append(f"package dependencies are not exactly pinned: {unpinned}")
        scripts = package.get("scripts") or {}
        for script in ("typecheck", "build", "test"):
            if not scripts.get(script):
                errors.append(f"package scripts lack {script}")
        physics_paths = list(root.glob("src/physics/**/*.ts"))
        physics_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in physics_paths)
        if re.search(r"(?:from\s+['\"]three|document\.|window\.|HTMLElement|WebGL)", physics_text):
            errors.append("physics kernel is coupled to Three.js/DOM/rendering")
        physics_lower = physics_text.lower()
        physics_requirements = {
            "SI dimensional units": ("dimension", "unit"),
            "CODATA constants": ("codata", "gravitational"),
            "gravity": ("gravity", "inverse"),
            "electrostatics": ("coulomb", "charge"),
            "symplectic integration": ("verlet", "symplectic"),
            "collisions": ("collision", "impulse"),
            "validity/error model": ("validity", "error"),
            "hierarchical chunking": ("chunk", "refine", "coarsen"),
            "spatial hierarchy": ("octree", "barnes", "spatial"),
        }
        for label, alternatives in physics_requirements.items():
            if not any(term in physics_lower for term in alternatives):
                errors.append(f"physics kernel lacks {label}")
        rendering_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in root.glob("src/rendering/**/*.ts")
        ).lower()
        for label, alternatives in {
            "instancing": ("instancedmesh", "instance"),
            "LOD": ("lod", "levelofdetail"),
            "frustum culling": ("frustum", "frustumculled"),
            "origin rebasing": ("origin", "rebase"),
            "render budget": ("draw call", "renderbudget"),
        }.items():
            if not any(term in rendering_text for term in alternatives):
                errors.append(f"render adapter lacks {label}")
        test_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in root.glob("src/**/*.test.ts")
        ).lower()
        for marker in ("unit", "inverse-square", "momentum", "energy", "collision", "determin", "coarsen", "tolerance"):
            if marker not in test_text:
                errors.append(f"physics tests lack {marker} evidence")
        readmes = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in root.glob("README*"))
        for marker in ("validity", "error budget", "codata", "i5", "32 gb", "roadmap", "not implemented"):
            if marker not in readmes.lower():
                errors.append(f"README lacks {marker} documentation")
        for command in (["npm", "install", "--no-audit", "--no-fund"],
                        ["npm", "run", "typecheck"], ["npm", "run", "build"], ["npm", "test"]):
            ok, output = run(command, root, timeout=1200)
            evidence.append({"command": command, "ok": ok, "output": output})
            if not ok:
                errors.append(f"command failed: {' '.join(command)}\n{output[-4000:]}")
                break
    elif case == "ionic8-environmental-instrument":
        root = project_root(root, "angular.json")
        errors += require(root, ["package.json", "angular.json", "README*", "src/**/*.ts", "src/**/*.spec.ts"])
        package = {}
        try:
            package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid package.json: {exc}")
        dependencies = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
        ionic_version = str(dependencies.get("@ionic/angular", ""))
        if not re.search(r"(?:\^|~|>=)?8(?:\.|$)", ionic_version):
            errors.append(f"@ionic/angular is not pinned to major 8: {ionic_version!r}")
        unpinned = [name for name, version in dependencies.items()
                    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", str(version))]
        if unpinned:
            errors.append(f"package dependencies are not exactly pinned: {unpinned}")
        node_engine = str((package.get("engines") or {}).get("node", ""))
        if not re.fullmatch(r"v?\d+\.\d+\.\d+", node_engine):
            errors.append(f"Node engine is not exactly pinned: {node_engine!r}")
        test_script = str((package.get("scripts") or {}).get("test", ""))
        if not test_script or not re.search(r"(?:watch[=:]false|--no-watch|runInBand)", test_script, re.IGNORECASE):
            errors.append("package test script is missing or not explicitly non-watch")
        service_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in root.glob("src/**/*.ts")
            if not path.name.endswith(".spec.ts")
        )
        if "Math.random" in service_text:
            errors.append("simulation is nondeterministic because production code uses Math.random")
        spec_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in root.glob("src/**/*.spec.ts")
        ).lower()
        coverage_markers = {
            "calibration": r"calibr",
            "alarm": r"alarm",
            "deterministic repeatability": r"determin|identical|same seed|repeat",
        }
        for label, pattern in coverage_markers.items():
            if not re.search(pattern, spec_text):
                errors.append(f"unit tests lack {label} coverage")
        commands = [["npm", "install", "--no-audit", "--no-fund"], ["npm", "run", "build"]]
        if test_script:
            commands.append(["npm", "test"])
        for command in commands:
            ok, output = run(command, root, timeout=1200)
            evidence.append({"command": command, "ok": ok, "output": output})
            if not ok:
                errors.append(f"command failed: {' '.join(command)}\n{output[-4000:]}")
                break
    elif case == "qt6-oscilloscope-instrument":
        root = project_root(root, "CMakeLists.txt")
        errors += require(root, ["CMakeLists.txt", "README*", "**/*.cpp", "**/test*.cpp"])
        if not (list(root.glob("**/*.h")) or list(root.glob("**/*.hpp"))):
            errors.append("missing C/C++ header files (.h or .hpp)")
        cmake_path = root / "CMakeLists.txt"
        cmake = cmake_path.read_text(encoding="utf-8", errors="replace") if cmake_path.exists() else ""
        readmes = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in root.glob("README*"))
        cpp_sources = {
            path: path.read_text(encoding="utf-8", errors="replace") for path in root.glob("**/*.cpp")
        }
        non_test_text = "\n".join(text for path, text in cpp_sources.items() if "test" not in path.name.lower())
        test_text = "\n".join(text for path, text in cpp_sources.items() if "test" in path.name.lower()).lower()
        if not re.search(r"QtWidgets|QMainWindow|QApplication", non_test_text):
            errors.append("GUI sources do not demonstrate Qt Widgets usage")
        for marker in ("waveform", "trigger", "measurement", "csv"):
            if marker not in test_text:
                errors.append(f"headless C++ tests lack {marker} coverage")
        checks = {
            "C++20": r"CXX_STANDARD\s+20|cxx_std_20",
            "Qt6 Widgets": r"find_package\s*\(\s*Qt6[^)]*Widgets",
            "CTest/testing": r"enable_testing\s*\(|include\s*\(\s*CTest",
            "test target": r"add_(?:test|executable)\s*\([^)]*test",
        }
        for label, pattern in checks.items():
            if not re.search(pattern, cmake, re.IGNORECASE | re.DOTALL):
                errors.append(f"CMake lacks {label}")
        if not re.search(r"not installed|unavailable|not available|could not (?:build|run)", readmes, re.IGNORECASE):
            errors.append("README does not disclose the unavailable Qt/compiler validation limitation")
        evidence.append({"static_cmake_checks": checks, "ok": not errors})
    else:
        errors.append(f"unknown case: {case}")
    return {"case": case, "ok": not errors, "errors": errors, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args(argv)
    result = validate(args.case, Path(args.workdir).resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
