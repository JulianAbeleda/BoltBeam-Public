"""Upgrade engines: upstream release parsing (releases only), version order, the current/pending/previous/older states,
the deletion rules, TensorRT-LLM's fit from its published metadata, and the command and pipeline wiring. No network, no
GPU, no install: every reader is faked."""
import argparse
import io
import json
import os
import pathlib

import pytest

from boltbeam.collectors import engine_builds as eb
from boltbeam.target import vendors
from boltbeam.workflow import engine_update, screen
from boltbeam.workflow import pipeline as pp

CUDA = vendors.for_backend("CUDA")
TRT_INDEX = ('<a href="x">tensorrt_llm-1.2.1-cp310-cp310-linux_x86_64.whl</a><br>'
             '<a href="x">tensorrt_llm-1.2.1-cp312-cp312-linux_x86_64.whl</a><br>'
             '<a href="x">tensorrt_llm-1.2.1-cp312-cp312-linux_aarch64.whl</a><br>'
             '<a href="x">tensorrt_llm-1.3.0rc29-cp312-cp312-linux_x86_64.whl</a><br>'
             '<a href="x">tensorrt_llm-1.4.0rc3-cp310-cp310-manylinux_2_28_x86_64.whl</a><br>'
             '<a href="x">tensorrt_llm-0.10.0.dev2024041600-cp310-cp310-linux_x86_64.whl</a><br>'
             '<a href="x">tensorrt_llm-1.2.0.tar.gz</a><br>')
# tensorrt-llm 1.2.1's own metadata on PyPI, 2026-11-08 (requires_dist cut to what the fit reads)
TRT_META = {"info": {"requires_python": "<4,>=3.10", "requires_dist": [
  "cuda-python>=13", "torch<=2.10.0a0,>=2.9.1", "torchvision", "torch-c-dlpack-ext==0.1.3", "nvidia-modelopt[torch]~=0.37.0"]}}


@pytest.fixture(autouse=True)
def machine(tmp_path, monkeypatch):
  monkeypatch.setenv("BOLTBEAM_ENGINES_FILE", str(tmp_path / "engines.json"))
  for var in ("BOLTBEAM_LLAMA_BENCH", "BOLTBEAM_LLAMA_BATCHED_BENCH", "BOLTBEAM_GGML_CUDA_SRC", "BOLTBEAM_VLLM_PYTHON"):
    monkeypatch.delenv(var, raising=False)
  return tmp_path


def build(version, folder="/b"):
  return {"version": version, "folder": f"{folder}/{version}", "paths": {"BOLTBEAM_VLLM_PYTHON": f"{folder}/{version}/venv/bin/python"}}


def pypi(*versions, yanked=()):
  return {"releases": {v: [{"yanked": v in yanked}] for v in versions}}


@pytest.mark.parametrize("lo,hi", [
  ("0.31.0", "0.33.0"), ("0.9.2", "0.10.0"), ("1.3.0rc29", "1.3.0"), ("0.9.0", "0.9.0.1"), ("0.8.5", "0.8.5.post1"),
  ("1.4.0rc2.post1", "1.4.0rc3"), ("0.10.0.dev2024041600", "0.10.0rc1"), ("0.40.2", "v0.40.15"), ("v0.8.0", "v0.9.0")])
def test_release_versions_order(lo, hi):
  assert eb.version_key(lo) < eb.version_key(hi)


def test_a_build_that_is_not_a_release_has_no_order():
  for v in ("b12211", "0.6.0-dev", "0.11.0+cu130", "master", None):
    assert eb.version_key(v) is None
  assert eb.newer("0.33.0", None) is None  # unknown is not an answer, not "older"


def test_releases_only_per_engine():
  latest = {"tag_name": "v0.9.0", "target_commitish": "cd959bae99ff4bed22bef2604f45241c09bbb8c8"}
  assert eb.newest(eb.row("llama.cpp"), latest) == "0.9.0"  # the v0.x line
  assert eb.newest(eb.row("llama.cpp"), [{"tag_name": "b12211", "prerelease": True}]) is None  # nightlies never
  assert eb.newest(eb.row("llama.cpp"), {"tag_name": "v1.0.0", "prerelease": True}) is None
  assert eb.newest(eb.row("ollama"), {"tag_name": "v0.40.15"}) == "0.40.15"
  assert eb.newest(eb.row("vllm"), pypi("0.31.0", "0.33.0", "0.34.0rc1", "0.34.0.dev0", "0.35.0", yanked=("0.35.0",))) == "0.33.0"
  assert eb.newest(eb.row("tensorrt-llm"), TRT_INDEX) == "1.2.1"
  assert eb.check("llama.cpp", lambda url: latest)["commit"] == latest["target_commitish"]


def test_a_failed_look_is_could_not_check_never_up_to_date():
  def offline(url):
    raise TimeoutError("timed out")
  got = eb.check("vllm", offline)
  assert got["latest"] is None and "TimeoutError" in got["error"]
  assert eb.check("llama.cpp", lambda url: {"tag_name": "b12211", "prerelease": True})["error"] == "upstream named no stable release"


def test_first_build_is_current_then_newer_is_pending_until_compared():
  state = {"schema": eb.SCHEMA, "engines": {}}
  assert eb.installed(state, "vllm", build("0.31.0")) == "current"
  assert eb.installed(state, "vllm", build("0.33.0")) == "pending"
  with pytest.raises(ValueError, match="not newer"):
    eb.installed(state, "vllm", build("0.30.0"))
  e = eb.compared(state, "vllm", {"before_run": "r-was-0.31.0"})
  assert e["current"]["version"] == "0.33.0" and e["previous"]["version"] == "0.31.0" and e["pending"] is None
  assert e["comparisons"][0] | {"at": None} == {"before_run": "r-was-0.31.0", "before": "0.31.0", "after": "0.33.0", "at": None}
  with pytest.raises(ValueError, match="no pending build"):
    eb.compared(state, "vllm", {})


def test_after_a_comparison_only_current_and_previous_stay_and_never_the_users_build(tmp_path):
  root = tmp_path / "engines"
  def made(v):
    (root / "vllm" / v).mkdir(parents=True)
    (root / "vllm" / v / eb.MARK).write_text("vllm\n")
    return {**build(v), "folder": str(root / "vllm" / v)}
  state = {"schema": eb.SCHEMA, "engines": {}, "root": str(root)}
  eb.installed(state, "vllm", {**build("0.30.0"), "folder": None})  # found here at first run: the user's
  for v in ("0.31.0", "0.33.0", "0.34.0"):
    eb.installed(state, "vllm", made(v))
    eb.compared(state, "vllm", {})
  gone = engine_update.prune(state, "vllm")
  e = eb.entry(state, "vllm")
  assert [g["version"] for g in gone] == ["0.31.0"] and not (root / "vllm" / "0.31.0").exists()
  assert (root / "vllm" / "0.33.0").exists() and (root / "vllm" / "0.34.0").exists()  # previous and current
  assert e["older"] == [{**build("0.30.0"), "folder": None, "kept": "found on this machine, not installed by BoltBeam"}]
  assert e["deleted"][0]["version"] == "0.31.0"


def test_a_folder_without_boltbeams_mark_or_outside_its_root_is_never_deleted(tmp_path):
  (tmp_path / "engines" / "x").mkdir(parents=True)
  assert "not a folder BoltBeam installed" in eb.deletable({"folder": str(tmp_path / "engines" / "x")}, str(tmp_path / "engines"))
  (tmp_path / "elsewhere").mkdir()
  (tmp_path / "elsewhere" / eb.MARK).write_text("")
  assert "not a folder BoltBeam installed" in eb.deletable({"folder": str(tmp_path / "elsewhere")}, str(tmp_path / "engines"))


def test_the_users_env_var_wins_and_selection_is_undone(monkeypatch):
  b = {"paths": {"BOLTBEAM_LLAMA_BENCH": "/new/llama-bench", "BOLTBEAM_GGML_CUDA_SRC": "/new/src"}}
  monkeypatch.setenv("BOLTBEAM_LLAMA_BENCH", "/mine/llama-bench")
  with eb.using(b) as overridden:
    assert overridden == ["BOLTBEAM_LLAMA_BENCH"]
    assert os.environ["BOLTBEAM_LLAMA_BENCH"] == "/mine/llama-bench" and os.environ["BOLTBEAM_GGML_CUDA_SRC"] == "/new/src"
  assert "BOLTBEAM_GGML_CUDA_SRC" not in os.environ


def test_currency_says_latest_at_run_time_or_what_was_newer_or_not_checked():
  state = {"engines": {"vllm": {"checked": {"latest": "0.33.0", "checked_at": eb.now(), "error": None}}}}
  assert eb.currency("vllm", "0.33.0", state)["words"].startswith("build 0.33.0 (latest at run time), checked")
  older = eb.currency("vllm", "0.31.0", state)
  assert older["is_latest"] is False and older["words"].startswith("build 0.31.0; latest was 0.33.0")
  state["engines"]["vllm"]["checked"]["checked_at"] = "2026-01-01T00:00:00Z"
  assert eb.currency("vllm", "0.33.0", state)["is_latest"] is None  # stale: not checked
  state["engines"]["vllm"]["checked"] = {"latest": None, "checked_at": eb.now(), "error": "TimeoutError: x"}
  assert "the last check failed: TimeoutError" in eb.currency("vllm", "0.33.0", state)["words"]
  assert "never checked" in eb.currency("vllm", "0.33.0", {"engines": {}})["words"]
  assert eb.currency("tinygrad", "abc", state) is None  # no upstream release source: no claim


def test_a_llama_run_is_named_by_the_release_its_commit_belongs_to(tmp_path, monkeypatch):
  from boltbeam.collectors import providers
  state = {"engines": {"llama.cpp": {"current": {"version": "0.6.0-dev", "commit": "097cf33d9", "folder": None},
                                     "pending": {"version": "0.9.0", "commit": "cd959ba"},
                                     "checked": {"latest": "0.9.0", "commit": "cd959bae99ff4bed22bef2604f45241c09bbb8c8",
                                                 "checked_at": eb.now(), "error": None}}}}
  monkeypatch.setattr(providers, "run_build", lambda run: "cd959bae9")
  assert eb.run_version("llama.cpp", tmp_path, state) == "0.9.0"
  assert eb.currency("llama.cpp", "0.9.0", state)["words"].startswith("build 0.9.0 (latest at run time)")
  monkeypatch.setattr(providers, "run_build", lambda run: "097cf33d9")
  ran = eb.run_version("llama.cpp", tmp_path, state)
  assert ran == "0.6.0-dev" and eb.currency("llama.cpp", ran, state)["words"].startswith("build 0.6.0-dev; latest was 0.9.0")


def update(read, *, seed=None, put=None, fit=None, engines=("vllm",)):
  puts = []
  def fake_put(engine, version, folder, vendor, fill):
    puts.append(version)
    return build(version, str(folder.parent))
  rows = engine_update.update(list(engines), backend="CUDA", read=read, seed=seed or (lambda e: build("0.31.0")),
                              put=put or fake_put, fit=fit or (lambda *a: {}), say=lambda _: None)
  return rows, puts


def test_update_installs_a_newer_release_as_pending_once():
  rows, puts = update(lambda url: pypi("0.31.0", "0.33.0"))
  assert puts == ["0.33.0"] and rows[0]["action"] == "installed" and rows[0]["pending"] == "0.33.0"
  assert rows[0]["current"] == "0.31.0"
  rows, puts = update(lambda url: pypi("0.31.0", "0.33.0"))
  assert puts == [] and rows[0]["action"] == "waiting"
  state = eb.load()
  assert state["engines"]["vllm"]["checked"]["latest"] == "0.33.0" and state["root"].endswith("engines")


def test_upstream_is_asked_at_most_once_per_policy_window():
  asked = []
  def read(url):
    asked.append(url)
    return pypi("0.31.0")
  update(read)
  update(read)
  assert len(asked) == 1
  state = eb.load()
  state["engines"]["vllm"]["checked"]["checked_at"] = "2026-01-01T00:00:00Z"  # older than engine_latest_max_age_h
  eb.save(state)
  update(read)
  assert len(asked) == 2


def test_update_installs_nothing_when_upstream_cannot_be_read_or_nothing_is_newer():
  def offline(url):
    raise OSError("no route")
  rows, puts = update(offline)
  assert puts == [] and rows[0]["action"] == "could_not_check" and "no route" in rows[0]["words"]
  rows, puts = update(lambda url: pypi("0.31.0"))  # a failed check is not fresh: asked again
  assert puts == [] and rows[0]["action"] == "up_to_date"


def test_a_release_that_does_not_fit_is_recorded_and_current_stays():
  def no(*a):
    raise engine_update.NotCompatible("it needs CUDA 13, a driver of 580 or newer; this machine has 570.1")
  rows, puts = update(lambda url: pypi("0.33.0"), fit=no)
  assert puts == [] and rows[0]["action"] == "not_compatible" and rows[0]["current"] == "0.31.0"
  assert rows[0]["words"] == "0.33.0 not compatible here: it needs CUDA 13, a driver of 580 or newer; this machine has 570.1; current stays"
  e = eb.load()["engines"]["vllm"]
  assert e["not_installed"]["version"] == "0.33.0" and "not compatible here" in e["not_installed"]["reason"] and not e.get("pending")


def trt_read(meta=TRT_META, index=TRT_INDEX):
  def read(url):
    if isinstance(meta, Exception) and "pypi.org" in url:
      raise meta
    return meta if "pypi.org" in url else index
  return read


def test_tensorrt_llm_fits_from_its_own_metadata():
  fill = engine_update.fits("tensorrt-llm", "1.2.1", CUDA, trt_read(), python="3.12", machine="x86_64", driver="595.99.02")
  assert fill == {"torch_spec": "<=2.10.0a0,>=2.9.1", "torch_index": "https://download.pytorch.org/whl/cu130"}


@pytest.mark.parametrize("kw,reads,why", [
  ({"driver": "570.86.10"}, {}, "it needs CUDA 13, a driver of 580 or newer; this machine has 570.86.10"),
  ({"driver": None}, {}, "this machine's driver could not be read"),
  ({"python": "3.11"}, {}, "publishes no wheel for Python 3.11 on x86_64"),
  ({"python": "3.9"}, {}, "it needs Python <4,>=3.10; BoltBeam runs on 3.9"),
  ({}, {"meta": {"info": {"requires_python": ">=3.10", "requires_dist": ["torch>=2.9"]}}}, "requirement could not be read"),
  ({}, {"meta": {"info": {"requires_dist": []}}}, "its Python requirement could not be read"),
  ({}, {"meta": OSError("404")}, "its requirements could not be read (OSError: 404)")])
def test_tensorrt_llm_that_does_not_fit_or_cannot_be_read_is_not_installed(kw, reads, why):
  args = {"python": "3.12", "machine": "x86_64", "driver": "595.99.02", **kw}
  with pytest.raises(engine_update.NotCompatible, match=why.replace("(", r"\(").replace(")", r"\)")):
    engine_update.fits("tensorrt-llm", "1.2.1", CUDA, trt_read(**reads), **args)


def test_venv_engines_refuse_another_vendor_before_downloading():
  with pytest.raises(engine_update.NotCompatible, match="for CUDA GPUs; this one is Metal"):
    engine_update.fits("vllm", "0.33.0", vendors.for_backend("Metal"))


@pytest.mark.parametrize("printed,version,commit", [
  ("version: 0.6.0-dev (build 11571, commit 097cf33d9)", "0.6.0-dev", "097cf33d9"),  # b11571 on the 5090 box
  ("version: 11571 (097cf33d)\nbuilt with cc", None, "097cf33d"), ("nothing", None, None)])
def test_the_build_here_reports_its_version_and_commit(monkeypatch, printed, version, commit):
  import subprocess
  monkeypatch.setenv("BOLTBEAM_LLAMA_BENCH", "/x/llama-bench")
  fake = lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", printed)  # noqa: E731
  got = engine_update.here("llama.cpp", run=fake)
  assert (got["version"], got["commit"], got["folder"]) == (version, commit, None)


def test_llama_build_clones_the_release_tag_with_its_history(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(engine_update, "sh", lambda argv, log, **kw: calls.append(argv) or "")
  monkeypatch.setattr(engine_update, "_arch", lambda vendor: "120")
  engine_update._llama(eb.row("llama.cpp")["install"], "0.9.0", tmp_path, CUDA, tmp_path / "log", {})
  assert calls[0][:5] == ["git", "clone", "--filter=tree:0", "--branch", "v0.9.0"] and "--depth" not in calls[0]
  assert "-DCMAKE_CUDA_ARCHITECTURES=120" in calls[1] and calls[2][-2:] == ["llama-bench", "llama-batched-bench"]


def test_engines_update_command_reports_json(monkeypatch, capsys):
  from boltbeam.collectors import engine_scan, providers
  def offline(url, **kw):
    raise TimeoutError("offline")
  monkeypatch.setattr(providers, "upstream", offline)
  monkeypatch.setattr(engine_scan, "cached_or_scan", lambda: {"found": {}})
  assert screen.main(["engines", "--update", "--engine", "vllm", "--target", "nvidia_sm120"]) == 0
  out = json.loads(capsys.readouterr().out)
  assert out["kind"] == "engines-update" and out["engines"][0]["action"] == "could_not_check"
  assert out["engines"][0]["current"] is None  # no vLLM here: nothing seeded


def pending_machine(tmp_path, older=None):
  state = {"schema": eb.SCHEMA, "engines": {}, "root": str(tmp_path / "engines")}
  eb.installed(state, "vllm", build("0.31.0"))
  eb.installed(state, "vllm", build("0.33.0"))
  if older:
    eb.entry(state, "vllm")["older"] = [older]
  eb.save(state)


def args(tmp_path, **kw):
  return argparse.Namespace(**{"provider": "vllm", "analyze": True, "run": str(tmp_path / "runs" / "r"), "target": "nvidia_sm120",
                               "bands": None, "update_engines": False, **kw})


def fake_pipeline(seen, fail=()):
  def run(a, out=None):
    seen.append((pathlib.Path(a.run).name, os.environ.get("BOLTBEAM_VLLM_PYTHON")))
    pathlib.Path(a.run).mkdir(parents=True, exist_ok=True)
    (out or io.StringIO()).write("stage load: start\n")
    return 1 if pathlib.Path(a.run).name in fail else 0
  return run


def old_build(tmp_path):
  folder = tmp_path / "engines" / "vllm" / "0.29.0"
  folder.mkdir(parents=True)
  (folder / eb.MARK).write_text("")
  return {**build("0.29.0"), "folder": str(folder)}


def test_a_pending_build_runs_the_previous_once_compares_becomes_current_and_prunes(tmp_path, monkeypatch):
  from boltbeam.workflow import replay
  old = old_build(tmp_path)
  pending_machine(tmp_path, older=old)
  seen, out = [], io.StringIO()
  monkeypatch.setattr(pp, "pipeline", fake_pipeline(seen))
  monkeypatch.setattr(pp, "output_run", lambda run: None)
  monkeypatch.setattr(replay, "diff", lambda a, b, bands=None: {"parts": []})
  monkeypatch.setattr(replay, "diff_lines", lambda d: ["Engine upgrade audit: ..."])
  assert pp.pipeline_current(args(tmp_path), out) == 0
  assert seen == [("r-was-0.31.0", "/b/0.31.0/venv/bin/python"), ("r", "/b/0.33.0/venv/bin/python")]
  assert "previous build: stage load: start" in out.getvalue()  # the screen reads only the new build's stage lines
  record = json.loads((tmp_path / "runs" / "r" / eb.UPGRADE_FILE).read_text())
  assert record["lines"] == ["Engine upgrade audit: ..."] and record["before_run"] == "r-was-0.31.0"
  assert eb.comparison(tmp_path / "runs" / "r")["words"].startswith("compared once with the previous build 0.31.0")
  e = eb.load()["engines"]["vllm"]
  assert e["current"]["version"] == "0.33.0" and e["previous"]["version"] == "0.31.0" and e["pending"] is None
  assert not pathlib.Path(old["folder"]).exists() and e["deleted"][0]["version"] == "0.29.0"
  seen.clear()
  assert pp.pipeline_current(args(tmp_path, run=str(tmp_path / "runs" / "s")), out) == 0
  assert seen == [("s", "/b/0.33.0/venv/bin/python")]  # once: the next run uses the new current build alone


def test_a_refused_comparison_deletes_nothing(tmp_path, monkeypatch):
  from boltbeam.workflow import diff, replay
  old = old_build(tmp_path)
  pending_machine(tmp_path, older=old)
  monkeypatch.setattr(pp, "pipeline", fake_pipeline([]))
  monkeypatch.setattr(pp, "output_run", lambda run: None)
  monkeypatch.setattr(replay, "diff", lambda a, b, bands=None: (_ for _ in ()).throw(diff.DiffRefused("a run holds no captured where table")))
  assert pp.pipeline_current(args(tmp_path), io.StringIO()) == 0
  e = eb.load()["engines"]["vllm"]
  assert pathlib.Path(old["folder"]).exists() and not e.get("deleted")
  assert [b["version"] for b in e["older"]] == ["0.29.0"]


def test_a_failed_new_build_run_keeps_the_previous_build_current(tmp_path, monkeypatch):
  pending_machine(tmp_path)
  monkeypatch.setattr(pp, "pipeline", fake_pipeline([], fail=("r",)))
  assert pp.pipeline_current(args(tmp_path), io.StringIO()) == 1
  e = eb.load()["engines"]["vllm"]
  assert e["current"]["version"] == "0.31.0" and e["pending"]["version"] == "0.33.0"


def test_a_run_without_analyze_or_with_the_users_build_makes_no_comparison(tmp_path, monkeypatch):
  pending_machine(tmp_path)
  seen = []
  monkeypatch.setattr(pp, "pipeline", fake_pipeline(seen))
  assert pp.pipeline_current(args(tmp_path, analyze=False), io.StringIO()) == 0
  monkeypatch.setenv("BOLTBEAM_VLLM_PYTHON", "/mine/python")
  assert pp.pipeline_current(args(tmp_path), io.StringIO()) == 0
  assert seen == [("r", "/b/0.31.0/venv/bin/python"), ("r", "/mine/python")]
  assert eb.load()["engines"]["vllm"]["pending"]["version"] == "0.33.0"


def test_build_lines_are_the_words_summary_report_and_tui_show():
  engine = {"build": {"latest": {"words": "build 0.9.0 (latest at run time), checked 2026-11-08T10:00:00Z"}},
            "comparison": {"words": "compared once with the previous build 0.8.0 (run r-was-0.8.0); see engine_upgrade.json"}}
  assert eb.build_lines(engine) == ["Upstream: build 0.9.0 (latest at run time), checked 2026-11-08T10:00:00Z.",
                                    "Upgrade comparison: compared once with the previous build 0.8.0 (run r-was-0.8.0); see engine_upgrade.json."]
  assert eb.build_lines({"build": None, "comparison": None}) == []


def test_standing_is_behind_waiting_up_to_date_or_unknown_from_the_saved_check():
  def at(checked, **e):
    return eb.standing("vllm", {"engines": {"vllm": {"checked": checked, **e}}})
  ok = {"latest": "0.33.0", "checked_at": eb.now(), "error": None}
  assert at(ok, current=build("0.31.0"))["state"] == "behind" and at(ok, current=build("0.31.0"))["words"] == "0.31.0 · 0.33.0 is out"
  assert at(ok, current=build("0.33.0"))["state"] == "up_to_date"
  assert at(ok, current=build("0.31.0"), pending=build("0.33.0"))["state"] == "waiting"
  assert at({}, current=build("0.31.0"))["words"] == "latest release unknown (never checked)"
  failed = at({"latest": None, "checked_at": eb.now(), "error": "TimeoutError: x"}, current=build("0.33.0"))
  assert failed["state"] == "unknown" and "the last check failed: TimeoutError" in failed["words"]
  assert at({**ok, "checked_at": "2026-01-01T00:00:00Z"}, current=build("0.33.0"))["state"] == "unknown"  # a stale "equal"
  assert eb.standing("vllm", {"engines": {"vllm": {"checked": ok}}}, found=build("0.31.0"))["current"] == "0.31.0"
  assert eb.standing("tinygrad", {"engines": {}}) is None  # no upstream release source: no claim


def test_engines_latest_looks_once_per_window_installs_nothing_and_records_no_build(monkeypatch, capsys):
  from boltbeam.collectors import providers
  asked = []
  monkeypatch.setattr(providers, "upstream", lambda url, **kw: asked.append(url) or pypi("0.31.0", "0.33.0"))
  monkeypatch.setattr(engine_update, "here", lambda engine: build("0.31.0"))
  monkeypatch.setattr(engine_update, "install", lambda *a, **kw: pytest.fail("--latest installs nothing"))
  for _ in range(2):
    assert screen.main(["engines", "--latest", "--engine", "vllm"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "engines-latest" and out["engines"][0]["state"] == "behind" and out["engines"][0]["latest"] == "0.33.0"
  assert len(asked) == 1 and "current" not in eb.load()["engines"]["vllm"]


def test_the_feature_is_upgrade_engines_and_the_old_flag_still_parses(monkeypatch, tmp_path):
  root = pathlib.Path(__file__).resolve().parent.parent
  said = [p for d in ("boltbeam", "docs", "tui") for p in (root / d).rglob("*") if p.suffix in (".py", ".md", ".go", ".txt", ".ansi")
          and not p.name.endswith("_test.go") and not p.name.startswith("release-")  # a release record is history
          and "stay current" in p.read_text(errors="ignore").lower()]
  assert said == []
  seen = []
  monkeypatch.setattr(pp, "pipeline_current", lambda args, out=None: seen.append(args.update_engines) or 0)
  for flag in ("--upgrade-engines", "--update-engines", None):
    assert screen.main(["pipeline", "m", "--run", str(tmp_path / "r"), "--target", "nvidia_sm120", *([flag] if flag else [])]) == 0
  assert seen == [True, True, False]
