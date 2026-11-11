"""What the TUI's Setup reads from the seam for three pipeline features: the engine settings a run may ask for
(capture_settings.offered, one table with the pipeline's parser), repeated runs and their band (screen pipeline
--repeats, repeat_bands.summary), and Measure MTP (vllm_mtp.offer in the recommendation)."""
import io
import json

import pytest

from boltbeam.collectors import capture_scope
from boltbeam.collectors import capture_settings as cs, vllm_mtp
from boltbeam.workflow import recommend as rec, repeat_bands as rb, screen
from boltbeam.workflow import pipeline as pp
from tests.test_recommend import checkpoint, facts, run
from tests.test_repeat_bands import repeat


def test_offered_settings_follow_the_engine_and_the_parser(monkeypatch, tmp_path):
  assert [s["name"] for s in cs.offered("vllm")] == [k for k in cs.ASKED if k in cs.VLLM_SETTINGS]
  assert [s["name"] for s in cs.offered("llama.cpp")] == ["chunk_size", "n_gpu_layers", "n_cpu_moe", "override_tensor", "cpu_moe", "moe_cache_mib",
                                                         "moe_cache_cpu_share", "spec_type", "draft_model", "cache_type_k_draft", "cache_ram"]
  by_llama = {s["name"]: s for s in cs.offered("llama.cpp")}  # a whole number whose least is 0 says so (the TUI takes 0)
  assert by_llama["cache_ram"]["least"] == 0 and by_llama["n_cpu_moe"]["least"] == 0 and "least" not in by_llama["chunk_size"]
  assert cs.offered("tinygrad") == []
  by = {s["name"]: s for s in cs.offered("vllm")}
  assert by["mtp_prompt"]["choices"] == vllm_mtp.prompt_set()["ids"] and by["gemm_backends"]["choices"] == list(cs.GEMM_BACKENDS)
  assert by["mtp"]["draft_head"] and by["mtp"]["whole"] and not by["attention_backend"]["draft_head"]
  seen = {}
  monkeypatch.setattr(pp, "pipeline_current", lambda args, out=None: seen.setdefault("args", args) and 0)
  argv = ["pipeline", "m", "--run", str(tmp_path / "r"), "--target", "nvidia_sm120", "--provider", "vllm"]
  for s, v in zip(by.values(), ["512", "TRITON_ATTN", "TRITON", "2", "explain-tcp"]):
    argv += [s["flag"], v]  # every flag the screen is given parses, and lands in the request with its type
  assert screen.main(argv) == 0
  assert pp._engine_settings(seen["args"]) == {"chunk_size": 512, "attention_backend": "TRITON_ATTN",
                                                   "gemm_backends": "TRITON", "mtp": 2, "mtp_prompt": "explain-tcp"}


def test_cpu_moe_and_the_draft_cache_type_parse_like_the_other_llama_flags(monkeypatch, tmp_path):
  by = {s["name"]: s for s in cs.offered("llama.cpp")}
  assert by["cpu_moe"]["choices"] == ["on"] and not by["cpu_moe"]["whole"] and by["cache_type_k_draft"]["choices"] is None
  seen = {}
  monkeypatch.setattr(pp, "pipeline_current", lambda args, out=None: seen.setdefault("args", args) and 0)
  argv = ["pipeline", "m", "--run", str(tmp_path / "r"), "--target", "nvidia_sm120", "--provider", "llama.cpp",
          by["cpu_moe"]["flag"], "on", by["cache_type_k_draft"]["flag"], "q8_0", "--moe-cache-mib", "20000"]
  assert screen.main(argv) == 0
  got = pp._engine_settings(seen["args"])
  assert got == {"cpu_moe": True, "cache_type_k_draft": "q8_0", "moe_cache_mib": 20000}
  cs.check_llama(got)
  assert cs.render_llama(got) == ["--cpu-moe", "--moe-cache-mib", "20000", "-ctkd", "q8_0"]
  with pytest.raises(SystemExit):  # a switch has one value
    screen.main(argv[:7] + ["--cpu-moe", "off"])


def test_repeats_run_the_setup_n_times_then_save_the_band(monkeypatch, tmp_path):
  said = io.StringIO()
  monkeypatch.setattr(pp.pipeline_repeated, "__defaults__", (said,))  # its lines, as the screen reads them
  runs, saved = [], {}
  monkeypatch.setattr(pp, "pipeline_current", lambda args, out: runs.append((args.run, args.update_engines, args.hold_machine)) or 0)
  monkeypatch.setattr(pp, "output_run", lambda run: None)
  monkeypatch.setattr(rb, "save", lambda paths, out, fixed: saved.update(paths=paths, out=out, fixed=fixed) or capture_scope.strength(len(paths)))
  r = tmp_path / "q-001"
  assert screen.main(["pipeline", "m", "--run", str(r), "--target", "nvidia_sm120", "--repeats", "3", "--update-engines"]) == 0
  # the repeats hold run 1's machine facts: a re-read between them moved the limit and the band refused (5090, 2026-11-09)
  assert runs == [(str(r), True, None), (str(tmp_path / "q-001-repeat-2"), False, r), (str(tmp_path / "q-001-repeat-3"), False, r)]
  assert saved["out"] == r / capture_scope.ARTIFACT and saved["fixed"]["n"] == 3 and json.loads((r / rb.PLAN).read_text())["n"] == 3
  lines = said.getvalue().splitlines()
  assert lines[0] == "repeat 1/3: q-001" and lines[-1] == "repeat band: done: n = 3, a row reads moved by chance 50% of the time"
  monkeypatch.setattr(rb, "save", lambda *a: (_ for _ in ()).throw(ValueError("repeat work differs")))
  assert screen.main(["pipeline", "m", "--run", str(tmp_path / "x"), "--target", "nvidia_sm120", "--repeats", "2"]) == 1
  assert said.getvalue().splitlines()[-1] == "repeat band: failed: repeat work differs"
  assert screen.main(["pipeline", "m", "--run", str(tmp_path / "y"), "--target", "nvidia_sm120", "--repeats", "1"]) == 2
  monkeypatch.setattr(pp, "pipeline_current", lambda args, out: 1 if args.run.endswith("-2") else 0)
  assert screen.main(["pipeline", "m", "--run", str(tmp_path / "z"), "--target", "nvidia_sm120", "--repeats", "3"]) == 1


def test_repeat_summary_words_the_band_of_this_setup(tmp_path):
  a = repeat(tmp_path, "a")
  b = repeat(tmp_path, "b", {"big": {"ms": 100.1}, "small": {"ms": 0.012}}, date="2026-10-13")
  assert rb.summary(a) is None
  rb.save([a, b], a / capture_scope.ARTIFACT)
  got = rb.summary(a)
  assert got["n"] == 2 and got["weak"] and "needs at least 4" in got["pin"]
  assert got["lines"] == ["prefill 512: median 100.561 ms, 100.510 to 100.612 ms over 2 runs (spread 0.1%)"]
  assert rb.summary(b) is None  # the band is kept in the first run only


def test_measure_mtp_is_offered_for_vllm_with_a_draft_head(tmp_path):
  assert vllm_mtp.offer("llama.cpp", 1, facts=True) is None and vllm_mtp.offer("vllm", 0, facts=True) is None
  assert vllm_mtp.offer("vllm", 1, facts=True) == {"ks": list(vllm_mtp.KS), "blocked": None}
  assert "facts" in vllm_mtp.offer("vllm", 1, facts=False)["blocked"]
  out = run(checkpoint(tmp_path, 15.7), facts())
  assert out["mtp_measure"]["ks"] == list(vllm_mtp.KS) and out["mtp_measured"] is None
  assert run(checkpoint(tmp_path, 15.7, mtp=0), facts())["mtp_measure"] is None
  record = {"depths": {"short": {}}, "prompts": {}, "tokens": 256, "long_tokens": 0, "measured_at": "2026-11-08T10:00:00"}
  kept = rec.facts_with_mtp(facts(), "vllm", str(tmp_path), record)
  assert run(checkpoint(tmp_path, 15.7), kept)["mtp_measured"]["measured_at"] == "2026-11-08T10:00:00"


def test_providers_carry_each_engines_settings_and_the_repeat_n(monkeypatch):
  from boltbeam.collectors import providers
  from boltbeam.target.targets import get_target
  monkeypatch.setattr(providers, "available", lambda *a, **k: [{"provider": "vllm"}, {"provider": "tinygrad"}])
  monkeypatch.setattr(screen, "hardware_profile", lambda: {"gpu": {"name": "x", "devices": [{"name": "x", "memory_bytes": 1}]}})
  got = screen.provider_list(get_target("nvidia_sm120"))
  assert got["repeat_runs"] == capture_scope.PIN_RUNS
  assert [r["settings"] for r in got["providers"]] == [cs.offered("vllm"), []]


def test_a_held_repeat_takes_the_first_runs_machine_facts_as_they_are(tmp_path):
  """llama.cpp's 27B on the 5090: repeat 2 read the GPU again (1709.3 GB/s against 1710.0), every limit moved by a
  fraction of a percent, and the band refused 'repeat work differs' (2026-11-09)."""
  from boltbeam.artifacts.run_io import MACHINE
  first, second = tmp_path / "r-001", tmp_path / "r-001-repeat-2"
  first.mkdir(); second.mkdir()
  facts = {"driver": "595.99.02", "gpus": [{"index": 0, "read_gbs": 1710.0, "read_from": "probe", "uncertainty": 0.0023}]}
  (first / MACHINE).write_text(json.dumps(facts))
  got = pp.machine(second, hold=first)
  assert got["reused"] and got["reused_because"] == "held fixed for the repeat band: the facts of r-001"
  assert json.loads((second / MACHINE).read_text()) == facts
