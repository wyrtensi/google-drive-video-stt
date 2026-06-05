from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.config import Config
from src.presets import Preset
from src import preset_pipeline
from src.preset_pipeline import PresetResult, aggregate_error, run_presets


def _config(**over) -> Config:
    base = dict(
        folder_ids=[],
        poll_interval=600,
        bitrate="96k",
        data_dir=Path("data"),
        proxy_url="",
        stt_provider="",
        openai_api_key="sk-test",
        deepgram_api_key="",
        stt_language="",
    )
    base.update(over)
    return Config(**base)


class FakePipeline:
    """Records construction + run args; output echoes instructions and input.

    A preset whose instructions contain ``FAIL`` raises, to exercise partial
    failure handling. Optionally blocks on a shared barrier to assert that
    independent presets run concurrently.
    """

    instances: list["FakePipeline"] = []
    barrier: threading.Barrier | None = None

    def __init__(self, *, api_key, model, proxy_url, use_batch):
        self.model = model
        self.use_batch = use_batch
        self.calls: list[tuple[str, str]] = []
        FakePipeline.instances.append(self)

    def run(self, instructions, input_text):
        self.calls.append((instructions, input_text))
        if FakePipeline.barrier is not None:
            # Will raise BrokenBarrierError on timeout if the other independent
            # preset is not dispatched at the same time.
            FakePipeline.barrier.wait(timeout=5)
        if "FAIL" in instructions:
            raise RuntimeError(f"boom: {instructions}")
        return f"out[{instructions}]<<{input_text}>>", {"output_tokens": 1}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    FakePipeline.instances = []
    FakePipeline.barrier = None
    monkeypatch.setattr(preset_pipeline, "OpenAIPipeline", FakePipeline)
    yield


def _preset(name, instructions=None, **over) -> Preset:
    return Preset(name=name, instructions=instructions or f"INSTR_{name}", **over)


def test_no_dep_preset_gets_transcript_prompt():
    presets = {"keypoints": _preset("keypoints")}
    results = run_presets(
        "Speaker 1: hi", "Alice and Bob.mp4", _config(), presets,
        speaker_names=["Alice", "Bob"],
    )
    assert set(results) == {"keypoints"}
    res = results["keypoints"]
    assert res.ok
    # The raw-transcript prompt carries participant hints + transcript text.
    instr, input_text = FakePipeline.instances[0].calls[0]
    assert instr == "INSTR_keypoints"
    assert "Alice, Bob" in input_text
    assert "Speaker 1: hi" in input_text


def test_dependency_input_is_labeled_concatenation():
    presets = {
        "cleanup": _preset("cleanup"),
        "keypoints": _preset("keypoints", depends_on=("cleanup",)),
    }
    results = run_presets("transcript", "f.mp4", _config(), presets)
    assert results["cleanup"].ok and results["keypoints"].ok

    # The dependent preset's input is its dependency outputs with a labeled header.
    kp = next(p for p in FakePipeline.instances if "INSTR_keypoints" in p.calls[0][0])
    _instr, kp_input = kp.calls[0]
    assert kp_input.startswith("## cleanup")
    assert "out[INSTR_cleanup]" in kp_input


def test_topological_order_multiple_dependencies():
    presets = {
        "cleanup": _preset("cleanup"),
        "a": _preset("a", depends_on=("cleanup",)),
        "b": _preset("b", depends_on=("cleanup",)),
        "merge": _preset("merge", depends_on=("a", "b")),
    }
    results = run_presets("t", "f.mp4", _config(), presets)
    assert all(results[name].ok for name in presets)
    merge = next(p for p in FakePipeline.instances if "INSTR_merge" in p.calls[0][0])
    merge_input = merge.calls[0][1]
    assert "## a" in merge_input and "## b" in merge_input
    assert "out[INSTR_a]" in merge_input and "out[INSTR_b]" in merge_input


def test_independent_presets_dispatch_concurrently():
    FakePipeline.barrier = threading.Barrier(2)
    presets = {"a": _preset("a"), "b": _preset("b")}
    results = run_presets("t", "f.mp4", _config(openai_max_parallel=2), presets)
    # Both reaching the 2-party barrier within the timeout proves concurrency.
    assert results["a"].ok and results["b"].ok


def test_model_and_batch_fallback():
    presets = {
        "default": _preset("default"),
        "custom": _preset("custom", model="gpt-special", batch=True),
    }
    run_presets(
        "t", "f.mp4", _config(openai_model="gpt-global", openai_batch=False), presets
    )
    by_model = {inst.model: inst for inst in FakePipeline.instances}
    assert "gpt-global" in by_model
    assert "gpt-special" in by_model
    assert by_model["gpt-global"].use_batch is False
    assert by_model["gpt-special"].use_batch is True


@pytest.mark.parametrize(
    "preset_batch, openai_batch, expected",
    [
        (True, True, True),
        (True, False, True),
        (False, True, False),
        (False, False, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_batch_override_matrix(preset_batch, openai_batch, expected):
    # preset.batch wins when set; otherwise the global openai.batch applies.
    presets = {"a": _preset("a", batch=preset_batch)}
    run_presets("t", "f.mp4", _config(openai_batch=openai_batch), presets)
    assert FakePipeline.instances[0].use_batch is expected


def test_batch_wait_default_true_runs():
    # batch_wait unset on the preset and the global default (true) lets the DAG run.
    presets = {"a": _preset("a")}
    results = run_presets("t", "f.mp4", _config(), presets)
    assert results["a"].ok


def test_resolved_batch_wait_false_raises_not_implemented():
    # A batch preset that resolves batch_wait=false (per-preset or via the global
    # openai.batch_wait) is rejected: only the synchronous wait path is implemented.
    # batch_wait only governs batch submissions, so the preset must be in batch mode.
    preset = _preset("a", batch=True, batch_wait=False)
    with pytest.raises(NotImplementedError, match="batch_wait"):
        preset_pipeline._run_one(
            preset,
            transcript="t",
            file_name="f.mp4",
            config=_config(),
            speaker_names=None,
            dep_results={},
        )


def test_global_batch_wait_false_raises_not_implemented():
    preset = _preset("a", batch=True)
    with pytest.raises(NotImplementedError, match="batch_wait"):
        preset_pipeline._run_one(
            preset,
            transcript="t",
            file_name="f.mp4",
            config=_config(openai_batch_wait=False),
            speaker_names=None,
            dep_results={},
        )


def test_non_batch_preset_ignores_batch_wait_false():
    # A synchronous (non-batch) preset is already inline, so a resolved
    # batch_wait=false must be a no-op rather than raising.
    preset = _preset("a", batch=False, batch_wait=False)
    res = preset_pipeline._run_one(
        preset,
        transcript="t",
        file_name="f.mp4",
        config=_config(openai_batch_wait=False),
        speaker_names=None,
        dep_results={},
    )
    assert res.ok


def test_preset_batch_wait_true_overrides_global_false():
    # An explicit per-preset batch_wait=true beats a global openai.batch_wait=false.
    preset = _preset("a", batch_wait=True)
    res = preset_pipeline._run_one(
        preset,
        transcript="t",
        file_name="f.mp4",
        config=_config(openai_batch_wait=False),
        speaker_names=None,
        dep_results={},
    )
    assert res.ok


def test_only_runs_closure():
    presets = {
        "cleanup": _preset("cleanup"),
        "keypoints": _preset("keypoints", depends_on=("cleanup",)),
        "other": _preset("other"),
    }
    results = run_presets("t", "f.mp4", _config(), presets, only=["keypoints"])
    # keypoints + its dependency cleanup run; the unrelated preset does not.
    assert set(results) == {"cleanup", "keypoints"}


def test_only_single_independent_preset():
    presets = {"a": _preset("a"), "b": _preset("b")}
    results = run_presets("t", "f.mp4", _config(), presets, only=["a"])
    assert set(results) == {"a"}


def test_only_unknown_preset_raises():
    presets = {"a": _preset("a")}
    with pytest.raises(ValueError, match="unknown preset"):
        run_presets("t", "f.mp4", _config(), presets, only=["missing"])


def test_precomputed_dependency_is_reused_not_rerun():
    presets = {
        "cleanup": _preset("cleanup"),
        "keypoints": _preset("keypoints", depends_on=("cleanup",)),
    }
    results = run_presets(
        "t",
        "f.mp4",
        _config(),
        presets,
        only=["keypoints"],
        precomputed={"cleanup": "persisted cleanup"},
    )
    # cleanup is not re-run: no pipeline call carries its instructions, and the
    # seeded text feeds keypoints' input verbatim.
    cleanup_runs = [i for i in FakePipeline.instances if any("INSTR_cleanup" in c[0] for c in i.calls)]
    assert cleanup_runs == []
    assert results["cleanup"].text == "persisted cleanup"
    keypoints_input = FakePipeline.instances[0].calls[0][1]
    assert "persisted cleanup" in keypoints_input


def test_precomputed_does_not_override_explicitly_requested_preset():
    presets = {"cleanup": _preset("cleanup")}
    results = run_presets(
        "t",
        "f.mp4",
        _config(),
        presets,
        only=["cleanup"],
        precomputed={"cleanup": "stale"},
    )
    # cleanup is explicitly requested, so it runs even though a stale output exists.
    assert results["cleanup"].text != "stale"
    assert FakePipeline.instances[0].calls


def test_precomputed_does_not_override_one_shot_only_iterable():
    # ``only`` is consumed twice internally; a generator must not be exhausted by
    # the first use, or the explicitly requested preset would be wrongly seeded
    # from ``precomputed`` and skipped.
    presets = {"cleanup": _preset("cleanup")}
    results = run_presets(
        "t",
        "f.mp4",
        _config(),
        presets,
        only=(name for name in ["cleanup"]),
        precomputed={"cleanup": "stale"},
    )
    assert results["cleanup"].text != "stale"
    assert FakePipeline.instances[0].calls


def test_dependency_names_excludes_targets_themselves():
    presets = {
        "cleanup": _preset("cleanup"),
        "keypoints": _preset("keypoints", depends_on=("cleanup",)),
        "managers": _preset("managers", depends_on=("cleanup",)),
    }
    assert preset_pipeline.dependency_names(presets, ["keypoints"]) == {"cleanup"}
    assert preset_pipeline.dependency_names(presets, ["cleanup"]) == set()


def test_partial_failure_skips_dependents_runs_independent():
    presets = {
        "bad": _preset("bad", instructions="FAIL_root"),
        "child": _preset("child", depends_on=("bad",)),
        "grandchild": _preset("grandchild", depends_on=("child",)),
        "independent": _preset("independent"),
    }
    results = run_presets("t", "f.mp4", _config(), presets)

    assert results["bad"].error is not None and not results["bad"].skipped
    assert results["child"].skipped
    assert results["grandchild"].skipped
    assert results["independent"].ok

    combined = aggregate_error(results)
    assert combined is not None
    assert "bad" in combined and "child" in combined


def test_aggregate_error_none_when_all_ok():
    presets = {"a": _preset("a"), "b": _preset("b")}
    results = run_presets("t", "f.mp4", _config(), presets)
    assert aggregate_error(results) is None


def test_preset_result_usage_carried():
    presets = {"a": _preset("a")}
    results = run_presets("t", "f.mp4", _config(), presets)
    assert results["a"].usage == {"output_tokens": 1}
    assert isinstance(results["a"], PresetResult)
