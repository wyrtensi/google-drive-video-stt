from __future__ import annotations

from pathlib import Path

import pytest

from src import presets as presets_module
from src.presets import (
    BUILTIN_PRESETS,
    INSTRUCTIONS,
    Preset,
    default_artifact_suffix,
    load_packaged_prompt,
    merge_presets,
    validate_dag,
)


# --- suffix derivation ------------------------------------------------------


def test_default_artifact_suffix_derives_from_name():
    assert default_artifact_suffix("keypoints") == ".keypoints.md"
    assert default_artifact_suffix("expertizeme-managers") == ".expertizeme-managers.md"


def test_preset_fills_default_suffix_when_empty():
    preset = Preset(name="cleanup", instructions="do it")
    assert preset.artifact_suffix == ".cleanup.md"


def test_preset_keeps_explicit_suffix():
    preset = Preset(name="cleanup", instructions="do it", artifact_suffix=".clean.txt")
    assert preset.artifact_suffix == ".clean.txt"


def test_preset_rejects_empty_name():
    with pytest.raises(ValueError):
        Preset(name="", instructions="x")


# --- built-ins --------------------------------------------------------------


def test_builtin_keypoints_present():
    by_name = {p.name: p for p in BUILTIN_PRESETS}
    assert "keypoints" in by_name
    assert by_name["keypoints"].instructions == INSTRUCTIONS
    assert by_name["keypoints"].artifact_suffix == ".keypoints.md"
    assert by_name["keypoints"].enabled is True
    assert by_name["keypoints"].prompt_file == "keypoints.md"


# --- packaged prompt assets -------------------------------------------------


def test_load_packaged_prompt_returns_keypoints_text():
    text = load_packaged_prompt("keypoints.md")
    assert "## Задачи" in text
    assert "## Тезисы" in text
    assert "## Открытые вопросы" in text


def test_instructions_equals_keypoints_asset_text():
    assert INSTRUCTIONS == load_packaged_prompt("keypoints.md")


def test_load_packaged_prompt_missing_raises():
    with pytest.raises(ValueError, match="missing or empty"):
        load_packaged_prompt("does-not-exist.md")


def test_load_packaged_prompt_uses_importlib_resources_without_source_dir(monkeypatch):
    # Simulate an installed/container layout where the source-dir fallback is absent
    # (no top-level ``assets/`` and no ``src/assets/prompts`` on disk). The prompt must
    # still resolve via ``importlib.resources`` package data alone.
    monkeypatch.setattr(
        presets_module, "_SRC_PROMPTS_DIR", Path("/nonexistent/assets/prompts")
    )
    text = load_packaged_prompt("keypoints.md")
    assert "## Задачи" in text
    assert text.strip()


def test_load_packaged_prompt_falls_back_to_source_dir(monkeypatch):
    # Force the ``importlib.resources`` lookup to fail so the ``Path``-relative
    # source-dir fallback is exercised and still finds the shipped package data.
    monkeypatch.setattr(
        presets_module, "_PACKAGED_PROMPTS_PACKAGE", "src.assets.does_not_exist"
    )
    text = load_packaged_prompt("keypoints.md")
    assert "## Задачи" in text
    assert text.strip()


def test_preset_accepts_prompt_file_field():
    preset = Preset(name="kp", instructions="x", prompt_file="kp.md")
    assert preset.prompt_file == "kp.md"


# --- merge ------------------------------------------------------------------


def test_merge_with_no_config_returns_builtins():
    merged = merge_presets(BUILTIN_PRESETS, None)
    assert set(merged) == {"keypoints"}
    assert merged["keypoints"].instructions == INSTRUCTIONS


def test_merge_overrides_builtin_field_by_field():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {"keypoints": {"depends_on": ["transcript-cleanup"], "model": "gpt-5.4"}},
    )
    kp = merged["keypoints"]
    # Overridden fields change...
    assert kp.depends_on == ("transcript-cleanup",)
    assert kp.model == "gpt-5.4"
    # ...untouched fields keep the built-in values.
    assert kp.instructions == INSTRUCTIONS
    assert kp.artifact_suffix == ".keypoints.md"


def test_merge_adds_new_preset():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {"expertizeme-managers": {"instructions": "summarize for managers"}},
    )
    assert set(merged) == {"keypoints", "expertizeme-managers"}
    new = merged["expertizeme-managers"]
    assert new.instructions == "summarize for managers"
    assert new.artifact_suffix == ".expertizeme-managers.md"


def test_merge_new_preset_requires_instructions_or_prompt_file():
    with pytest.raises(ValueError, match="must define instructions"):
        merge_presets(BUILTIN_PRESETS, {"orphan": {"depends_on": ["keypoints"]}})


def test_merge_new_preset_accepts_prompt_file_without_instructions():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {"managers": {"prompt_file": "managers.md"}},
    )
    new = merged["managers"]
    assert new.prompt_file == "managers.md"
    # prompt_file presets carry no inline instructions; config.py resolves them.
    assert new.instructions == ""


def test_merge_new_preset_keeps_instructions_over_prompt_file():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {"managers": {"instructions": "inline", "prompt_file": "managers.md"}},
    )
    new = merged["managers"]
    assert new.instructions == "inline"
    assert new.prompt_file == "managers.md"


def test_merge_disables_builtin():
    merged = merge_presets(BUILTIN_PRESETS, {"keypoints": {"enabled": False}})
    assert merged == {}


def test_merge_disables_new_preset():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {"extra": {"instructions": "x", "enabled": False}},
    )
    assert set(merged) == {"keypoints"}


def test_merge_preserves_order_builtins_then_new():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {
            "transcript-cleanup": {"instructions": "clean"},
            "expertizeme-managers": {
                "instructions": "managers",
                "depends_on": ["transcript-cleanup"],
            },
        },
    )
    assert list(merged) == ["keypoints", "transcript-cleanup", "expertizeme-managers"]


def test_merge_batch_and_model_fallback_none():
    merged = merge_presets(BUILTIN_PRESETS, {"extra": {"instructions": "x"}})
    assert merged["extra"].model is None
    assert merged["extra"].batch is None
    assert merged["extra"].batch_wait is None


def test_merge_parses_batch_wait():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {
            "wait": {"instructions": "x", "batch_wait": True},
            "nowait": {"instructions": "y", "batch_wait": False},
        },
    )
    assert merged["wait"].batch_wait is True
    assert merged["nowait"].batch_wait is False


def test_merge_overrides_builtin_batch_wait():
    merged = merge_presets(BUILTIN_PRESETS, {"keypoints": {"batch_wait": False}})
    assert merged["keypoints"].batch_wait is False


def test_merge_rejects_non_mapping_entry():
    with pytest.raises(ValueError, match="must be a mapping"):
        merge_presets(BUILTIN_PRESETS, {"bad": ["not", "a", "map"]})


# --- DAG validation ---------------------------------------------------------


def _named(name, deps=()):
    return Preset(name=name, instructions="x", depends_on=tuple(deps))


def test_validate_dag_accepts_canonical_chain():
    presets = merge_presets(
        BUILTIN_PRESETS,
        {
            "transcript-cleanup": {"instructions": "clean"},
            "keypoints": {"depends_on": ["transcript-cleanup"]},
            "expertizeme-managers": {
                "instructions": "managers",
                "depends_on": ["transcript-cleanup"],
            },
        },
    )
    validate_dag(presets)  # does not raise


def test_validate_dag_rejects_missing_dependency():
    presets = {"a": _named("a", ["ghost"])}
    with pytest.raises(ValueError, match="unknown or disabled"):
        validate_dag(presets)


def test_validate_dag_rejects_dependency_on_disabled():
    merged = merge_presets(
        BUILTIN_PRESETS,
        {
            "cleanup": {"instructions": "clean", "enabled": False},
            "summary": {"instructions": "sum", "depends_on": ["cleanup"]},
        },
    )
    with pytest.raises(ValueError, match="unknown or disabled"):
        validate_dag(merged)


def test_validate_dag_rejects_self_cycle():
    presets = {"a": _named("a", ["a"])}
    with pytest.raises(ValueError, match="cycle"):
        validate_dag(presets)


def test_validate_dag_rejects_two_node_cycle():
    presets = {
        "a": _named("a", ["b"]),
        "b": _named("b", ["a"]),
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_dag(presets)


def test_validate_dag_rejects_three_node_cycle():
    presets = {
        "a": _named("a", ["b"]),
        "b": _named("b", ["c"]),
        "c": _named("c", ["a"]),
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_dag(presets)


def test_validate_dag_accepts_diamond():
    presets = {
        "root": _named("root"),
        "left": _named("left", ["root"]),
        "right": _named("right", ["root"]),
        "join": _named("join", ["left", "right"]),
    }
    validate_dag(presets)  # diamond is a DAG, not a cycle
