"""DAG executor for the config-defined OpenAI preset pipeline.

Each :class:`~src.presets.Preset` is one LLM pass. A preset with no dependencies
feeds on the speaker-named transcript prompt; a preset with dependencies feeds on
its dependency outputs concatenated with a labeled separator. Independent presets
run concurrently via a :class:`ThreadPoolExecutor` capped at
``openai.max_parallel``; a preset is dispatched only once all of its dependencies
have completed successfully.

Partial failures are tolerated: a failed preset's results are recorded as errors,
its (transitive) dependents are skipped, and independent branches still run and
return their outputs. :func:`aggregate_error` builds a combined message that the
caller can raise after persisting the successful artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from src.config import Config
from src.openai_pipeline import OpenAIPipeline, build_prompt
from src.presets import Preset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresetResult:
    """Outcome of executing one preset.

    On success ``text`` holds the LLM output and ``error``/``skipped`` are unset.
    A preset whose own call raised has ``error`` set; a preset skipped because a
    dependency failed has ``skipped=True`` (and ``error`` naming the dependency).
    """

    name: str
    text: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.skipped


def _as_preset_map(presets: Mapping[str, Preset] | Iterable[Preset]) -> dict[str, Preset]:
    if isinstance(presets, Mapping):
        return dict(presets)
    return {preset.name: preset for preset in presets}


def _closure(preset_map: Mapping[str, Preset], only: Iterable[str] | None) -> set[str]:
    """Names to execute: ``only`` plus every transitive dependency, or all presets.

    A preset's dependencies must run to produce its input, so ``only`` is expanded
    to its dependency closure rather than executed in isolation.
    """
    if only is None:
        return set(preset_map)
    targets: set[str] = set()
    stack = list(only)
    while stack:
        name = stack.pop()
        if name in targets:
            continue
        if name not in preset_map:
            raise ValueError(f"unknown preset requested in only=: {name!r}")
        targets.add(name)
        stack.extend(preset_map[name].depends_on)
    return targets


def dependency_names(
    presets: Mapping[str, Preset] | Iterable[Preset], names: Iterable[str]
) -> set[str]:
    """Transitive dependencies of ``names``, excluding ``names`` themselves.

    Used by callers to discover which already-produced presets a retry would feed
    on, so their persisted artifacts can be passed back as ``precomputed`` instead
    of being re-run.
    """
    preset_map = _as_preset_map(presets)
    names = list(names)
    return _closure(preset_map, names) - set(names)


def _dependency_input(preset: Preset, results: Mapping[str, PresetResult]) -> str:
    """Concatenate dependency outputs with a labeled separator per dependency."""
    sections: list[str] = []
    for dep in preset.depends_on:
        text = results[dep].text.strip()
        sections.append(f"## {dep}\n\n{text}")
    return "\n\n".join(sections)


def _run_one(
    preset: Preset,
    *,
    transcript: str,
    file_name: str,
    config: Config,
    speaker_names: list[str] | None,
    dep_results: Mapping[str, PresetResult],
) -> PresetResult:
    if preset.depends_on:
        input_text = _dependency_input(preset, dep_results)
    else:
        input_text = build_prompt(transcript, file_name, speaker_names=speaker_names)

    use_batch = preset.batch if preset.batch is not None else config.openai_batch
    # batch_wait resolution: per-preset override wins, else the global default
    # (openai.batch_wait, itself defaulting to true). Only the synchronous
    # submit-and-wait path is implemented; an async pending-job state (batch_wait
    # false) is intentionally unsupported.
    batch_wait = (
        preset.batch_wait if preset.batch_wait is not None else config.openai_batch_wait
    )
    # batch_wait only governs batch submissions; a synchronous (non-batch) preset is
    # already inline, so batch_wait=false is a no-op there and must not error.
    if use_batch and not batch_wait:
        raise NotImplementedError(
            f"preset {preset.name!r}: batch_wait=false is not supported; the DAG "
            f"submits batch jobs and waits synchronously. Set batch_wait: true "
            f"(or leave it unset to inherit the default)."
        )

    pipeline = OpenAIPipeline(
        api_key=config.openai_api_key,
        model=preset.model or config.openai_model,
        proxy_url=config.proxy_url,
        use_batch=use_batch,
    )
    try:
        text, usage = pipeline.run(preset.instructions, input_text)
    finally:
        pipeline.close()
    return PresetResult(name=preset.name, text=text, usage=usage)


def run_presets(
    transcript: str,
    file_name: str,
    config: Config,
    presets: Mapping[str, Preset] | Iterable[Preset],
    *,
    speaker_names: list[str] | None = None,
    only: Iterable[str] | None = None,
    precomputed: Mapping[str, str] | None = None,
) -> dict[str, PresetResult]:
    """Execute the preset DAG and return a result per executed preset.

    Presets run in dependency order; independent presets are dispatched
    concurrently with up to ``config.openai_max_parallel`` workers. ``only``
    restricts execution to the named presets plus their transitive dependencies.
    ``precomputed`` supplies already-produced outputs (e.g. a persisted artifact
    from an earlier cycle) keyed by preset name: a dependency present there is
    reused as-is instead of being re-run, so a retry does not re-bill presets that
    already completed. Presets named in ``only`` always run even if ``precomputed``
    carries a stale output for them. Failures do not abort the run: the failed
    preset and its dependents are recorded (``error``/``skipped``) while
    independent branches still produce output. Use :func:`aggregate_error` on the
    returned mapping to surface a combined failure.
    """
    preset_map = _as_preset_map(presets)
    # Materialize ``only`` once: it is consumed by both ``_closure`` and the
    # ``only_set`` build below, and a one-shot iterable would otherwise be
    # exhausted by the first use, leaving ``only_set`` empty so ``precomputed``
    # could wrongly override an explicitly requested preset.
    only = None if only is None else list(only)
    targets = _closure(preset_map, only)
    only_set = set(only) if only is not None else set(preset_map)

    results: dict[str, PresetResult] = {}
    # Seed reusable dependency outputs so the DAG skips re-running them. Only
    # presets that are pure dependencies here (in the closure but not explicitly
    # requested) are seeded; an explicitly requested preset always re-runs.
    for name, text in (precomputed or {}).items():
        if name in targets and name not in only_set:
            results[name] = PresetResult(name=name, text=text)
    pending: set[str] = set(targets) - set(results)
    max_workers = max(1, config.openai_max_parallel)

    def failed_dep(name: str) -> str | None:
        for dep in preset_map[name].depends_on:
            res = results.get(dep)
            if res is not None and not res.ok:
                return dep
        return None

    def deps_ready(name: str) -> bool:
        return all(dep in results for dep in preset_map[name].depends_on)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {}

        def schedule() -> None:
            # Re-evaluate until stable so cascading skips (a skipped preset that
            # in turn blocks its own dependents) settle within one pass.
            changed = True
            while changed:
                changed = False
                for name in list(pending):
                    blocked_by = failed_dep(name)
                    if blocked_by is not None:
                        results[name] = PresetResult(
                            name=name,
                            error=f"skipped: dependency {blocked_by!r} did not complete",
                            skipped=True,
                        )
                        pending.discard(name)
                        changed = True
                    elif deps_ready(name):
                        fut = executor.submit(
                            _run_one,
                            preset_map[name],
                            transcript=transcript,
                            file_name=file_name,
                            config=config,
                            speaker_names=speaker_names,
                            dep_results={
                                dep: results[dep]
                                for dep in preset_map[name].depends_on
                            },
                        )
                        futures[fut] = name
                        pending.discard(name)
                        changed = True

        schedule()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                name = futures.pop(fut)
                try:
                    results[name] = fut.result()
                except Exception as exc:  # noqa: BLE001 - record, keep others running
                    logger.warning("Preset %s failed: %s", name, exc)
                    results[name] = PresetResult(name=name, error=str(exc))
            schedule()

    return results


def aggregate_error(results: Mapping[str, PresetResult]) -> str | None:
    """Combine failed/skipped presets into one message, or ``None`` if all ran."""
    failures = [res for res in results.values() if not res.ok]
    if not failures:
        return None
    parts = [f"{res.name}: {res.error}" for res in failures]
    return "preset DAG had failures - " + "; ".join(parts)
