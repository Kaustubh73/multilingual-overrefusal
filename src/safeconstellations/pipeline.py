"""End-to-end SafeConstellations toy pipeline orchestration.

Wiring (mirrors the paper):

    load hidden states  ->  fit Task Embeddings Store (refusal/target centroids)
                        ->  detect task + conditionally steer at inference
                        ->  evaluate OR vs harmful-refusal, baseline vs steered
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.safeconstellations.config import HiddenStateConfig, SteeringConfig
from src.safeconstellations.data import Example, split_examples, toy_dataset
from src.safeconstellations.evaluate import EvalSummary, evaluate
from src.safeconstellations.hidden_states import build_backend
from src.safeconstellations.steering import SteeringController
from src.safeconstellations.task_store import TaskEmbeddingsStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    store: TaskEmbeddingsStore
    summary: EvalSummary


def run_pipeline(
    *,
    examples: list[Example] | None = None,
    hidden_config: HiddenStateConfig | None = None,
    steering_config: SteeringConfig | None = None,
) -> PipelineResult:
    """Fit the store on train, steer + evaluate on test, return artifacts."""
    examples = examples or toy_dataset()
    hidden_config = hidden_config or HiddenStateConfig()
    steering_config = steering_config or SteeringConfig()

    tasks = tuple(sorted({e.task for e in examples}))
    backend = build_backend(hidden_config, tasks=tasks)

    train = split_examples(examples, "train")
    test = split_examples(examples, "test")
    logger.info("Pipeline: %d train / %d test examples over tasks %s", len(train), len(test), tasks)

    store = TaskEmbeddingsStore.fit(train, backend, cfg=hidden_config.toy)
    controller = SteeringController(store, steering_config)
    summary = evaluate(test, backend, controller)
    return PipelineResult(store=store, summary=summary)


def format_report(result: PipelineResult) -> str:
    """Human-readable summary of baseline vs steered OR / harmful-refusal rates."""
    s = result.summary
    bor_c, bor_n = s.baseline_or
    sor_c, sor_n = s.steered_or
    bhr_c, bhr_n = s.baseline_hr
    shr_c, shr_n = s.steered_hr

    def pct(c: int, n: int) -> str:
        return f"{(c / n) if n else 0.0:.0%} ({c}/{n})"

    lines = [
        "SafeConstellations toy pipeline — results",
        "=" * 44,
        f"Fitted tasks         : {', '.join(result.store.tasks)}",
        f"Steered layers (top-K): {result.store.top_k_layers}",
        f"Detection layer       : {result.store.detection_layer}",
        "",
        "Over-refusal (benign tasks — lower is better):",
        f"  baseline OR : {pct(bor_c, bor_n)}",
        f"  steered  OR : {pct(sor_c, sor_n)}",
        "",
        "Harmful-refusal (truly harmful — should stay high):",
        f"  baseline HR : {pct(bhr_c, bhr_n)}",
        f"  steered  HR : {pct(shr_c, shr_n)}",
        "",
        "Per-row detail (task | benign | detected@conf | steered | base->steer):",
    ]
    for r in s.rows:
        lines.append(
            f"  {r.sample_id:<10} {r.task:<11} benign={str(r.is_benign_task):<5} "
            f"det={r.detected_task}@{r.confidence:.2f} steered={str(r.steered):<5} "
            f"{'REFUSE' if r.baseline_refused else 'answer'} -> "
            f"{'REFUSE' if r.steered_refused else 'answer'}"
        )
    return "\n".join(lines)
