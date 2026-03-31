"""Runtime telemetry collection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time


@dataclass
class LayerMetric:
    """Aggregated timing for one logical layer."""

    calls: int = 0
    total_seconds: float = 0.0


@dataclass
class TelemetryRecorder:
    """Collect lightweight generation telemetry."""

    prompt_tokens: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    first_token_seconds: float | None = None
    generated_tokens: int = 0
    logical_bytes_mapped: int = 0
    tensors_touched: set[str] = field(default_factory=set)
    layer_metrics: dict[str, LayerMetric] = field(default_factory=dict)

    def record_tensor_map(self, tensor_name: str, nbytes: int) -> None:
        self.logical_bytes_mapped += int(nbytes)
        self.tensors_touched.add(tensor_name)

    def record_layer_time(self, layer_name: str, elapsed_seconds: float) -> None:
        metric = self.layer_metrics.setdefault(layer_name, LayerMetric())
        metric.calls += 1
        metric.total_seconds += elapsed_seconds

    def mark_first_token(self) -> None:
        if self.first_token_seconds is None:
            self.first_token_seconds = time.perf_counter() - self.started_at

    def record_generated_token(self) -> None:
        self.generated_tokens += 1

    @contextmanager
    def time_layer(self, layer_name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_layer_time(layer_name, time.perf_counter() - started)

    def summary(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at
        tps = self.generated_tokens / elapsed if elapsed > 0 else 0.0
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "elapsed_seconds": elapsed,
            "first_token_seconds": self.first_token_seconds,
            "tokens_per_second": tps,
            "logical_bytes_mapped": self.logical_bytes_mapped,
            "tensors_touched": len(self.tensors_touched),
            "layer_times": {
                name: {
                    "calls": metric.calls,
                    "total_seconds": metric.total_seconds,
                    "avg_seconds": metric.total_seconds / metric.calls if metric.calls else 0.0,
                }
                for name, metric in sorted(self.layer_metrics.items())
            },
        }
