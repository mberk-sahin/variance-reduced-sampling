import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

@dataclass
class CUDAMemoryLogger:
    """
    Logs CUDA memory stats for named blocks at a chosen interval.

    Records format (each entry):
      {
        "step": int,
        "t": float|None,
        "baseline": {...},
        "<block_name>": {"delta_alloc_mib": ..., "peak_mib": ..., "after_alloc_mib": ...},
        "end": {...}
      }
    """
    interval: int = 0                    # 0 disables logging
    enabled: bool = True
    step: int = 0
    records: List[Dict[str, Any]] = field(default_factory=list)

    # internal per-step state
    _active: bool = False
    _entry: Optional[Dict[str, Any]] = None
    _dev: Optional[int] = None

    @staticmethod
    def _mib(x_bytes: int) -> float:
        return float(x_bytes) / (1024.0 ** 2)

    def reset(self) -> None:
        self.step = 0
        self.records.clear()
        self._active = False
        self._entry = None
        self._dev = None

    def _should_log(self) -> bool:
        return self.enabled and self.interval is not None and self.interval > 0 and (self.step % self.interval == 0)

    def _to_dev_index(self, device: torch.device) -> int:
        if device.type != "cuda":
            raise ValueError("CUDAMemoryLogger requires CUDA tensors.")
        return device.index if device.index is not None else torch.cuda.current_device()

    def _snapshot(self) -> Dict[str, float]:
        torch.cuda.synchronize(self._dev)
        return {
            "alloc_mib": self._mib(torch.cuda.memory_allocated(self._dev)),
            "reserved_mib": self._mib(torch.cuda.memory_reserved(self._dev)),
            "peak_mib": self._mib(torch.cuda.max_memory_allocated(self._dev)),
        }

    def begin_step(self, x: torch.Tensor, t: Any = None) -> bool:
        """
        Call once per algorithm iteration (e.g., at start of drift).
        Returns True if logging is active for this step.
        """
        self.step += 1
        if not (x.is_cuda and x.device.type == "cuda" and self._should_log()):
            self._active = False
            self._entry = None
            self._dev = None
            return False

        self._dev = self._to_dev_index(x.device)
        torch.cuda.synchronize(self._dev)
        torch.cuda.reset_peak_memory_stats(self._dev)

        # create new record
        entry = {"step": int(self.step) - 1} # follows the 0 index starting convention

        # store t as a python float if possible
        if torch.is_tensor(t):
            entry["t"] = float(t.detach().cpu().item())
        else:
            entry["t"] = float(t)

        entry["baseline"] = self._snapshot()

        self._active = True
        self._entry = entry
        return True

    @contextmanager
    def measure_block(self, name: str):
        """
        Context manager to measure a named block within the active step.
        """
        if not self._active:
            yield
            return

        torch.cuda.synchronize(self._dev)
        before_alloc = torch.cuda.memory_allocated(self._dev)
        torch.cuda.reset_peak_memory_stats(self._dev)

        yield # wait for calculation to occur

        torch.cuda.synchronize(self._dev)
        after_alloc = torch.cuda.memory_allocated(self._dev)
        peak = torch.cuda.max_memory_allocated(self._dev)

        self._entry[name] = {
            "delta_alloc_mib": self._mib(after_alloc - before_alloc),
            "peak_mib": self._mib(peak),
            "after_alloc_mib": self._mib(after_alloc),
        }

    def end_step(self) -> None:
        """
        Call once at the end of the iteration (e.g., end of drift).
        """
        if not self._active:
            return
        self._entry["end"] = self._snapshot()
        self.records.append(self._entry)

        self._active = False
        self._entry = None
        self._dev = None
