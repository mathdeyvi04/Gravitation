"""Captura periódica de snapshots do tracemalloc.

O `tracemalloc` já sabe gravar e ler snapshots (`Snapshot.dump` /
`Snapshot.load`). Este módulo organiza *quando* capturar — em
intervalos regulares, com limite opcional — e *com que metadados*,
para que o script de análise correlacione alocações a eventos da
simulação.
"""

import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional


class MemoryMonitor:
    """Grava snapshots rotulados do tracemalloc automaticamente.

    Cada `capture()` produz dois arquivos com o mesmo nome-base:

    - `<timestamp>_<label>.snap` — snapshot binário do tracemalloc,
      consumido por `tracemalloc.Snapshot.load` no script de análise.
    - `<timestamp>_<label>.json` — rótulo, hora, profundidade de
      traceback e `context` arbitrário passado pelo chamador.

    O rastreamento e o relógio interno são controlados pelo monitor:
    `should_capture()` diz se o intervalo venceu, e `capture()` grava
    e agenda o próximo. O chamador só precisa consultar uma vez por
    frame.
    """

    __slots__ = (
        "dump_dir",
        "traceback_depth",
        "interval",
        "max_dumps",
        "_counter",
        "_next_capture_at",
    )

    def __init__(
            self,
            dump_dir: str | Path = "mem_dumps",
            traceback_depth: int = 25,
            interval: float = 10.0,
            max_dumps: Optional[int] = None,
            start_immediately: bool = True,
    ) -> None:
        if traceback_depth < 1:
            raise ValueError("traceback_depth deve ser >= 1.")
        if interval <= 0.0:
            raise ValueError("interval deve ser > 0.")
        if max_dumps is not None and max_dumps < 1:
            raise ValueError("max_dumps deve ser >= 1 ou None.")

        # Cada execução ganha a sua própria subpasta com carimbo de início.
        # Isso evita que snapshots de sessões distintas se misturem no
        # diretório raiz e poluam a análise posterior.
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dump_dir = Path(dump_dir) / f"session_{session_stamp}"
        self.dump_dir.mkdir(parents=True, exist_ok=True)

        self.traceback_depth = traceback_depth
        self.interval = float(interval)
        self.max_dumps = max_dumps
        self._counter = 0
        self._next_capture_at = 0.0

        if start_immediately:
            self.start()

    @property
    def active(self) -> bool:
        return tracemalloc.is_tracing()

    @property
    def session_dir(self) -> Path:
        """Subpasta onde os snapshots desta execução estão sendo gravados."""
        return self.dump_dir

    @property
    def count(self) -> int:
        return self._counter

    @property
    def exhausted(self) -> bool:
        return self.max_dumps is not None and self._counter >= self.max_dumps

    def start(self) -> None:
        """Liga o tracemalloc e agenda a primeira captura para já."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.traceback_depth)
        self._next_capture_at = 0.0

    def stop(self) -> None:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def should_capture(self) -> bool:
        """True quando o intervalo venceu e ainda há cota disponível."""
        if self.exhausted:
            return False
        return perf_counter() >= self._next_capture_at

    def capture(
        self,
        label: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Grava um snapshot e agenda o próximo.

        Se o rastreamento estiver desligado, liga antes de capturar —
        o primeiro `capture()` sempre produz dados úteis como baseline.
        """
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.traceback_depth)

        snapshot = tracemalloc.take_snapshot()

        self._counter += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = label or f"snap{self._counter:03d}"
        base = f"{stamp}_{tag}"

        self.dump_dir.mkdir(parents=True, exist_ok=True)
        snap_path = self.dump_dir / f"{base}.snap"
        meta_path = self.dump_dir / f"{base}.json"

        snapshot.dump(str(snap_path))
        meta_path.write_text(
            json.dumps(
                {
                    "label": tag,
                    "timestamp": stamp,
                    "traceback_depth": self.traceback_depth,
                    "context": context or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self._next_capture_at = perf_counter() + self.interval
        return snap_path