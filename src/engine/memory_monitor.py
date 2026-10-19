"""@brief Captura periódica de snapshots do tracemalloc.

@details O `tracemalloc` já sabe gravar e ler snapshots
(`Snapshot.dump` / `Snapshot.load`). Este módulo organiza *quando*
capturar — em intervalos regulares, com limite opcional — e *com que
metadados*, para que o script de análise correlacione alocações a
eventos da simulação.
"""

import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional


class MemoryMonitor:
    """@brief Grava snapshots rotulados do tracemalloc automaticamente.

    @details Cada `capture()` produz dois arquivos com o mesmo nome-base:

    - `<timestamp>_<label>.snap`: snapshot binário do tracemalloc,
      consumido por `tracemalloc.Snapshot.load` no script de análise.
    - `<timestamp>_<label>.json`: rótulo, hora, profundidade de
      traceback e `context` arbitrário passado pelo chamador.

    O rastreamento e o relógio interno são controlados pelo monitor:
    `should_capture()` diz se o intervalo venceu, e `capture()` grava e
    agenda o próximo. O chamador só precisa consultar uma vez por frame.
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
        """@brief Configura o monitor e (opcionalmente) liga o rastreamento.

        @details Cria uma subpasta `session_<timestamp>` dentro de
        `dump_dir` para isolar os snapshots desta execução — assim
        sessões distintas não se misturam no diretório raiz.

        @param dump_dir Diretório raiz onde a subpasta da sessão é
            criada.
        @param traceback_depth Frames de traceback por alocação.
        @param interval Intervalo entre capturas automáticas, em
            segundos.
        @param max_dumps Limite de capturas por sessão; `None` desliga.
        @param start_immediately Liga o `tracemalloc` já no construtor.
        """
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
        """@brief Indica se o `tracemalloc` está rastreando agora.

        @return `True` se `tracemalloc.is_tracing()` for verdadeiro.
        """
        return tracemalloc.is_tracing()

    @property
    def session_dir(self) -> Path:
        """@brief Subpasta onde os snapshots desta execução são gravados.

        @return Caminho `dump_dir/session_<timestamp>`.
        """
        return self.dump_dir

    @property
    def count(self) -> int:
        """@brief Número de capturas já realizadas nesta sessão.

        @return Contador interno de `capture()`.
        """
        return self._counter

    @property
    def exhausted(self) -> bool:
        """@brief Indica se a cota de capturas já foi atingida.

        @details `False` quando `max_dumps` é `None`.

        @return `True` se `_counter >= max_dumps`.
        """
        return self.max_dumps is not None and self._counter >= self.max_dumps

    def start(self) -> None:
        """@brief Liga o tracemalloc e agenda a primeira captura para já."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.traceback_depth)
        # `_next_capture_at = 0` faz o próximo `should_capture()` ser
        # imediatamente `True` — útil para capturar um baseline.
        self._next_capture_at = 0.0

    def stop(self) -> None:
        """@brief Desliga o tracemalloc, se estiver rastreando."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def should_capture(self) -> bool:
        """@brief Indica se já é hora de capturar outro snapshot.

        @details `True` quando o intervalo venceu e ainda há cota
        disponível.

        @return `True` se `capture()` deve ser chamado agora.
        """
        if self.exhausted:
            return False
        return perf_counter() >= self._next_capture_at

    def capture(
        self,
        label: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> Path:
        """@brief Grava um snapshot e agenda o próximo.

        @details Se o rastreamento estiver desligado, liga antes de
        capturar — o primeiro `capture()` sempre produz dados úteis
        como baseline.

        @param label Rótulo usado no nome do arquivo; se vazio, usa
            `snap<NNN>` com o contador.
        @param context Metadados arbitrários salvos no `.json` ao lado
            do snapshot.
        @return Caminho do `.snap` gravado.
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

        # Agenda o próximo `should_capture()` para `interval` segundos
        # depois do *fim* desta captura — capturas pesadas não aceleram
        # o relógio artificialmente.
        self._next_capture_at = perf_counter() + self.interval
        return snap_path