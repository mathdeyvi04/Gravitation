"""@brief Analisa snapshots do tracemalloc gravados pelo `MemoryMonitor`.

@details Uso:

    python tools/mem_report.py mem_dumps/            # gráfico + resumo
    python tools/mem_report.py mem_dumps/ --save fig.png
    python tools/mem_report.py mem_dumps/ --objs     # plota nº de objetos
"""

import argparse
import json
import sys
import tracemalloc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def resolve_dump_dir(path: Path) -> Path:
    """@brief Resolve o diretório efetivo de snapshots.

    @details Aceita tanto um diretório já apontando para uma sessão
    específica (contém `.snap` direto) quanto o diretório-raiz que
    contém várias `session_*`. Neste último caso, escolhe a mais
    recente por ordem alfabética — que coincide com ordem cronológica
    dado o carimbo `YYYYMMDD_HHMMSS`.

    @param path Diretório passado pelo usuário.
    @return Diretório da sessão a analisar.
    """
    # Sessão já resolvida: `.snap` presentes no próprio diretório.
    if list(path.glob("*.snap")):
        return path

    sessions = sorted(
        (p for p in path.iterdir() if p.is_dir() and p.name.startswith("session_")),
        reverse=True,
    )
    if sessions:
        print(f"usando sessão mais recente: {sessions[0].name}")
        return sessions[0]
    return path


def load_entries(dump_dir: Path):
    """@brief Carrega pares `(.snap, .json)` de uma sessão.

    @details O `.json` é opcional: snapshots gravados manualmente sem
    metadados entram com `meta = {}` e o label cai para `#<índice>`.

    @param dump_dir Diretório da sessão.
    @return Lista de dicionários `{"path", "meta"}`.
    """
    entries = []
    for snap_path in sorted(dump_dir.glob("*.snap")):
        meta_path = snap_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entries.append({"path": snap_path, "meta": meta})
    return entries


def total_size_and_count(snap_path: Path) -> tuple[int, int]:
    """@brief Soma tamanho e número de objetos de um snapshot.

    @details Agrega as estatísticas por arquivo — o nível mais grosso
    do `tracemalloc`, adequado para tendência ao longo do tempo.

    @param snap_path Caminho do `.snap` a carregar.
    @return `(bytes_totais, número_de_objetos)`.
    """
    snap = tracemalloc.Snapshot.load(str(snap_path))
    stats = snap.statistics("filename")
    return sum(s.size for s in stats), sum(s.count for s in stats)


def collect(entries):
    """@brief Consolida entradas em linhas prontas para plotagem.

    @param entries Lista `{"path", "meta"}` produzida por `load_entries`.
    @return Lista de dicionários com `index`, `label`, `sim_time`,
        `size_mb`, `count` e `context`.
    """
    rows = []
    for i, entry in enumerate(entries):
        size, count = total_size_and_count(entry["path"])
        ctx = entry["meta"].get("context", {})
        meta = entry["meta"]

        sim_time = ctx.get("total_time")
        rows.append({
            "index": i,
            "label": meta.get("label", f"#{i}"),
            "sim_time": float(sim_time) if sim_time is not None else None,
            "size_mb": size / 1024 / 1024,
            "count": count,
            "context": ctx,
        })
    return rows


def _annotate_curve(ax, xs, ys, labels):
    """@brief Anota cada ponto, escolhendo o lado oposto à inclinação local.

    @details Se a curva sobe, o rótulo desce; se desce, sobe. Assim o
    texto ocupa a região vazia formada pela própria curva, em vez de
    sobrepô-la.

    @param ax Eixos do matplotlib onde anotar.
    @param xs Coordenadas X dos pontos.
    @param ys Coordenadas Y dos pontos.
    @param labels Rótulos correspondentes.
    """
    n = len(ys)
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        # Inclinação local: diferença central no interior; diferenças
        # laterais nas pontas. Sinal decide para onde empurrar o rótulo.
        if n == 1:
            slope = 0.0
        elif i == 0:
            slope = ys[1] - ys[0]
        elif i == n - 1:
            slope = ys[-1] - ys[-2]
        else:
            slope = ys[i + 1] - ys[i - 1]

        if slope > 0:
            offset, va = (7, -14), "top"
        else:
            offset, va = (7, 12), "bottom"

        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=offset,
            fontsize=8, color="#555", va=va,
        )


def plot(rows, save_path: Path | None):
    """@brief Plota memória e objetos ao longo do tempo.

    @details Usa `sim_time` como eixo X quando *todos* os snapshots têm
    esse metadado; caso contrário cai para o índice de captura, para
    não misturar duas escalas na mesma curva.

    @param rows Linhas produzidas por `collect`.
    @param save_path Caminho do PNG de saída; `None` abre a janela.
    """
    xs, labels = [], []
    sizes_mb, counts = [], []
    use_sim_time = all(r["sim_time"] is not None for r in rows)

    for r in rows:
        xs.append(r["sim_time"] if use_sim_time else r["index"])
        sizes_mb.append(r["size_mb"])
        counts.append(r["count"])
        labels.append(r["label"])

    fig, (ax_mem, ax_obj) = plt.subplots(
        2, 1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    # --- Painel superior: memória ---
    ax_mem.plot(xs, sizes_mb, marker="o", color="#c0392b",
                linewidth=2, markersize=6, label="Memória rastreada")
    _annotate_curve(ax_mem, xs, sizes_mb, labels)
    ax_mem.set_ylabel("Memória rastreada (MB)")
    ax_mem.set_title("Evolução da memória rastreada pelo tracemalloc")
    ax_mem.grid(True, alpha=0.3)

    # --- Painel inferior: objetos ---
    ax_obj.plot(xs, counts, marker="o", color="#2c6fbb",
                linewidth=2, markersize=6, label="Objetos rastreados")
    _annotate_curve(ax_obj, xs, counts, labels)
    ax_obj.set_ylabel("Objetos rastreados")
    ax_obj.set_title("Evolução do número de objetos rastreados")
    ax_obj.set_xlabel(
        "Tempo de simulação (s)" if use_sim_time else "Ordem do snapshot"
    )
    ax_obj.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"gráfico salvo em {save_path}")
    else:
        plt.show()


def main():
    """@brief Ponto de entrada: parseia argumentos e gera o relatório.

    @details Resolve o diretório da sessão, carrega os snapshots,
    consolida as métricas e plota (ou salva) o gráfico.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument(
        "--save", type=Path, metavar="ARQUIVO",
        help="Salva os gráficos em disco em vez de abrir a janela.",
    )
    args = parser.parse_args()

    if not args.dump_dir.is_dir():
        sys.exit(f"diretório não encontrado: {args.dump_dir}")

    session_dir = resolve_dump_dir(args.dump_dir)
    entries = load_entries(session_dir)
    if not entries:
        sys.exit(f"nenhum .snap em {session_dir}")

    rows = collect(entries)
    plot(rows, save_path=args.save)


if __name__ == "__main__":
    main()