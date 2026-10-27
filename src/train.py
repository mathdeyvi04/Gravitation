"""Entry point unificado para o treinamento das fases.

Uso:
    python -m src.train --phase 1
    python -m src.train --phase 2
    python -m src.train --phase 2 --checkpoint custom/path.pt
"""

import argparse
import random
from pathlib import Path

import torch

from src.config import Configs, SimulationConfig
from src.ai.neural_population import NeuralPopulation
from src.ai.train_phase_1 import UndockingTrainer
from src.ai.train_phase_2 import OrbitRaiseTrainer, load_population_from_phase1


# Registro das fases. A Fase 2 é warm-start da melhor rede da Fase 1.
PHASE_REGISTRY = {
    1: {
        "name": "Desacoplagem",
        "cls": UndockingTrainer,
        "generations": 50,
        "episodes_per_gen": 3,
        "mutation_rate_start": 0.05,
        "warm_start_from": None,
    },
    2: {
        "name": "Órbita maior",
        "cls": OrbitRaiseTrainer,
        "generations": 60,
        "episodes_per_gen": 3,
        # Fine-tuning começa com mutação mais baixa que a Fase 1.
        "mutation_rate_start": 0.025,
        "warm_start_from": "phase1_best.pt",
    },
}

# Constantes de treino (edite aqui para mudar o pipeline inteiro).
POPULATION_SIZE = 100
ELITE_FRACTION = 20        # 1/20 da população
MUTATION_RATE_END = 0.01
MUTATION_STD = 0.03
BASE_SEED = 42
DIVERSITY_STD = 0.02       # ruído dos clones não-elite no warm-start
OUTPUT_DIR = Path("checkpoints")


def _build_population(args, phase_cfg) -> NeuralPopulation:
    """Cria a população aleatória (Fase 1) ou a população warm-start."""
    if phase_cfg["warm_start_from"] is None:
        if args.checkpoint is not None:
            print(
                f"[warn] fase {args.phase} não usa warm-start — "
                "ignorando --checkpoint"
            )
        return NeuralPopulation(size=POPULATION_SIZE, seed=BASE_SEED)

    checkpoint = args.checkpoint or (OUTPUT_DIR / phase_cfg["warm_start_from"])
    if not checkpoint.exists():
        raise SystemExit(
            f"[erro] checkpoint não encontrado: {checkpoint}\n"
            f"        rode a fase {args.phase - 1} primeiro, ou passe "
            "--checkpoint CAMINHO."
        )

    print(f"[load] warm-start de {checkpoint}")
    return load_population_from_phase1(
        checkpoint,
        size=POPULATION_SIZE,
        seed=BASE_SEED,
        elite_count=max(3, POPULATION_SIZE // ELITE_FRACTION),
        diversity_std=DIVERSITY_STD,
    )


def main() -> None:
    phase_help = "Fases disponíveis:\n" + "\n".join(
        f"  {key}: {value['name']}" for key, value in PHASE_REGISTRY.items()
    )
    parser = argparse.ArgumentParser(
        description="Treinamento das fases do piloto autônomo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=phase_help,
    )
    parser.add_argument(
        "--phase", type=int, required=True,
        choices=sorted(PHASE_REGISTRY.keys()),
        help="Fase a treinar.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint de warm-start (fases 2+). Default: "
             "checkpoints/phase<N-1>_best.pt",
    )
    args = parser.parse_args()
    phase_cfg = PHASE_REGISTRY[args.phase]

    config = Configs(
        width=1500,
        height=1000,
        title=f"Gravitation — Fase {args.phase} ({phase_cfg['name']})",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    sim_config = SimulationConfig()

    torch.manual_seed(BASE_SEED)
    random.seed(BASE_SEED)
    population = _build_population(args, phase_cfg)

    app = phase_cfg["cls"](
        config,
        sim_config,
        max_generations=phase_cfg["generations"],
        episodes_per_gen=phase_cfg["episodes_per_gen"],
        method="sexual",
        elite_count=max(2, POPULATION_SIZE // ELITE_FRACTION),
        mutation_rate_start=phase_cfg["mutation_rate_start"],
        mutation_rate_end=MUTATION_RATE_END,
        mutation_std=MUTATION_STD,
        base_seed=BASE_SEED,
        output_dir=OUTPUT_DIR,
    )
    app.start_training(population)

    print(f"[log] métricas -> {app.log_path}")
    app.run()
    print(f"[done] log final: {app.log_path}")


if __name__ == "__main__":
    main()
