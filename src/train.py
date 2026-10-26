from src.ai.train_phase_1 import UndockingTrainer
from src.ai.neural_population import NeuralPopulation

import argparse
from src.config import Configs, SimulationConfig
from pathlib import Path
import torch
import random

def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1 — Desacoplagem.")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--episodes-per-gen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    config = Configs(
        width=1500, height=1000,
        title="Gravitation",
        background_path=Path(__file__).parent / "assets" / "sky.jpg",
        max_fps=60,
    )
    sim_config = SimulationConfig()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pop = NeuralPopulation(size=args.population, seed=args.seed)

    app = UndockingTrainer(
        config, sim_config,
        max_generations=args.generations,
        episodes_per_gen=args.episodes_per_gen,
        method="sexual",
        elite_count=max(2, args.population // 20),
        mutation_rate_start=0.05,
        mutation_rate_end=0.01,
        mutation_std=0.03,
        base_seed=args.seed,
        output_dir=args.output,
    )
    app.start_training(pop)

    print(f"[log] métricas -> {app.log_path}")
    app.run()
    print(f"[done] log final: {app.log_path}")


if __name__ == "__main__":
    main()