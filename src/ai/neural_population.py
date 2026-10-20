from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.ai.network import ActorCritic


class NeuralPopulation:
    """@brief Conjunto de redes neurais, uma por "indivíduo".

    @details Cada rede é uma política independente, instanciada com a
    arquitetura padrão de PPO (`ActorCritic`: obs_dim=9, n_actions=6,
    hidden_dim=128) e inicializada pela receita do próprio módulo
    (ortogonal, ganho 0.01 na cabeça do Ator). Os pesos são gravados
    de forma independente por indivíduo, via semente derivada da
    semente mestra.
    """

    def __init__(
        self,
        size: int,
        obs_dim: int = 9,
        n_actions: int = 6,
        hidden_dim: int = 128,
        seed: Optional[int] = 42,
    ) -> None:
        """@brief Gera `size` redes aleatórias.

        @param size Número de indivíduos (>= 1).
        @param obs_dim Dimensão da observação de cada rede.
        @param n_actions Número de ações discretas.
        @param hidden_dim Largura das camadas escondidas.
        @param seed Semente mestra. `None` desliga a reprodutibilidade
            (a inicialização passa a usar o RNG global do PyTorch).
        @throws ValueError Se `size < 1`.
        """
        if size < 1:
            raise ValueError("size deve ser >= 1.")

        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.seed = seed

        self._networks: list[ActorCritic] = []
        self._generate_initial()

    # ------------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------------

    def _generate_initial(self) -> None:
        """@brief Instancia os `size` indivíduos com sementes derivadas."""
        if self.seed is None:
            child_states = [None] * self.size
        else:
            child_states = np.random.SeedSequence(self.seed).spawn(self.size)

        for child in child_states:
            self._networks.append(self._make_network(child))

    def _make_network(self, child_state) -> ActorCritic:
        """@brief Instancia uma `ActorCritic` sob a semente `child_state`.

        @details A semente global do PyTorch é salva antes da
        inicialização e restaurada ao final. Assim a geração de cada
        indivíduo é determinística e independente da ordem de criação
        dos demais, e nenhum efeito colateral vaza para o resto do
        processo (ver `DESIGN.md §12`).
        """
        if child_state is None:
            net = ActorCritic(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dim=self.hidden_dim,
            )
            net.eval()
            return net

        int_seed = int(child_state.generate_state(1)[0])
        saved = torch.random.get_rng_state()
        try:
            torch.manual_seed(int_seed)
            net = ActorCritic(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                hidden_dim=self.hidden_dim,
            )
        finally:
            torch.random.set_rng_state(saved)

        net.eval()
        return net

    def _empty_like(self, seed: Optional[int]) -> "NeuralPopulation":
        """@brief População vazia com os mesmos metadados.

        @details Bypassa `__init__` (que geraria `size` redes aleatórias
        descartáveis). O chamador preenche `_networks`.
        """
        pop = NeuralPopulation.__new__(NeuralPopulation)
        pop.size = self.size
        pop.obs_dim = self.obs_dim
        pop.n_actions = self.n_actions
        pop.hidden_dim = self.hidden_dim
        pop.seed = seed
        pop._networks = []
        return pop

    def _clone(self, network: ActorCritic) -> ActorCritic:
        """@brief Cópia independente dos pesos (compartilhamento zero)."""
        clone = ActorCritic(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
        )
        clone.load_state_dict(network.state_dict())
        clone.eval()
        return clone

    def _crossover(
            self,
            parent_a: ActorCritic,
            parent_b: ActorCritic,
            generator: Optional[torch.Generator],
    ) -> ActorCritic:
        """@brief Crossover elemento-a-elemento nos tensores do `state_dict`.

        @details Para cada tensor, sorteia máscara Bernoulli(0.5) e
        escolhe, por elemento, entre pai A e pai B. Mantém forma e dtype
        dos parâmetros e dispensa achatar/reformatar o `state_dict`.
        """
        child = ActorCritic(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
        )
        state_a = parent_a.state_dict()
        state_b = parent_b.state_dict()
        new_state = {}
        for key, tensor_a in state_a.items():
            tensor_b = state_b[key]
            mask = (
                    torch.rand(tensor_a.shape, generator=generator) < 0.5
            ).to(tensor_a.dtype)
            new_state[key] = tensor_a * (1.0 - mask) + tensor_b * mask
        child.load_state_dict(new_state)
        child.eval()
        return child

    def _mutate(
            self,
            network: ActorCritic,
            generator: Optional[torch.Generator],
            rate: float,
            std: float,
    ) -> None:
        """@brief Mutação gaussiana in-place, por peso.

        @details Cada parâmetro recebe ruído N(0, std²) com probabilidade
        `rate`. `torch.no_grad()` evita empilhar autograd — a operação é
        puramente numérica.
        """
        if rate <= 0.0 or std <= 0.0:
            return
        with torch.no_grad():
            for param in network.parameters():
                noise = torch.randn(
                    param.shape,
                    generator=generator,
                    dtype=param.dtype,
                ) * std
                mask = (
                        torch.rand(param.shape, generator=generator) < rate
                ).to(param.dtype)
                param.add_(noise * mask)

    def evolve(
            self,
            fitness: Sequence[float],
            method: str = "sexual",
            elite_count: int = 2,
            mutation_rate: float = 0.05,
            mutation_std: float = 0.02,
            seed: Optional[int] = None,
    ) -> "NeuralPopulation":
        """@brief Gera a próxima população a partir de fitness por indivíduo.

        @details Os `elite_count` melhores sobrevivem **sem alteração**. O
        melhor par (rank 0 e rank 1) é usado como pais dos demais:
        "sexual" faz crossover elemento-a-elemento + mutação gaussiana;
        "asexual" clona o melhor e muta. O tamanho da nova população é
        preservado, assim como `obs_dim`, `n_actions` e `hidden_dim`.

        @param fitness Lista de fitness (maior = melhor), na mesma ordem
            de `self._networks`. Deve ter o mesmo comprimento.
        @param method "sexual" (recombinação) ou "asexual" (clone).
        @param elite_count Quantos dos melhores são preservados
            intactos. `0` = pura descendência; `size` = pura elite.
        @param mutation_rate Probabilidade de mutar cada peso.
        @param mutation_std Desvio-padrão do ruído gaussiano.
        @param seed Semente do gerador de mutação/crossover. `None` usa
            o RNG global do PyTorch.
        @return Nova `NeuralPopulation`, do mesmo tamanho.
        @throws ValueError Se `fitness` tiver tamanho diferente, se
            `method` for desconhecido, ou se `elite_count` sair de
            `[0, size]`.
        """
        if len(fitness) != len(self._networks):
            raise ValueError(
                f"fitness tem {len(fitness)} valores, população tem "
                f"{len(self._networks)} indivíduos."
            )
        if method not in ("sexual", "asexual"):
            raise ValueError(f"método inválido: {method!r}")
        if not 0 <= elite_count <= self.size:
            raise ValueError(
                f"elite_count fora de [0, {self.size}]: {elite_count}"
            )

        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        # Ranking decrescente; `sorted` é estável → empates preservam ordem.
        ranked = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)

        parent_a = self._networks[ranked[0]]
        # Sexual exige dois pais distintos; cai para assexual se size == 1.
        can_sexual = method == "sexual" and len(ranked) >= 2
        parent_b = self._networks[ranked[1]] if can_sexual else parent_a

        child_pop = self._empty_like(seed)

        # Elites: clones intactos.
        for idx in ranked[:elite_count]:
            child_pop._networks.append(self._clone(self._networks[idx]))

        # Descendência.
        while len(child_pop._networks) < self.size:
            if can_sexual:
                child = self._crossover(parent_a, parent_b, generator)
            else:
                child = self._clone(parent_a)
            self._mutate(child, generator, mutation_rate, mutation_std)
            child_pop._networks.append(child)

        return child_pop

    # ------------------------------------------------------------------
    # Interface de contêiner
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._networks)

    def __iter__(self) -> Iterator[ActorCritic]:
        return iter(self._networks)

    def __getitem__(self, index: int) -> ActorCritic:
        return self._networks[index]

    def __setitem__(self, index: int, network: ActorCritic) -> None:
        self._networks[index] = network

    def add(self, network: ActorCritic) -> int:
        """@brief Adiciona `network` ao final e devolve seu índice."""
        self._networks.append(network)
        return len(self._networks) - 1

    # ------------------------------------------------------------------
    # Persistência (útil para checkpoints e debug)
    # ------------------------------------------------------------------

    def save_all(self, directory: Path) -> list[Path]:
        """@brief Salva cada indivíduo em `directory/individual_NNNN.pt`.

        @details Usa `state_dict()` — o mesmo formato consumido por
        `Experimenter.load_network`, então um arquivo individual pode
        ser passado diretamente a `--model` para inspeção visual.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for i, net in enumerate(self._networks):
            path = directory / f"individual_{i:04d}.pt"
            torch.save(net.state_dict(), path)
            paths.append(path)
        return paths