import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """@brief Rede neural que aprende, ao mesmo tempo, "o que fazer" e
    "quão promissora é a situação atual".

    @details Imagine um jogador com dois talentos que compartilham o
    mesmo "cérebro" (as primeiras camadas da rede):

      * o **Ator** (Actor) decide **qual ação tomar** — ele devolve uma
        pontuação bruta por ação possível (6 ações do foguete);
      * o **Crítico** (Critic) **avalia a situação** — ele devolve um
        único número dizendo "quão bom é o estado em que estamos".

    As duas saídas nascem do mesmo tronco de duas camadas porque as
    características úteis para "decidir" e para "avaliar" costumam ser
    as mesmas. Treinar um tronco único é mais barato e generaliza
    melhor do que treinar duas redes separadas do zero.

    A inicialização dos pesos segue a receita clássica do PPO
    (Schulman et al. 2017). O detalhe mais importante é o **ganho
    pequeno (0.01) na última camada do Ator**: ele faz a política
    começar "em cima do muro", com probabilidade quase igual para
    todas as ações. Isso evita que o treino comece já viciado em uma
    ação só por causa de um sorteio ruim dos pesos iniciais.
    """

    def __init__(
        self,
        obs_dim: int = 9,
        n_actions: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        """@brief Monta as peças da rede.

        @param obs_dim Quantos números o ambiente entrega ao agente a
            cada passo (a "observação"). Ver DESIGN.md §5.
        @param n_actions Quantas ações discretas o foguete pode tomar
            (6, segundo DESIGN.md §3).
        @param hidden_dim Largura das duas camadas escondidas do tronco.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Tronco compartilhado: uma pequena MLP (perceptron multicamadas)
        # que converte a observação em uma representação interna.
        # Tanh é a ativação padrão em PPO: mantém valores em [-1, 1] e
        # tem gradientes bem comportados (nem explodem, nem somem).
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Cabeça do Ator: uma camada linear que devolve uma pontuação
        # bruta ("logit") por ação. Ainda NÃO é probabilidade — a
        # conversão para probabilidade (softmax) acontece depois, em
        # `select_action` / `evaluate_actions`.
        self.actor = nn.Linear(hidden_dim, n_actions)
        # Cabeça do Crítico: uma camada linear que devolve um único
        # número por amostra — a estimativa V(s) do valor do estado.
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """@brief Dá valores iniciais "saudáveis" aos pesos.

        @details A **inicialização ortogonal** sorteia vetores de peso
        mutuamente perpendiculares. Isso preserva a magnitude dos
        sinais ao longo das camadas (não explode nem desaparece). O
        parâmetro `gain` controla essa magnitude: sqrt(2) é o valor
        recomendado quando a camada é seguida de Tanh.
        """
        # Duas camadas do tronco: ganho sqrt(2) ≈ 1.414 (regra padrão
        # para ativação Tanh).
        for layer in (self.shared[0], self.shared[2]):
            nn.init.orthogonal_(layer.weight, gain=2.0 ** 0.5)
            nn.init.constant_(layer.bias, 0.0)
        # Ganho PEQUENO no Ator (0.01): os logits começam perto de
        # zero, o softmax fica ~uniforme, e a política inicial não tem
        # "vício" em nenhuma ação. É proposital.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        # Crítico com ganho 1.0: magnitude "normal", já que V(s) pode
        # assumir valores de ordem 1..dezenas nos retornos do PPO.
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """@brief Processa um lote de observações e devolve as duas
        cabeças.

        @param obs Tensor de forma `(B, obs_dim)` — um lote de B
            observações empilhadas.

        @return Uma dupla `(logits_das_ações, valor_do_estado)`:
            * `logits_das_ações`: `(B, n_actions)` — pontuação bruta por
              ação (o softmax converte em probabilidade depois);
            * `valor_do_estado`: `(B,)` — V(s) por amostra. O
              `squeeze(-1)` apenas remove o eixo "1" que a cabeça do
              crítico devolve (`(B, 1)` → `(B,)`).
        """
        hidden = self.shared(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    # -- Inferência ----------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """@brief Escolhe uma ação para cada observação do lote.

        @details O decorador `@torch.no_grad()` desliga o rastreamento
        de gradientes: durante a coleta de dados não vamos treinar,
        então não precisamos guardar nada para o backward — economiza
        memória e tempo.

        Com `deterministic=False` (padrão), amostramos da distribuição,
        o que dá **exploração** ao agente durante o treino. Com
        `deterministic=True`, pegamos o `argmax`, que é o que se quer
        em **avaliação**.

        @param obs Tensor `(B, obs_dim)`.
        @param deterministic `True` para escolher a ação mais provável
            (avaliação); `False` para amostrar (treino).
        @param generator Gerador de números aleatórios dedicado. Quando
            informado, a amostragem usa **esse** gerador em vez do RNG
            global — é o que DESIGN.md §12 exige para tornar os
            experimentos reprodutíveis sem efeitos colaterais globais.

        @return Trio `(ação, log_prob, valor)`:
            * `ação`: `(B,)`, índice inteiro em [0, n_actions);
            * `log_prob`: `(B,)`, log π(a|s) da ação efetivamente
              escolhida;
            * `valor`: `(B,)`, V(s) estimado pelo Crítico.
        """
        logits, value = self.forward(obs)
        # `log_softmax` já devolve log π(a|s) de forma numericamente
        # estável (truque do "log-sum-exp"). O PPO usa essa forma de
        # log-probabilidade direto na razão de verossimilhança
        # π_novo / π_antigo, então calculamos aqui de uma vez.
        log_probs_all = torch.log_softmax(logits, dim=-1)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            # `multinomial` amostra de uma distribuição categórica, mas
            # espera **probabilidades** (não logits) — daí o softmax
            # extra antes da chamada.
            probs = torch.softmax(logits, dim=-1)
            if generator is not None:
                action = torch.multinomial(
                    probs, 1, generator=generator
                ).squeeze(-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

        # `gather(-1, ...)` seleciona, em cada linha, o log_prob da
        # ação que foi efetivamente tomada. É o log π(a|s) que
        # queremos guardar na trajetória.
        log_prob = log_probs_all.gather(
            -1, action.unsqueeze(-1)
        ).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """@brief Reprocessa pares `(estado, ação)` antigos para a
        atualização do PPO.

        @details Depois de coletar uma trajetória, o PPO revisita cada
        par `(s, a)` **várias vezes** com os pesos atuais. Este method
        devolve as três quantidades que entram no objetivo do PPO
        (DESIGN.md §11):

          1. `log π(a|s)` — para montar a razão de verossimilhança
             `π_novo / π_antigo` que será "cortada" (clipping) se
             mudar demais;
          2. `V(s)` — para estimar a vantagem e o alvo do crítico;
          3. entropia da política — termo que incentiva **exploração**
             (impede a política de virar determinística cedo demais).

        @param obs Tensor `(B, obs_dim)`.
        @param actions Tensor `(B,)` com as ações que foram tomadas.

        @return `(log_prob, valor, entropia)`, todos com forma `(B,)`.
        """
        logits, value = self.forward(obs)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # Mesma ideia do `select_action`: pega o log_prob da ação
        # específica que o agente tomou no passo antigo.
        log_prob = log_probs_all.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        # Entropia de Shannon da distribuição categórica:
        #   H(π) = -Σ_a π(a|s) · log π(a|s)
        # Alta quando a política está "em dúvida" (bom para explorar),
        # baixa quando já está confiante em uma ação só.
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_prob, value, entropy