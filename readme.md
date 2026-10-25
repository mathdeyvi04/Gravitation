# Piloto Autônomo de Inserção Orbital

## Sumário

1. [Descrição do Projeto Completo](#descrição-do-projeto-completo)
   - [Visão geral](#visão-geral)
   - [Arquitetura de módulos](#arquitetura-de-módulos)
   - [Física e simulação](#física-e-simulação)
   - [Foguete e controle](#foguete-e-controle)
   - [Agente (rede neural)](#agente-rede-neural)
   - [Função de recompensa](#função-de-recompensa)
   - [Treinamento (algoritmo genético)](#treinamento-algoritmo-genético)
   - [Ambiente de teste (`Gravitation`)](#ambiente-de-teste-gravitation)
   - [Logs e análise](#logs-e-análise)
2. [Como executar](#como-executar)
   - [Requisitos](#requisitos)
   - [Simulação manual](#simulação-manual)
   - [Simulação com modelo treinado](#simulação-com-modelo-treinado)
   - [Treinamento](#treinamento)
   - [Análise de logs](#análise-de-logs)
   - [Monitor de memória](#monitor-de-memória)
3. [Resultados](#resultados)

---

## Descrição do Projeto Completo

### Visão geral

Simulador gravitacional 2D construído sobre Pygame, com física newtoniana de **N corpos** integrada numericamente em passos fixos, e um **foguete controlável** por jogador ou por rede neural. O objetivo do agente é **partir do repouso acoplado à superfície de um corpo celeste** (decolar), **inserir-se em órbita estável** ao redor desse corpo com **mínimo de combustível e tempo**, e ser capaz de **pousar** em outros hospedeiros.

O alvo orbital é parametrizado por `α = r_ref / R_body` (raio da órbita-alvo em múltiplos do raio do corpo), com `α` padrão 2.5 e faixa válida `[1.5, 8.0]`. Toda a observação do agente é normalizada por quantidades naturais dessa solução, o que confere **invariância de escala** e permite que a mesma política funcione para qualquer α dentro do range — desde que treinada nesse intervalo.

A política é uma **rede neural `ActorCritic`** (MLP de duas camadas de 128, ativação `tanh`) que emite 6 ações discretas. O **treinamento é por algoritmo genético** (seleção por aptidão, crossover elemento-a-elemento e mutação gaussiana) sobre uma população de redes. As decisões do design estão documentadas em `src/ai/DESIGN.md`, que serve como contrato entre as fases (espaço de ações, observação, recompensa, terminais, física e hiperparâmetros).

### Arquitetura de módulos

```
src/
├── config.py                     # Configs (janela/loop) e SimulationConfig (física)
├── main.py                       # Gravitation: simulação N-corpos completa
├── train.py                      # Experimenter: treino GA
├── ai/
│   ├── DESIGN.md                 # documento de decisões de design
│   ├── network.py                # ActorCritic (MLP actor+critic)
│   ├── neural_population.py      # GA: crossover, mutação, elite
│   └── read_log.py               # análise do training_log.jsonl
├── engine/
│   ├── application.py            # loop principal com timestep fixo
│   ├── clock.py                  # delta variável + passo fixo
│   ├── camera.py                 # world↔screen + zoom + culling
│   ├── renderer.py               # desenho de primitivas e texto
│   ├── background.py             # fundo parallax
│   ├── world.py                  # contêiner genérico de entidades
│   ├── inspector.py              # HUD de propriedades + relações orbitais
│   ├── vector_visualizer.py      # setas de força
│   ├── trajectory_visualizer.py  # predição futura de trajetória
│   └── memory_monitor.py         # snapshots do tracemalloc
├── entities/
│   ├── entity.py                 # ABC com update/fixed_update/render
│   ├── massobject.py             # corpo gravitacional
│   ├── rocket.py                 # foguete (herda MassObject)
│   └── entity_picker.py          # hit-test para clique com tolerância
├── physics/
│   ├── gravity.py                # NewtonianGravity (N-corpos)
│   ├── isolated_host_gravity.py  # host estático (usado no treino)
│   ├── rk4.py                    # Runge-Kutta de 4ª ordem
│   ├── semi_implicit_euler.py    # Euler simplético
│   ├── composite_acceleration.py # gravidade + empuxo
│   ├── collision.py              # colisão multi-círculo
│   ├── relative_orbit.py         # elementos orbitais relativos
│   └── integrator.py             # protocols + ABC
├── control/
│   └── keyboard.py               # Input (teclado + mouse + clique vs. pan)
└── assets/
    ├── sky.jpg                   # fundo parallax
    └── mem_report.py             # gráficos do MemoryMonitor
```

### Física e simulação

- **`NewtonianGravity`** — gravidade de N corpos, com softening opcional. Aproveita a simetria do par `(i, j)` para computar cada interação uma única vez.
- **`RK4Integrator`** — Runge-Kutta de 4ª ordem. O estado inicial nunca é mutado até o último estágio; as acelerações retornadas são do **último estágio (K4)** e alimentam o vetor de força visual.
- **`CompositeAccelerationModel`** — soma a gravidade do modelo base com `additional_acceleration` de cada corpo (o **empuxo** do foguete). A lista de corpos que contribuem é filtrada uma única vez na construção.
- **`IsolatedHostGravity`** — usado apenas no treino. O hospedeiro é **estático** (sua aceleração é sempre zero) e cada foguete sente apenas a atração do host — foguetes não interagem entre si. Isso isola os indivíduos da população.
- **`RelativeOrbit2D`** — calcula elementos orbitais (excentricidade, periapsis/apoapsis, semieixo maior, período kepleriano, Δv de escape) a partir de posição, velocidade e massa relativas. Usado pelo inspector e pela observação da rede.
- **Colisão e fusão** — dois corpos em contato **fundem** se a velocidade relativa exceder `merge_min_relative_speed`; abaixo disso, o contato é tratado como "pouso" (o foguete pode repousar sem ser absorvido). A fusão preserva massa, momento e volume (`r ∝ m^(1/3)`).
- **Acoplamento (dock)** — colisão perfeitamente inelástica: foguete e hospedeiro passam a compartilhar o centro de massa; a posição do foguete é sincronizada a cada passo fixo (`post_integrate`) e o bico é mantido na direção radial.

Toda a física roda em **passos fixos** de `dt = 1/120 s`. O acumulador do `Clock` corta deltas anormais e limita o número de passos por frame (`max_fixed_steps = 8`), evitando a "espiral da morte". O render interpola entre `previous_position` e `position` via `interpolation_alpha`, de modo que a exibição permaneça suave mesmo quando a taxa de frames diverge da de passos fixos.

### Foguete e controle

- Foguete é um `MassObject` com **orientação**, **rotação** (120°/s por default) e **empuxo binário** (`thrust_acceleration = 120 u/s²` no construtor padrão; `1000` nos experimentos).
- Colisão real usa **4 círculos** alinhados ao eixo do foguete (traseira, meio, frente, bico), cada um com o raio da meia-largura do corpo.
- **Consumo de combustível** por passo fixo: `fuel_thrust_rate · dt` com o motor ativo, `fuel_rotation_rate · dt` com rotação ativa. `_thrust_authorized` é gravado **antes** do débito, para que o último passo de queima autorizado ainda produza efeito.
- **Desacoplagem** acontece na **borda de subida** do empuxo (`not self.thrusting → self.thrusting`) e somente se houver combustível. É o que permite girar sobre a superfície sem cair e só sair quando o jogador/rede decidir queimar.
- **`try_land`** reacopla o foguete a um corpo ao cruzar a casca `r_dock = host.radius + body_length/2 + DOCKING_MARGIN`, desde que a velocidade relativa esteja abaixo de `landing_max_speed`.

### Agente (rede neural)

**Espaço de ações (6 discretas):**

| idx | rotação | propulsão |
|-----|---------|-----------|
| 0   | −1      | off       |
| 1   |  0      | off       |
| 2   | +1      | off       |
| 3   | −1      | on        |
| 4   |  0      | on        |
| 5   | +1      | on        |

**Espaço de observação (9 floats, todos normalizados por quantidades naturais de dois corpos):**

| idx | grandeza | normalização |
|-----|----------|--------------|
| 0   | `r`      | `r_ref = α · R_host` |
| 1   | `v_r`    | `v_ref = sqrt(μ/r_ref)` |
| 2   | `v_t`    | `v_ref` |
| 3   | `sin(θ_rel)` | ângulo bico−radial |
| 4   | `cos(θ_rel)` | ângulo bico−radial |
| 5   | `docked`  | `{0, 1}` |
| 6   | `fuel / fuel_max` | `[0, 1]` |
| 7   | `log10(a_thrust / a_g_ref)` | saturável em `[-1, 2]` |
| 8   | `log10(ω_rocket / ω_ref)`   | saturável em `[-1, 2]` |

**Rede `ActorCritic`** (`src/ai/network.py`):

- Tronco compartilhado: `Linear(9, 128) → Tanh → Linear(128, 128) → Tanh`.
- Cabeça **Actor**: `Linear(128, 6)` → logits das ações.
- Cabeça **Critic**: `Linear(128, 1)` → `V(s)` (mantido por compatibilidade com a arquitetura, mas a evolução genética usa apenas a política).
- Inicialização **ortogonal** (ganho `√2` no tronco, `0.01` no ator). O ganho pequeno no ator faz a política inicial nascer "em cima do muro", sem viés para nenhuma ação.

**Inferência** (`select_action`): com `deterministic=True`, `argmax` dos logits; com `deterministic=False`, amostragem categórica. No treino e na simulação, `deterministic=True`.

### Função de recompensa

Baseada em **shaping por potencial** (Ng et al. 1999) com terminais explícitos. A cada passo:

```
r_t = Φ(s_{t+1}) − Φ(s_t)                    (shaping telescopado)
    − c_thrust · 1[empuxo]                   (custo denso de queima)
    − c_rotate · |rotação|                   (custo denso de giro)
    − c_time                                 (custo denso de tempo)
    + c_in_band · 1[em banda]                (recompensa densa de permanência)
```

O potencial `Φ(s) = −5 · (2·e_raio + 0.75·e_radial + 0.75·e_tang)` mede o afastamento da órbita-alvo; cada termo é limitado por `_POTENTIAL_CAP` (anti-explosão). **Banda estável** exige simultaneamente:

```
|r / r_ref − 1|       < 0.05
|v_r| / v_ref         < 0.10
||v_t| − v_ref|/v_ref < 0.10
```

Terminais:

| evento | Δr | tipo |
|--------|----|----|
| Sucesso (2π de progresso em banda) | +100 | terminal |
| Meia-órbita (π de progresso, uma vez por episódio) | +50 | não-terminal |
| Colisão destrutiva | −100 | terminal |
| Fuga (`r > 8·R_host`) | −50 | terminal |
| Re-acoplagem após decolar | −50 | terminal |
| Combustível esgotado | −20 | terminal |
| Timeout (`3 × T_orbital / dt` passos) | −30 | terminal |

### Treinamento (algoritmo genético)

O treino (`src/train.py`) roda em **`Experimenter`**, que instancia **1 host estático** + **N foguetes** (um por rede da população) e os avalia **simultaneamente** no mesmo cenário. A cada geração:

1. **Sorteia cenário**: massa e raio do host, ângulo de acoplamento e α (via **currículo**).
2. **50%** dos foguetes começam **acoplados** à superfície; **50%** começam **em voo**, com `r/r_ref` log-uniforme em `[0.6, 2.2]`, `v_t` proporcional a `v_circ` (40% retrógrado) e `v_r` com jitter de ±25%. O estado inicial em voo dá exemplos de **descida orbital**, que o regime acoplado nunca produz.
3. **Cada foguete é controlado pela sua rede** durante até `3 × T_orbital` passos fixos.
4. **Aptidão** de cada rede = recompensa acumulada.
5. **`NeuralPopulation.evolve(fitness)`**:
   - Os `elite_count` melhores são preservados **sem alteração**.
   - Cada descendente é gerado por **crossover elemento-a-elemento** entre os dois melhores (`torch.rand < 0.5` por peso) seguido de **mutação gaussiana** (`N(0, std²)` com probabilidade `rate` por peso).
   - A **taxa de mutação decai linearmente** de `MUTATION_RATE_MAX (0.06)` a `MUTATION_RATE_MIN (0.02)` ao longo das gerações.
6. **Currículo de α**: o limite superior de `α` cresce linearmente de `2.5` (gen 0) a `5.0` (última geração), com limite inferior fixo em `1.5`. As primeiras gerações veem apenas alvos baixos; a generalização vem depois que a base está sólida.
7. **Ao fim de cada geração** um registro **JSON Lines** é anexado em `checkpoints/training_log.jsonl`, com métricas agregadas, terminais por tipo e detalhes por indivíduo (recompensa, passos, `max_r_ratio`, `in_band_ratio`, `fuel_used`, `half_awarded`).
8. **Ao encerrar** (janela fechada, `max_generations` atingido ou Ctrl+C), `shutdown()` salva a **melhor rede global** em `checkpoints/best_ever.pt` (state_dict puro, carregável por `load_network`) e escreve a entrada `session_end` no log.

**Configuração padrão** em `main()`: 100 indivíduos, 150 gerações, elite 5, `mutation_std=0.03`, semente `42`. Cada passo fixo consulta as 100 redes em sequência, mas como cada rede é uma MLP pequena, o custo por passo é dominado pela física e pelo `Control_with_network`, não pela inferência.

### Ambiente de teste (`Gravitation`)

`src/main.py` roda a simulação completa de **N corpos** (estrelas, planetas, luas, meteoros e o foguete), populada por `_seed_solar_system`:

- **Estrelas** (1 a 1 por default): repouso no centro ou polígono regular com órbita baricêntrica.
- **Planetas** (0 por default): órbitas circulares com jitter.
- **Luas** (0 por default): órbitas circulares ao redor dos planetas.
- **Meteoros** (0 por default): elipses de excentricidade `[0.6, 0.9]` ao redor de planetas (60%) ou estrelas (40%).
- **Foguete**: nasce acoplado a um planeta sorteado (ou à estrela, se não houver planetas).

Recursos interativos:

- **Câmera**: zoom ancorado no cursor, pan por arrasto, follow da entidade selecionada (`V`).
- **Seleção**: clique esquerdo seleciona; clique direito define a **referência** para o inspector relacional.
- **Inspector**: mostra propriedades orbitais relativas (excentricidade, peri/apoapsis, período, Δv de escape) e os coeficientes de desacoplagem/acoplagem calculados analiticamente.
- **Vetores de força** (`F`) e **predição de trajetória** (`T`).
- **Controle do foguete**: teclado (setas rotacionam; espaço propulsa; ↑/↓ ajustam α) ou **rede neural** (`I` alterna; requer `--model`).
- **Carregamento de modelo**: `--model checkpoints/best_ever.pt` carrega a rede no `__init__`; o HUD no rodapé mostra `REDE NEURAL` ou `JOGADOR`.
- **Monitor de memória** (`--monitor-memory`): snapshots periódicos do `tracemalloc`, analisáveis por `mem_report.py`.

O loop principal (`src/engine/application.py`) suporta **aceleração da simulação** via `speed_multiplier` (mantendo `dt` fixo, portanto sem afetar precisão física): o `Experimenter` expõe `←`/`→` para ajustar de 1× a 3×. O `Gravitation` mantém 1×.

### Logs e análise

**`training_log.jsonl`** — uma linha JSON por geração, append-only. Campos principais:

- `event`: `"generation"` ou `"session_end"`.
- `gen`, `wall_time_s`, `alpha`, `host_mass`, `host_radius`, `r_ref`, `v_ref`, `orbital_period_s`, `max_steps`, `steps_run`, `population_size`.
- `fitness`: `{best, avg, median, std, min}`.
- `terminals`: contagem por motivo (`success`, `half_orbit`, `timeout`, `escape`, `collision`, `never_launched`, `redocked`, `fuel_exhausted`).
- `individuals`: por indivíduo — `reward`, `terminal`, `steps`, `max_r_ratio`, `in_band_ratio`, `half_awarded`, `fuel_used`.

**`best_ever.pt`** — `state_dict` puro da rede de melhor aptidão global. Compatível com `Experimenter.load_network` e com o `--model` do `Gravitation`.

**`src/ai/read_log.py`** — consome o JSONL com `pandas` e gera:

1. `learning_curve.png` — `best` e `avg` ao longo das gerações.
2. `terminals.png` — stacked bar de terminais por geração.
3. `proximity_vs_reward.png` — `max_r_ratio` × recompensa, colorido por geração.
4. Eficiência de combustível nos episódios de sucesso (`reward / fuel_used`).

**`src/assets/mem_report.py`** — consome snapshots do `MemoryMonitor` (`mem_dumps/session_*/`), gera `mem_over_time.png` com memória rastreada e número de objetos ao longo do tempo.

---

## Como executar

### Requisitos

- **Python 3.10+**
- Dependências:

```bash
pip install pygame torch numpy pandas matplotlib
```

- Verificar instalação: `python -c "import pygame, torch, numpy, pandas; print('ok')"`.

### Simulação manual

```bash
python -m src.main
```

Abre a janela de `Gravitation` com o sistema estelar gerado e o foguete acoplado. Controles:

| tecla | ação |
|-------|------|
| `↑` / `↓` | aumentar / diminuir α do alvo orbital |
| `←` / `→` | rotacionar o foguete |
| `Espaço` | acionar empuxo (desacopla se ainda estiver pousado) |
| `I` | alternar controle jogador ↔ rede neural (requer `--model`) |
| `F` | mostrar/ocultar vetores de força |
| `T` | mostrar/ocultar trajetória futura |
| `V` | seguir o corpo selecionado |
| scroll | zoom ancorado no cursor |
| arrastar botão esquerdo | pan |
| clique esquerdo | selecionar entidade |
| clique direito | definir corpo de referência para o inspector |

### Simulação com modelo treinado

```bash
python -m src.main --model checkpoints/best_ever.pt
```

Carrega a rede no startup. Pressione `I` para ativar o controle neural. O foguete passa a ser comandado pela política (determinística, `argmax`) e o HUD no rodapé exibe `Controle do foguete: REDE NEURAL`.

### Treinamento

```bash
python -m src.train
```

Roda o `Experimenter`. Parâmetros de execução (população, gerações, elite, `mutation_std`, semente) estão em `main()` de `src/train.py` — ajuste ali se quiser variar a escala do experimento. Durante o treino, `←`/`→` aceleram a simulação (1× a 3×, mantendo precisão física). Ao encerrar (janela fechada ou `max_generations` atingido), a melhor rede global é salva em `checkpoints/best_ever.pt` e o log é finalizado com a entrada `session_end`.

Saídas em `checkpoints/`:

- `best_ever.pt` — melhor rede global (state_dict puro).
- `training_log.jsonl` — métricas por geração (append-only).

### Análise de logs

```bash
python src/ai/read_log.py
```

Consome `checkpoints/training_log.jsonl` e gera os gráficos em `learning_curve.png`, `terminals.png`, `proximity_vs_reward.png`. Também imprime no terminal a eficiência de combustível dos episódios de sucesso por geração. Requer que exista pelo menos uma geração completa no log.

### Monitor de memória

```bash
python -m src.main --monitor-memory --dump-dir mem_dumps/ --dump-interval 5
python src/assets/mem_report.py mem_dumps/
```

O `--monitor-memory` liga o `tracemalloc` e agenda snapshots a cada `--dump-interval` segundos. `--max-dumps N` limita o total. O `mem_report.py` gera `mem_over_time.png` com memória rastreada e contagem de objetos ao longo do tempo. Sem a flag, nenhum custo é pago.

---

## Resultados

<!-- A preencher -->