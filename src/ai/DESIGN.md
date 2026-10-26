# DESIGN — Piloto autônomo de inserção orbital

## 1. Objetivo

Levar o foguete da superfície de um corpo celeste até uma órbita
estável com o mínimo de combustível e tempo, e opcionalmente pousar
em outro corpo. O aprendizado é dividido em quatro fases
curriculares, treinadas em sequência:

| Fase | Nome            | Meta                                          |
|------|-----------------|-----------------------------------------------|
| 1    | Desacoplagem    | Sair da superfície com segurança              |
| 2    | Órbita maior    | Atingir órbita estável com r_ref > r_inicial  |
| 3    | Órbita menor    | Atingir órbita estável com r_ref < r_inicial  |
| 4    | Acoplagem       | Pousar suavemente em outro corpo              |

Cada fase tem sua própria função de recompensa, critérios de
término e currículo de estados iniciais. A rede é compartilhada
entre as fases: a Fase 1 treina os pesos do zero; as seguintes
fazem fine-tuning da população.

## 2. Espaço de ações (discreto, 6)

| idx | rotação | propulsão |
|-----|---------|-----------|
| 0   | -1      | off |
| 1   |  0      | off |
| 2   | +1      | off |
| 3   | -1      | on  |
| 4   |  0      | on  |
| 5   | +1      | on  |

Rotação: `Rocket.rotation_speed` (rad/s). Propulsão binária;
consome `fuel_thrust_rate·dt` por passo. Rotação consome
`fuel_rotation_rate·dt` (dupla penalidade: tanque **e** `c_rotate`).

## 3. Espaço de observação (9 floats)

Normalizado por `μ = G·(M_host + M_rocket)`, `r_ref = α·R_host`,
`v_ref = √(μ/r_ref)`, `ω_ref = v_ref/r_ref`, `a_g_ref = μ/r_ref²`.

| idx | nome | faixa |
|-----|------|-------|
| 0 | r / r_ref                               | [0, 8] |
| 1 | v_r / v_ref                             | [-5, 5] |
| 2 | v_t / v_ref                             | [-5, 5] |
| 3 | sin(θ_bico − θ_radial)                  | [-1, 1] |
| 4 | cos(θ_bico − θ_radial)                  | [-1, 1] |
| 5 | docked                                  | {0, 1} |
| 6 | fuel / fuel_max                         | [0, 1] |
| 7 | log10(a_thrust / a_g_ref) clip [-1, 2]  | |
| 8 | log10(ω_rocket / ω_ref) clip [-1, 2]    | |

As mesmas 9 observações servem às quatro fases — a política não
precisa saber em qual fase está, apenas o estado físico.

## 4. Fase 1 — Desacoplagem

Estado inicial: foguete **acoplado** na superfície de um host com
`(M, R)` sorteados em faixas fixas. Ângulo radial aleatório.
Objetivo: atingir `r ≥ 2·R_host` com `v_r > 0`.

Recompensa por passo:

    r_t = Φ(s_{t+1}) − Φ(s_t)
          − c_thrust · 1[propulsão]
          − c_rotate · |rotação|
          − c_time

    Φ(s) = min( 20 · max(0, r/R_host − 1), 20 )

Terminais:

| evento                                   | Δr   | tipo       |
|------------------------------------------|------|------------|
| sucesso: `r ≥ 2·R_host` **e** `v_r > 0`  | +100 | terminated |
| colisão com o host após desacoplar       | −50  | terminated |
| reacoplagem após desacoplar              | −50  | terminated |
| fuga: `r > 8·R_host`                     | −50  | terminated |
| combustível esgotado                     | −30  | terminated |
| timeout                                  | −50  | truncated  |

Timeout por episódio: `0.5 × T_orb(R_host)` passos, onde
`T_orb(r) = 2π·√(r³/μ)`.

### Justificativa

- **Shaping por potencial (Ng et al. 1999).** `Φ(s') − Φ(s)` não
  altera a política ótima (somatório telescopa) e fornece gradiente
  denso durante a subida.
- **Escala de Φ = 20.** Uma subida de `1·R_host` rende +20; o
  terminal de queda (−50) domina, mas ainda deixa espaço para
  distinguir "subiu até 1.5R e caiu" de "mal saiu do lugar e caiu".
  Sem isso, o sinal seria binário demais para o GA discriminar.
- **Sucesso exige `v_r > 0`.** Cruzar `2·R_host` já caindo não
  conta — o foguete precisa estar ganhando altitude.
- **Queda simétrica ao timeout (−50).** Ambos são fracassos.
  Punir a queda mais que a inação ensinaria "não tente", o que é
  indesejável na fase de aprendizado.
- **Reacoplagem após desacoplar = falha.** O foguete voltou ao
  estado inicial; não aprendeu nada. `try_land` fica **desligado**
  durante a Fase 1 para não capturar o foguete antes que ele limpe
  a casca de acoplamento — a detecção de queda é feita via
  `collides` + velocidade relativa.

## 5. Fases 2–4

Placeholders — cada uma será detalhada quando a anterior
estabilizar. Requisitos de alto nível:

- **Fase 2 (órbita maior):** estado inicial = foguete em voo
  ascendente, fora da atmosfera; meta = inserir em órbita circular
  com `r_ref > r_inicial`. Sucesso = manter a banda `r_ref ± 12%`
  por `2π` de progresso angular.
- **Fase 3 (órbita menor):** simétrico à Fase 2.
- **Fase 4 (acoplagem):** estado inicial = órbita estável; meta =
  aproximar e pousar em outro corpo sem colisão destrutiva.

## 6. Algoritmo de treino

Algoritmo genético sobre os pesos da rede.

Justificativa: as recompensas terminais dominam e os episódios são
longos. PPO exigiria bootstrap de valor e mistura de gradientes
entre rollouts de uma mesma rede, o que é complicado quando cada
rollout explora uma condição inicial diferente e não há garantia
de estacionariedade entre gerações. GA evolui a população de
pesos diretamente e combina naturalmente com o currículo por
fases.

Cada geração:

1. Cada rede joga `K = 3` episódios com estado inicial
   randomizado (host, ângulo); fitness = média das recompensas.
2. Ordena por fitness.
3. `elite_count` melhores são clonados intactos.
4. Descendência: dois pais sorteados por **torneio** (`k = 3`),
   crossover uniforme elemento-a-elemento, mutação gaussiana por
   peso.
5. `mutation_rate` decai linearmente de 0.05 a 0.01; `mutation_std`
   fixo em 0.03.

## 7. Reprodutibilidade

- Semente mestra: 42.
- Rede inicial: `NeuralPopulation(seed=42)`; cada indivíduo deriva
  de `SeedSequence(42).spawn(N)`.
- Crossover e mutação usam `torch.Generator` semeado por geração.
- Física é determinística (RK4, `dt` fixo).
- Estocasticidade do ambiente (host, ângulos) usa `random`
  resetada por episódio com `base_seed + gen·K + ep`.

## 8. Critério de sucesso do projeto

1. Fase 1: ≥ 95% de sucesso em 100 episódios de avaliação
   determinística.
2. Fase 2: ≥ 70% de sucesso para `α ∈ [1.5, 3.0]`.
3. Fase 3: ≥ 70% de sucesso para `α ∈ [3.0, 5.0]`.
4. Fase 4: ≥ 70% de sucesso em 100 episódios.
5. Consumo médio de combustível nos sucessos menor que baseline
   "queimar sem parar".
6. `main.py --model checkpoints/phase1_best.pt` mostra a política
   funcionando sem erros.