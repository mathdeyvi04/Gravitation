# DESIGN.md — Piloto autônomo de inserção orbital

Documento de decisões de design do projeto de RL. Escrito antes de
qualquer linha de código, para servir de contrato entre as fases.

## 1. Objetivo do agente

Levar o foguete de um estado acoplado à superfície de um corpo celeste
até uma órbita estável (não necessariamente circular) ao redor desse corpo, com o MÍNIMO de recursos
(combustível e tempo) gastos. E também ser capaz de pousar corretamente em outro hospedeiro.

Detalhes:

- Órbita estável = órbita ligada (energia específica < 0, excentricidade
  < 1) cujo raio permanece dentro de uma faixa de tolerância por uma
  revolução completa (2π radianos de progresso angular acumulado).
  Não exige excentricidade zero — órbitas elípticas são válidas.
- O agente deve ser capaz de partir do repouso na superfície (decolar),
  manobrar para a órbita alvo, e opcionalmente retornar (pouso).
- O raio-alvo é um parâmetro ajustável α = r_ref / R_body, com valor
  padrão α = 2.5 e faixa válida [1.5, 8.0]. O agente deve funcionar
  para qualquer α nessa faixa.

## 2. Por que RL (e não controle clássico)

Inserção orbital tem solução analítica (Hohmann), mas:

- o foguete está acoplado e precisa decolar antes;
- empuxo é binário (on/off), sem modulação fina;
- a rotação consome combustível, então até mirar o bico é uma
  decisão de orçamento;
- queremos minimizar esforço e combustível, não apenas atingir o alvo.

PPO é adequado: simples, estável, e projetado para espaços de ação
discretos.

## 3. Espaço de ações

Discreto, 6 ações:

| idx | rotação | propulsão |
|-----|---------|-----------|
| 0   | -1      | off |
| 1   |  0      | off |
| 2   | +1      | off |
| 3   | -1      | on  |
| 4   |  0      | on  |
| 5   | +1      | on  |

- Rotação: -1, 0, +1 (esquerda, nada, direita). Velocidade angular
  constante (`Rocket.rotation_speed`, 120°/s por default).
- Propulsão: binária. Consome combustível por passo enquanto ativa.

Recurso finito por episódio. Os parâmetros vivem em
`SimulationConfig`: `fuel_max`, `fuel_thrust_rate`, `fuel_rotation_rate`.

Consumo por passo fixo (`dt = Configs.fixed_timestep`):

| comando | consumo por passo |
|---|---|
| empuxo ativo | `fuel_thrust_rate · dt` |
| rotação ativa | `fuel_rotation_rate · dt` |

## 5. Espaço de observação

Vetor de 9 floats. Todos normalizados por quantidades naturais da
solução de dois corpos.

Sejam `μ = G·(M_target + M_rocket)`,
`r_ref = α · R_target`, `v_ref = sqrt(μ/r_ref)`,
`ω_ref = v_ref/r_ref`, `a_g_ref = μ/r_ref²`.

| idx | nome | fórmula | faixa típica |
|-----|------|---------|--------------|
| 0 | r/r_ref                    | distância / raio-alvo | [1.5, 8]     |
| 1 | v_r/v_ref                  | velocidade radial / v_circ | [-5, 5]      |
| 2 | v_t/v_ref                  | velocidade tangencial / v_circ | [-5, 5]      |
| 3 | sin(θ_rel)                 | seno do ângulo bico−radial | [-1, 1]      |
| 4 | cos(θ_rel)                 | cosseno do ângulo bico−radial | [-1, 1]      |
| 5 | docked                     | 1 se acoplado | {0, 1}       |
| 6 | fuel_remaining/fuel_max    | fração de combustível restante | [0, 1]       |
| 7 | log10(a_thrust/a_g_ref)    | razão empuxo/gravidade local | [-1, 2]      |
| 8 | log10(ω_rocket/ω_ref)      | razão giro/vel. angular orbital | [-1, 2]      |

### Justificativa de cada índice

- **0–2: coordenadas polares normalizadas.** O objetivo é definido em
  termos de r e v angular, então damos exatamente essas quantidades. As
  normalizações `r_ref` e `v_ref` alinham a observação à estrutura do
  problema e conferem invariância de escala: a mesma política funciona
  para qualquer α ∈ [1.5, 8.0].
- **3–4: sin/cos do ângulo relativo.** Evita descontinuidade em ±π. Se
  déssemos o ângulo cru, a rede veria 3.14 e −3.14 como opostos, quando
  são quase iguais.
- **5: docked.** Estado binário que muda a dinâmica (acoplado = posição
  travada no hospedeiro).
- **6: combustível restante.** A rede precisa saber seu orçamento para
  decidir quando vale a pena queimar e — dado §4 — quando vale a pena
  **girar**. Sem isso, ela não pode aprender a conservar.
- **7: thrust_ratio em log10.** Razão entre aceleração de empuxo e
  aceleração gravitacional local na órbita-alvo. Diz "quanta autoridade
  de controle eu tenho". Varia em ordens de magnitude com α, por isso
  log10.
- **8: rotation_ratio em log10.** Razão entre velocidade angular do
  foguete e velocidade angular orbital. Diz "quão rápido consigo girar
  comparado ao período orbital". Mesma lógica.

## 6. Função de recompensa

Recompensa por passo:

    r_t = Φ(s_{t+1}) − Φ(s_t)                 (shaping)
        − c_thrust · 1[propulsão]             (custo denso de queima)
        − c_rotate · |rotação|                (custo denso de giro)
        − c_time                              (custo denso de tempo)

Com `c_thrust = 0.02`, `c_rotate = 0.003`, `c_time = 0.001`.

Potencial de shaping:

    Φ(s) = −( 2.0 · e_raio + 0.75 · e_radial + 0.75 · e_tang )

Cada termo limitado a 4.0 para evitar explosão:

    e_raio   = |r / r_ref − 1|
    e_radial = |v_r| / v_ref
    e_tang   = ||v_t| − v_ref| / v_ref

Recompensa terminal:

| evento         | Δr     | tipo       | motivo |
|----------------|--------|------------|--------|
| sucesso        | +100   | terminated | órbita estável mantida por 2π |
| colisão        | −100   | terminated | destruição do foguete |
| fuga (> 8r_ref)| −50    | terminated | não retorna mais |
| combustível = 0 sem sucesso | −20 | terminated | falha de planejamento |
| limite de tempo| −5     | truncated  | lento demais |

### Justificativa

- **Shaping por potencial (Ng et al. 1999).** Somar Φ(s') − Φ(s) não
  altera a política ótima, porque o somatório telescopa. Ganhamos
  gradiente denso sem enviesar a solução.
- **Pesos do potencial.** Raio é o primeiro obstáculo (chegar perto da
  órbita-alvo); velocidades se ajustam depois. Daí o fator 2.0 vs 0.75.
- **Custo denso de empuxo (0.02/step).** 100 passos de queima custam
  2.0 — significativo, mas não proibitivo. Pequeno o bastante para o
  agente aprender a usar empuxo, grande o bastante para desencorajar
  pulsing constante.
- **Δ Custo denso de giro.** O DESIGN original justificava `c_rotate`
  como "girar não consome combustível, apenas tempo". **Isso deixou de
  ser verdade** com a implementação atual (§4): a rotação debita
  `fuel_rotation_rate · dt` por passo. Hoje o giro é penalizado **duas
  vezes** — pelo shaping `c_rotate` e pelo consumo de tanque.
  **Decisão pendente**: (a) aceitar a dupla penalidade como sinal
  intencional, ou (b) reduzir/zerar `c_rotate` para evitar redundância.
  Escolher após medir o uso real de combustível por giro nos primeiros
  checkpoints.
- **Custo de tempo pequeno.** Quebra empates sem dominar.
- **Penalidade terminal por combustível esgotado.** Distinta da colisão
  e da fuga: é uma falha de planejamento, não de física. −20 é
  suficientemente negativa para desencorajar, menor que colisão porque
  não destrói o foguete.
- **Recompensa de sucesso grande.** +100 domina qualquer shaping
  acumulado; sinaliza que órbita estável é o que importa.

### Por que a órbita-alvo é parametrizada por α

Se `r_ref` fosse fixo, o agente aprenderia a atingir exatamente aquele
raio. Ao parametrizarmos por α e randomizarmos α no treino, forçamos a
política a generalizar: qualquer órbita dentro de [1.5, 8.0]·R_body é
válida, dependendo do α do episódio. As observações normalizadas por
`r_ref` fazem com que a tarefa, em espaço de observação, seja
essencialmente a mesma para todo α.

## 7. Condições de término

- **Sucesso:** progresso angular acumulado ≥ 2π **e** estado sempre
  dentro da faixa estável. Se sair da faixa, o progresso zera. Isso
  evita que a política "cole" na órbita por um instante e dispare
  sucesso.
- **Colisão:** qualquer dos quatro círculos de colisão do foguete
  intersecta o círculo do alvo, **com velocidade relativa acima de
  `SimulationConfig.merge_min_relative_speed`**. Abaixo desse limiar,
  a sobreposição é tratada como contato suave (pouso), não como
  colisão — o que permite o foguete repousar sem ser absorvido.
- **Fuga:** r > 8·r_ref.
- **Combustível esgotado sem sucesso.**
- **Truncamento:** 3600 passos.

> **Δ Nota sobre acoplagem durante o episódio.** `Rocket.try_land`
> reacopla o foguete a um hospedeiro sempre que a distância entra na
> casca `r_dock` **e** a velocidade relativa fica dentro de
> `SimulationConfig.landing_max_speed` (valor atual igual a
> `merge_min_relative_speed`). Após o desacoplamento inicial, um
> reencontro lento com o alvo dispararia esse caminho. O ambiente de
> RL precisa mapear isso para um dos eventos acima (provavelmente
> truncamento ou falha, jamais sucesso). **Decisão pendente.**

Faixa estável (todas precisam valer simultaneamente):

    |r/r_ref − 1|       < 0.12
    |v_r| / v_ref       < 0.20
    ||v_t| − v_ref|/v_ref < 0.20

Os valores são a "banda de tolerância" da órbita: 12% em raio, 20% em
cada velocidade. São generosos o bastante para uma órbita visualmente
elíptica, apertados o bastante para excluir trajetórias de escape.

## 8. Estado inicial

Foguete acoplado à superfície do alvo, em ângulo radial aleatório.
Velocidade relativa zero. Ângulo do foguete = ângulo radial.


## 9. Randomização entre episódios

Alvos do ambiente de RL (não confundir com a randomização da cena de
teste em `Gravitation`, que hoje usa valores compatíveis com a tabela
abaixo mas pode divergir):

| parâmetro | distribuição                                       |
|---|----------------------------------------------------|
| massa do alvo     | uniforme, faixa `Gravitation._PLANET_MASS_RANGE`   |
| raio do alvo      | uniforme, faixa `Gravitation._PLANET_RADIUS_RANGE` |
| ângulo de acoplagem | uniforme em [0, 2π)                                |
| α = r_ref / R_body | uniforme [1.5, 5.0]                                |

α é randomizado apenas durante treino. Em avaliação e em produção, α
é fixado (padrão 2.5) ou ajustado por teclado.

## 10. Constantes físicas e de simulação

Os valores abaixo **vivem no código**; esta seção apenas os referencia.
Alterações numéricas devem ser feitas nos arquivos citados — este
documento não é fonte de verdade para valores.

| símbolo | referência no código |
|---|---|
| G | `SimulationConfig.gravitational_constant` |
| softening | `SimulationConfig.gravitational_softening` |
| dt | `Configs.fixed_timestep` |
| massa do foguete | `Gravitation._ROCKET_MASS` |
| empuxo (aceleração) | `Gravitation._ROCKET_THRUST_ACCELERATION` |
| velocidade de giro | `Rocket.rotation_speed` (default) |
| corpo do foguete | `Gravitation._ROCKET_BODY_LENGTH` × `_ROCKET_BODY_WIDTH` |
| nariz do foguete | `Gravitation._ROCKET_NOSE_LENGTH` |
| margem de acoplamento | `Rocket.DOCKING_MARGIN` |
| `fuel_max` | `SimulationConfig.fuel_max` |
| `fuel_thrust_rate` | `SimulationConfig.fuel_thrust_rate` |
| `fuel_rotation_rate` | `SimulationConfig.fuel_rotation_rate` |
| `merge_min_relative_speed` | `SimulationConfig.merge_min_relative_speed` |
| `landing_max_speed` | `SimulationConfig.landing_max_speed` |
## 11. Hiperparâmetros de PPO (baseline)

| parâmetro | valor |
|---|---|
| hidden (2 camadas) | 128 |
| learning rate      | 3e-4 |
| n_steps por env    | 2048 |
| batch size         | 64 |
| n_epochs           | 10 |
| gamma              | 0.995 |
| gae_lambda         | 0.95 |
| clip_eps           | 0.2 |
| vf_coef            | 0.5 |
| ent_coef           | 0.01 |
| max_grad_norm      | 0.5 |

Valores clássicos do paper original de PPO. Ajustes finos somente
depois de um pipeline funcional.

## 12. Reprodutibilidade

- Seed mestra: 42.
- `torch.manual_seed(42)`, `np.random.seed(42)` no início do treino.
- Cada `env.reset()` recebe seed derivada via
  `np.random.SeedSequence(42).spawn(n_envs)`.
- Chamadas estocásticas da política usam `torch.Generator`, nunca o
  `random` global.

## 13. Critério de sucesso do projeto

1. ≥ 70% de sucesso em 100 episódios consecutivos de avaliação
   determinística, com α fixo em 2.5.
2. Generalização: mesma política com α ∈ {2.0, 3.0, 4.0, 5.0} mantém
   ≥ 50% de sucesso.
3. Consumo médio de combustível em episódios de sucesso
   significativamente menor que um baseline "queimar sem parar".
4. `watch_training` mostra a política em ação sem erros.
5. O controle por teclado altera α em tempo real e o agente responde
   corretamente (tenta a nova órbita).