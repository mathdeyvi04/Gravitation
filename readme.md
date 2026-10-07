# Simulação Gravitacional de N Corpos

Um ambiente interativo de simulação gravitacional construído em Python com Pygame, onde corpos massivos interagem por atração newtoniana, colidem, fundem-se e são inspecionados em tempo real por meio de uma interface pensada para exploração e depuração.

O projeto começou como uma experiência de física com uma janela e alguns círculos e evoluiu para um arcabouço com separação de responsabilidades bem definida: um motor de aplicação genérico, um modelo físico independente do desenho, um contêiner de entidades reaproveitável e uma cena concreta que orquestra a simulação. Cada camada pode ser substituída sem tocar nas outras.

---

## Arquitetura geral

O projeto está dividido em três grandes blocos:

- **`src/engine/`** — infraestrutura reutilizável: `Application`, `Clock`, `Renderer`, `Camera2D`, `Background`, `Input`, `World`, além dos visualizadores (`VectorVisualizer`, `TrajectoryVisualizer`), do picker (`EntityPicker`) e do sistema de inspeção (`Inspector`, `RelationInspector`, `InspectorHUD`).
- **`src/entities/`** — tipos de objeto: a base abstrata `Entity` e sua implementação concreta `MassObject`.
- **`src/physics/`** — modelo físico: `NewtonianGravity`, os integradores (`SemiImplicitEulerIntegrator`, `RK4Integrator`) e o cálculo de órbitas relativas (`RelativeOrbit2D`).

A cena concreta (`Gravitation`, em `src/main.py`) é quem amarra tudo: ela decide a ordem do passo físico, a política de fusão e como os dados são apresentados.

Nada em `engine/` sabe o que é um corpo com massa. Nada em `entities/` sabe como o tempo passa. Nada em `physics/` sabe que existe uma tela. Essa separação é o que permite, por exemplo, usar a `World` em outro projeto que só precise de um contêiner de entidades com ciclo de vida, ou trocar o integrador sem mexer no `MassObject`.

---

## Corpos e entidades

Toda entidade herda de `Entity`, uma classe abstrata que define o contrato mínimo: `update`, `fixed_update`, `render`, mais os métodos opcionais `hit_test`, `click_distance`, `get_world_bounds`, `is_visible`, `render_selection` e o hook `on_destroy`. O ciclo de vida é gerenciado na própria base: `destroy()` marca `active = False` de forma idempotente e dispara o hook.

O `MassObject` é a implementação concreta central. Ele carrega massa, raio, cor, posição, velocidade, e mantém um `previous_position` para interpolação visual. Ele sabe:

- **Colidir**: `overlaps(other)` responde se dois discos se sobrepõem.
- **Fundir-se**: `merge_with(other)` absorve outro corpo conservando massa, momento linear e volume (raio recalculado por `r ∝ m^(1/3)`), respeitando densidade uniforme.
- **Ser clicado**: `hit_test` responde se um ponto está dentro do disco; `click_distance` responde a distância do ponto à superfície, permitindo tolerância de clique.
- **Ser selecionado visualmente**: `render_selection` desenha um anel ao redor quando o corpo é a seleção atual.

O `MassObject` **não** integra a si mesmo. A integração é responsabilidade externa, o que permite trocar de método numérico sem tocar no corpo.

---

## Interação e visualização

A experiência do usuário é construída em torno de três gestos:

**Clique esquerdo** seleciona um corpo. A seleção passa pelo `EntityPicker`, que primeiro tenta um hit direto (o clique está dentro de algum disco?) e, se ninguém for encontrado, considera uma tolerância em pixels (convertida para unidades de mundo pelo zoom), escolhendo o corpo mais próximo. Isso resolve o caso de corpos pequenos ou rápidos que seriam quase impossíveis de acertar. O corpo selecionado ganha um **anel branco** desenhado em `render_selection`.

**Clique direito** sobre um segundo corpo, com o primeiro já selecionado, define uma **referência orbital**. O corpo de referência ganha um **anel laranja**, e o `InspectorHUD` passa a exibir, em laranja, as propriedades relativas — distância, velocidade relativa, excentricidade, tipo de órbita, semi-eixo maior e Δv de escape. A ideia é poder ler a órbita inteira num relance.

**Teclas de alternância** ativam camadas de visualização:

- **F** liga/desliga o vetor de força resultante em cada corpo (usando o `last_force` calculado pelo integrador).
- **T** liga/desliga a previsão de trajetória futura, que roda uma simulação paralela a partir do estado atual e desenha as polilinhas que cada corpo vai seguir.

O HUD ainda exibe, no canto inferior, o tempo do último passo físico (em ms) e, quando a previsão está ligada, o tempo total gasto na previsão.

---

## Previsão de trajetória

O `TrajectoryVisualizer` é a peça mais interessante do ponto de vista de design, porque faz um monte de coisas ao mesmo tempo e cada uma tem um motivo:

- **Estado temporário**: a previsão nunca toca nos corpos reais. Ela copia posição, velocidade, massa e raio para listas planas de floats e integra ali mesmo, com o mesmo esquema semi-implícito da simulação.
- **Culling**: corpos fora da câmera continuam sendo integrados — a gravidade deles afeta os demais — mas a trajetória deles não é acumulada nem desenhada, economizando uma transformação de tela e um par de floats por passo.
- **Parada em colisão**: a predição interrompe no primeiro passo em que qualquer par entra no raio de colisão. A simulação real funde os corpos nesse ponto, então prever adiante seria fisicamente inútil (a trajetória atravessaria a colisão) e visualmente confuso.
- **Desenho em polilinha**: uma chamada de `pygame.draw.lines` por corpo em vez de uma `draw.line` por segmento. Reduz de `steps × count` chamadas para `count`.

---

## Princípios de design que guiaram o projeto

Ao longo do desenvolvimento, algumas decisões se repetiram e viraram o "estilo" do código:

**Separação estrita de camadas.** `MassObject` não sabe integrar. `NewtonianGravity` não sabe desenhar. `World` não sabe física. `EntityPicker` não sabe onde os corpos estão. Cada peça tem uma responsabilidade e a delegação é explícita.

**Inversão via registros, quando faz sentido.** O `Inspector` não conhece `MassObject`. Quem sabe quais propriedades existem é o `main.py`, que as registra num dicionário indexado por tipo. Isso permite adicionar ou remover propriedades sem tocar no inspector, e reaproveitar o mesmo inspector em outras cenas com outros tipos de entidade.

**Ciclo de vida explícito.** `destroy()` é idempotente e dispara um hook `on_destroy`, o que permite que subclasses façam limpeza própria sem duplicar a lógica de estado. A `World` purga inativos em um único ponto, no fim do passo físico, evitando mutação durante iteração.

**Configuração separada de política.** `Configs` descreve a janela e o loop; `SimulationConfig` descreve o mundo e a física. Trocar a simulação significa trocar a instância de `SimulationConfig`, não reescrever código.

**Documentação como parte do código.** Toda função pública tem docstring. Comentários só aparecem onde registram uma decisão não óbvia — por que um determinado default foi escolhido, por que uma ordem de operação importa, por que um caso degenerado é tratado de determinada forma.