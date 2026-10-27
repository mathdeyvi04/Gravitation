import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("checkpoints/training_log.jsonl", lines=True)

# 1. Curva de aprendizado, colorida por α
fig, ax = plt.subplots()
ax.plot(df["gen"], df["fitness"].apply(lambda d: d["best"]), label="best")
ax.plot(df["gen"], df["fitness"].apply(lambda d: d["avg"]), label="avg")
ax.set_xlabel("geração"); ax.set_ylabel("fitness"); ax.legend()
fig.savefig("learning_curve.png")

# 2. Distribuição de terminais ao longo do tempo (stacked bar)
term_df = pd.DataFrame(df["terminals"].tolist()).fillna(0)
term_df.plot(kind="area", stacked=True, figsize=(12, 4))
plt.savefig("terminals.png")

# 3. Onde a população "chega perto": scatter max_r_ratio × reward
rows = []
for _, row in df.iterrows():
    for ind in row["individuals"]:
        rows.append({
            "gen": row["gen"], "alpha": row["alpha"],
            "reward": ind["reward"], "max_r_ratio": ind["max_r_ratio"],
            "terminal": ind["terminal"],
        })
ind_df = pd.DataFrame(rows)
ind_df.plot.scatter(x="max_r_ratio", y="reward", c="gen", cmap="viridis")
plt.savefig("proximity_vs_reward.png")

# 4. Eficiência de combustível nos sucessos
successes = ind_df[ind_df["terminal"] == "success"]
successes["efficiency"] = successes["reward"] / successes["fuel_used"].clip(lower=1)
print(successes.groupby("gen")["efficiency"].mean())