import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 12})

gpu_timings = pd.read_csv("./example/mmcanear_timings_archive/gpu_timings.csv")
mac_timings = pd.read_csv("./example/mmcanear_timings_archive/timings.csv")

gpu_timings["platform"] = "AMD"
mac_timings["platform"] = "Apple M4"

all_timings = pd.concat([gpu_timings, mac_timings], ignore_index=True)


fig, ax = plt.subplots(figsize=(10, 7), nrows=2, sharey=True)
x_ticks = [10, 50, 100, 500, 1000, 2000, 5000]

for i, (platform, plot_df) in enumerate(all_timings.groupby("platform")):
    for label, label_df in plot_df.groupby(["label", "device"]):
        lab, device = label
        clean_label = {
            "jax_ridge": "JAX",
            "np_ridge": "NumPy",
            "jax_ridge_jit": "JAX (jit)",
            "gpu": "GPU",
            "cpu": "CPU",
        }
        ax[i].plot(
            label_df["n_sims"],
            label_df["elapsed_time"],
            marker="o",
            label=f"{clean_label[lab]}-{clean_label[device]}",
        )
    ax[i].set_xscale("log")
    ax[i].set_xticks(x_ticks)
    ax[i].set_xticklabels(x_ticks)
    ax[i].set_yscale("log")
    ax[i].set_title(platform)
    ax[i].legend(title="Method-Device", loc="upper left")
    ax[i].set_xlabel("Number of Simulations")
    ax[i].set_ylabel("Elapsed Time (seconds)")
fig.tight_layout()
fig.suptitle("Simulation Timings by Platform and Method", y=1.02)

fig.savefig("./docs/simulation_timings.png", bbox_inches="tight")
