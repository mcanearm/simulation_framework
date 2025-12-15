import pandas as pd
from matplotlib import pyplot as plt

gpu_timings = pd.read_csv("./example/gpu_timings.csv")
mac_timings = pd.read_csv("./example/timings.csv")

gpu_timings["platform"] = "AMD"
mac_timings["platform"] = "Apple M4"


all_timings = pd.concat([gpu_timings, mac_timings], ignore_index=True)


fig, ax = plt.subplots(figsize=(8, 6), nrows=2)
for i, (platform, plot_df) in enumerate(all_timings.groupby("platform")):
    print(i)
    for label, label_df in plot_df.groupby(["label", "device"]):
        lab, device = label
        ax[i].plot(
            label_df["n_sims"],
            label_df["elapsed_time"],
            marker="o",
            label=f"{lab}-{device}",
        )
