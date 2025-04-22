import csv
import japanize_matplotlib
import matplotlib.pyplot as plt

schedulers = ["constant", "decay", "linear_decay", "cosine", "exp", "exp_warmup"]

scheduler_labels = {
    "constant": "定数ステップサイズ",
    "cosine": "コサインステップサイズ",
    "linear_decay": "線形減衰ステップサイズ",
    "exp": "指数増加ステップサイズ",
    "decay": "減衰ステップサイズ",
    "exp_warmup": "ウォームアップステップサイズ"
}

colors = ["blue", "green", "red", "orange", "purple", "cyan"]
linestyles = ["-"] * len(colors)

black_colors = ["black", "black", "black", "gray", "gray", "gray"]
black_linestyles = ["-", "--", "-.", "-", "--", "-."]

data = {}
for scheduler in schedulers:
    filepath = f"output/{scheduler}/average.csv"
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        t_list, f_list, grad_list, eta_list = [], [], [], []
        for row in reader:
            t_list.append(int(row["t"]))
            f_list.append(float(row["f_theta"]))
            grad_list.append(float(row["grad_f"]))
            try:
                eta_list.append(float(row["eta_t"]))
            except ValueError:
                eta_list.append(None)
    data[scheduler] = {
        "t": t_list,
        "f": f_list,
        "grad": grad_list,
        "eta": eta_list
    }

for mode in ["color", "bw"]:
    mode_suffix = "_bw" if mode == "bw" else ""
    current_colors = black_colors if mode == "bw" else colors
    current_linestyles = black_linestyles if mode == "bw" else linestyles

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))  # 横一列に3つ
    axes[0].set_position([0.125, 0.08, 0.35, 0.75])  # 下から少し上に、縦幅0.55
    axes[1].set_position([0.55, 0.075, 0.3, 0.7])

    # Plot 2: f(θ)
    for i, scheduler in enumerate(schedulers):
        axes[0].plot(data[scheduler]["t"], data[scheduler]["f"],
                     label=scheduler_labels[scheduler], color=current_colors[i], linestyle=current_linestyles[i])
    axes[0].set_xlabel("ステップ数 t")
    axes[0].set_ylabel("関数値 $f(\\theta_t)$")
    axes[0].set_ylim(1e-12, 1e4)
    axes[0].set_yscale("log")
    axes[0].grid(True)

    # Plot 3: ||∇f||
    for i, scheduler in enumerate(schedulers):
        axes[1].plot(data[scheduler]["t"], data[scheduler]["grad"],
                     label=scheduler_labels[scheduler], color=current_colors[i], linestyle=current_linestyles[i])
    axes[1].set_xlabel("ステップ数 t")
    axes[1].set_ylabel("勾配ノルム $\\|\\nabla f(\\theta_t)\\|$")
    axes[1].set_ylim(1e-6, 1e4)
    axes[1].set_yscale("log")
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    # 凡例はまとめて一番下に
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 凡例のスペース確保
    plt.savefig(f"comparison_plots{mode_suffix}.pdf")
    plt.close()
