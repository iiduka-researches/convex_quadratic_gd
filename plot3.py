import csv
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np


def clip_invalid_values(y, min_val=1e-10, max_val=1e30):
    y = np.array(y)
    y = np.where(np.isnan(y) | np.isinf(y), max_val, y)
    y = np.clip(y, min_val, max_val)
    return y


# スケジューラ名のリスト（outputディレクトリにあるもの）
schedulers = ["constant", "constant2", "constant3"]

scheduler_labels = {
    "constant": "定数ステップサイズ（$\\bar{\\eta} = 0.01$）",
    "constant2": "定数ステップサイズ（$\\bar{\\eta} = 0.02$）",
    "constant3": "定数ステップサイズ（$\\bar{\\eta} = 0.03$）",
    "cosine": "コサインステップサイズ",
    "linear_decay": "線形減衰ステップサイズ",
    "exp": "指数増加ステップサイズ",
    "decay": "減衰ステップサイズ",
    "exp_warmup": "ウォームアップステップサイズ"
}

# カラー版の色・スタイル
colors = ["blue", "green", "red", "orange", "purple", "cyan"]
linestyles = ["-"] * len(colors)

# 白黒版の色（全部黒）と異なる線種
black_colors = ["black", "black", "black", "gray", "gray", "gray"]
black_linestyles = ["-", "--", "-.", "-", "--", "-."]

# 各スケジューラのデータを格納
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
                eta_list.append(None)  # NaN対応
    data[scheduler] = {
        "t": t_list,
        "f": f_list,
        "grad": grad_list,
        "eta": eta_list
    }

# カラー版・白黒版を切り替えて2回ループ
for mode in ["color", "bw"]:
    mode_suffix = "_bw" if mode == "bw" else ""
    current_colors = black_colors if mode == "bw" else colors
    current_linestyles = black_linestyles if mode == "bw" else linestyles

    # -------- Plot 1: ステップサイズ η_t --------
    plt.figure(figsize=(8, 5))
    for i, scheduler in enumerate(schedulers):
        plt.plot(data[scheduler]["t"][:-1], data[scheduler]["eta"][:-1],
                 label=scheduler_labels[scheduler], color=current_colors[i], linestyle=current_linestyles[i])
    plt.xlabel("ステップ数 t")
    plt.ylabel("ステップサイズ $\\eta_t$")
    plt.ylim(1e-12, 0.035)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"3step_size_plot{mode_suffix}.pdf")

    # -------- Plot 2: 関数値 f --------
    plt.figure(figsize=(8, 5))
    for i, scheduler in enumerate(schedulers):
        y = clip_invalid_values(data[scheduler]["f"])
        plt.plot(data[scheduler]["t"], y,
                 label=scheduler_labels[scheduler], color=current_colors[i], linestyle=current_linestyles[i])
    plt.xlabel("ステップ数 t")
    plt.ylabel("関数値 $f(\\theta_t)$")
    plt.yscale("log")
    plt.ylim(1e-12, 1e20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"3function_value_plot{mode_suffix}.pdf")

    # -------- Plot 3: 勾配ノルム ||∇f|| --------
    plt.figure(figsize=(8, 5))
    for i, scheduler in enumerate(schedulers):
        y = clip_invalid_values(data[scheduler]["grad"])
        plt.plot(data[scheduler]["t"], y,
                 label=scheduler_labels[scheduler], color=current_colors[i], linestyle=current_linestyles[i])
    plt.xlabel("ステップ数 t")
    plt.ylabel("勾配ノルム $\\|\\nabla f(\\theta_t)\\|$")
    plt.yscale("log")
    plt.ylim(1e-6, 1e20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"3gradient_norm_plot{mode_suffix}.pdf")
