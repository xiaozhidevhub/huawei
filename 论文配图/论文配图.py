# -*- coding: utf-8 -*-
"""
论文配图
========

四道题的计算脚本已经把表格、JSON 和少量图写进「结果」文件夹。
本文件不再读取原始大文件，只根据这些已经算好的结果，补一批适合放进论文的图。

原来的图留在「结果」文件夹。本脚本和它画出的新图都放在「论文配图」文件夹。

单独出图（在仓库根目录执行）：
    python 论文配图/论文配图.py

四道题脚本在各自算完后，也会调用下面对应的函数。
因此按原来的顺序重跑四道题时，这些图会一起更新。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# 服务器上没有显示器，必须先选好这种“只保存文件”的后端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

# 本文件放在「论文配图」文件夹。仓库根目录是它的上一级。
# 表格仍从「结果」读取，新图画回本文件夹下的问题一至问题四。
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA1 = ROOT / "结果" / "问题一"
DATA2 = ROOT / "结果" / "问题二"
DATA3 = ROOT / "结果" / "问题三"
DATA4 = ROOT / "结果" / "问题四"
FIG1 = HERE / "问题一"
FIG2 = HERE / "问题二"
FIG3 = HERE / "问题三"
FIG4 = HERE / "问题四"

# 色盲友好的一组颜色，打印成黑白时深浅也还能分开。
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GRAY = "#7A7A7A"
INK = "#222222"


def use_paper_style():
    """统一字体、字号和边框，让四道题的图看起来像同一篇论文。"""
    # 当前环境装的是文泉驿微米黑。Windows 上如果有微软雅黑，也会优先使用。
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def clean_axes(ax):
    """去掉上、右边框。论文图里通常只留左、下两条轴。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, path: Path):
    """按论文常用的 300 dpi 保存，并裁掉多余空白。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存 {path.relative_to(ROOT)}")


def read_json(path: Path) -> dict:
    """读取某一问写出的 JSON。文件不存在时，提示先运行对应的问题脚本。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}。请先运行对应的问题脚本。")
    return json.loads(path.read_text(encoding="utf-8"))


def loss_of(n_billion, d_billion, q_value, law: dict, q_ref: float):
    """问题二的广义标度律。N、D 的单位都是十亿。

    Q 等于参照质量时，最后一项的倍数是 1，公式退回经典标度律。
    """
    q_safe = np.maximum(q_value, 1e-3)
    factor = (q_ref / q_safe) ** law["gamma"]
    n_term = law["A"] * np.power(n_billion, -law["alpha"])
    d_term = law["B"] * np.power(d_billion, -law["beta"]) * factor
    return law["E"] + n_term + d_term


# 指标中文名。熵权是在“反向之后”计算的，所以负向指标写成“低××”。
INDICATOR_LABEL = {
    "ad_en": "低广告",
    "rps_doc_word_count": "文档词数",
    "rps_doc_num_sentences": "句子数",
    "modernbert_reasoning": "推理性",
    "modernbert_cleanliness": "干净程度",
    "rps_doc_mean_word_length": "平均词长",
    "rps_lines_ending_with_terminal_punctution_mark": "句末标点",
    "fluency_en": "流畅度",
    "modernbert_readability": "可读性",
    "rps_doc_frac_unique_words": "独特词比例",
    "qurater": "综合评分",
    "fineweb_edu": "教育价值",
    "modernbert_professionalism": "专业性",
    "rps_doc_unigram_entropy": "用词丰富度",
    "rps_lines_uppercase_letter_fraction": "低大写比例",
    "rps_doc_frac_no_alph_words": "低非字母词",
    "rps_doc_frac_chars_top_2gram": "低二字重复",
    "rps_lines_numerical_chars_fraction": "低数字比例",
    "rps_doc_frac_chars_top_3gram": "低三字重复",
    "dsir_math": "数学相似度",
    "dsir_books": "书籍相似度",
    "dsir_wiki": "百科相似度",
}

# 和问题一脚本里的分组一致，用来给熵权图上色。
CONTENT_GROUP = {
    "fineweb_edu",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
}
HYGIENE_GROUP = {
    "modernbert_cleanliness",
    "modernbert_readability",
    "fluency_en",
    "ad_en",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_ending_with_terminal_punctution_mark",
}

# 配比域的短中文名。图里地方小，名字不宜太长。
DOMAIN_LABEL = {
    "dm_mathematics": "数学",
    "hackernews": "Hacker News",
    "ubuntu_irc": "Ubuntu IRC",
    "gutenberg_pg_19": "古腾堡",
    "book": "书籍",
    "arxiv": "arXiv",
    "github": "GitHub",
    "wikipedia": "维基百科",
    "wikipedia_en": "维基百科",
    "stackexchange": "StackExchange",
    "commoncrawl": "Common Crawl",
    "pile_cc": "Pile-CC",
    "c4": "C4",
}


def draw_problem1():
    """问题一：熵权、抽样对照、最优配比、跨规模稳健性、质量与系数。"""
    use_paper_style()
    weights = pd.read_csv(DATA1 / "熵权.csv", index_col=0).squeeze("columns")
    contrast = pd.read_csv(DATA1 / "抽样与扩展对照.csv")
    conflict = pd.read_csv(DATA1 / "冲突比例.csv", index_col=0).squeeze("columns")
    mixture = pd.read_csv(DATA1 / "最优配比.csv")
    checks = pd.read_csv(DATA1 / "配比模型检验.csv")
    linked = pd.read_csv(DATA1 / "质量与系数对照.csv")
    summary = read_json(DATA1 / "问题一输出.json")

    _plot_entropy(weights)
    _plot_sample_vs_extended(contrast, conflict)
    _plot_optimal_mixture(mixture, summary["Q_ref"])
    _plot_scale_robustness(checks)
    _plot_quality_vs_coef(linked, summary["spearman_Q_vs_coef"])


def _plot_entropy(weights: pd.Series):
    """熵权：差异越大的指标，权重越高。"""
    table = weights.sort_values(ascending=True)
    colors = []
    for name in table.index:
        if name in CONTENT_GROUP:
            colors.append(BLUE)
        elif name in HYGIENE_GROUP:
            colors.append(GREEN)
        else:
            colors.append(GRAY)
    labels = [INDICATOR_LABEL.get(name, name) for name in table.index]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.barh(labels, table.to_numpy(), color=colors, height=0.72, zorder=2)
    clean_axes(ax)
    ax.set_xlabel("熵权")
    ax.set_title("问题一：22 个质量指标的熵权")
    # 在最高的几根柱子末端标出数值，方便正文直接引用。
    for i, value in enumerate(table.to_numpy()):
        if value >= 0.05:
            ax.text(value + 0.004, i, f"{value:.2f}", va="center", fontsize=8, color=INK)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="内容价值"),
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="文本卫生"),
        plt.Rectangle((0, 0), 1, 1, color=GRAY, label="长度与形式"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlim(0, table.max() * 1.18)
    save_figure(fig, FIG1 / "熵权.png")


def _plot_sample_vs_extended(contrast: pd.DataFrame, conflict: pd.Series):
    """左图比较平均质量，右图比较冲突比例。两套数据应当接近。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))

    ax = axes[0]
    x = np.arange(len(contrast))
    width = 0.36
    sample_q = contrast["sample_Q"].to_numpy()
    extended_q = contrast["extended_Q"].to_numpy()
    ax.bar(x - width / 2, sample_q, width, color=SKY, label="抽样集", zorder=2)
    ax.bar(x + width / 2, extended_q, width, color=BLUE, label="扩展全量", zorder=2)
    labels = []
    for _, row in contrast.iterrows():
        labels.append(f"{DOMAIN_LABEL.get(row['domain'], row['domain'])}\nn={int(row['extended_n'])}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("领域平均质量分 Q")
    ax.set_ylim(0, max(sample_q.max(), extended_q.max()) * 1.28)
    ax.set_title("(a) 抽样集与扩展全量")
    clean_axes(ax)
    ax.legend(loc="upper right")
    for i, diff in enumerate(contrast["Q_diff"]):
        y = max(sample_q[i], extended_q[i]) + 0.015
        ax.text(i, y, f"差 {diff:+.3f}", ha="center", fontsize=8, color=INK)

    ax = axes[1]
    # 冲突比例表的三行顺序不固定，这里按论文叙述的顺序重排。
    order = ["sample", "extended_arxiv", "extended_github"]
    name_map = {"sample": "抽样集", "extended_arxiv": "arXiv 扩展", "extended_github": "GitHub 扩展"}
    rates = [float(conflict[key]) for key in order]
    colors = [SKY, BLUE, ORANGE]
    bars = ax.bar([name_map[key] for key in order], rates, color=colors, width=0.62, zorder=2)
    ax.set_ylabel("冲突样本比例")
    ax.set_ylim(0, max(rates) * 1.35)
    ax.set_title("(b) 内容与卫生显著背离的比例")
    clean_axes(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:.0%}"))
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.004, f"{rate:.1%}", ha="center", fontsize=9)

    fig.suptitle("问题一：质量分和冲突比例在扩展集上仍然接近", y=1.02, fontsize=13)
    save_figure(fig, FIG1 / "抽样扩展与冲突.png")


def _plot_optimal_mixture(mixture: pd.DataFrame, q_ref: float):
    """单域上限是 30%。只画出真正分到比例的领域。"""
    show = mixture[mixture["p"] > 0.001].sort_values("p", ascending=True)
    labels = [DOMAIN_LABEL.get(name, name) for name in show["domain"]]
    share = show["p"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(labels, share, color=[BLUE, GREEN, ORANGE, PURPLE][: len(share)][::-1], height=0.62, zorder=2)
    ax.axvline(30, color=VERMILION, linestyle="--", linewidth=1.1, label="单域上限 30%")
    clean_axes(ax)
    ax.set_xlabel("训练配比（%）")
    ax.set_xlim(0, 42)
    ax.set_title(f"问题一：最优领域配比（参照质量 Q_ref = {q_ref:.3f}）")
    for i, value in enumerate(share):
        ax.text(value + 0.6, i, f"{value:.0f}%", va="center", fontsize=10)
    ax.legend(loc="lower right")
    save_figure(fig, FIG1 / "最优配比.png")


def _plot_scale_robustness(checks: pd.DataFrame):
    """绝对误差随模型变大会失去可比性，所以这里只画排序相关。"""
    label_map = {
        "train_1m": "训练 1M",
        "test_1m": "检验 1M",
        "test_60m": "检验 60M",
        "test_1B": "检验 1B",
        "est_10b": "外推 10B",
        "est_70b": "外推 70B",
    }
    # 同一规模、更大但仍是实测、纯外推，用三种颜色分开。
    color_map = {
        "train_1m": BLUE,
        "test_1m": BLUE,
        "test_60m": ORANGE,
        "test_1B": ORANGE,
        "est_10b": VERMILION,
        "est_70b": VERMILION,
    }
    labels = [label_map[name] for name in checks["split"]]
    values = checks["spearman"].to_numpy()
    colors = [color_map[name] for name in checks["split"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.68, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    clean_axes(ax)
    ax.set_ylabel("Spearman 排序相关")
    ax.set_ylim(-0.75, 0.9)
    ax.set_title("问题一：配比模型在不同规模上的排序能力")
    for bar, value in zip(bars, values):
        offset = 0.03 if value >= 0 else -0.06
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}", ha="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="与训练同规模"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="更大实测规模"),
        plt.Rectangle((0, 0), 1, 1, color=VERMILION, label="外推表"),
    ]
    ax.legend(handles=handles, loc="upper right")
    save_figure(fig, FIG1 / "跨规模排序稳健性.png")


def _plot_quality_vs_coef(linked: pd.DataFrame, spearman: float):
    """只保留能对上质量分的领域。系数为正表示加大该域会抬高损失。"""
    points = linked.dropna(subset=["Q_domain"]).copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(points["Q_domain"], points["coef"], s=55, color=BLUE, zorder=3)
    for _, row in points.iterrows():
        label = DOMAIN_LABEL.get(row["quality_domain"], row["domain"])
        ax.annotate(
            label,
            (row["Q_domain"], row["coef"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    clean_axes(ax)
    ax.set_xlabel("领域质量分 Q")
    ax.set_ylabel("岭回归系数")
    ax.set_title(f"问题一：质量分与损失系数（Spearman = {spearman:.2f}）")
    save_figure(fig, FIG1 / "质量与系数散点.png")


def draw_problem2():
    """问题二：等高线、质量曲线、外部验证、质量指数、弹性替代。"""
    use_paper_style()
    law = read_json(DATA2 / "问题二输出.json")
    checks = pd.read_csv(DATA2 / "标度律验证.csv")
    gamma_table = pd.read_csv(DATA2 / "质量指数估计.csv")
    _plot_loss_contour(law)
    _plot_quality_curves(law)
    _plot_external_validation(checks)
    _plot_gamma(gamma_table)
    _plot_elasticity(law)


def _plot_loss_contour(law: dict):
    """在对数坐标上画出经典标度律的预测损失。星号是后文使用的代表点。"""
    q_ref = law["Q_ref"]
    n_grid = np.logspace(np.log10(0.2), np.log10(80), 220)
    d_grid = np.logspace(np.log10(8), np.log10(3000), 220)
    n_mesh, d_mesh = np.meshgrid(n_grid, d_grid)
    # Q 取参照值，质量倍数为 1，画的就是经典公式。
    z = loss_of(n_mesh, d_mesh, q_ref, law, q_ref)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    # 只标少数几条等值线，并给数字加白底，避免和色带混在一起。
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 12)
    contour = ax.contourf(n_mesh, d_mesh, z, levels=levels, cmap="viridis")
    label_levels = np.linspace(np.nanpercentile(z, 12), np.nanpercentile(z, 88), 5)
    lines = ax.contour(n_mesh, d_mesh, z, levels=label_levels, colors="white", linewidths=0.7)
    texts = ax.clabel(lines, fmt="%.2f", fontsize=8)
    for text in texts:
        text.set_color(INK)
        text.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.85})
    ax.scatter(
        [6.9],
        [300],
        marker="*",
        s=220,
        color=VERMILION,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="代表点 6.9B / 300B",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("参数量 N（十亿）")
    ax.set_ylabel("数据量 D（十亿 token）")
    ax.set_title("问题二：经典标度律预测的验证损失")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.02)
    colorbar.set_label("预测验证损失")
    ax.legend(loc="upper left")
    save_figure(fig, FIG2 / "损失等高线.png")


def _plot_quality_curves(law: dict):
    """固定 6.9B 参数，比较不同质量下损失怎样随数据量下降。"""
    q_ref = float(law["Q_ref"])
    d_grid = np.logspace(np.log10(20), np.log10(2000), 200)
    qualities = [0.20, q_ref, 0.50, 0.70]
    colors = [GRAY, BLUE, ORANGE, GREEN]
    labels = ["Q = 0.20", f"Q = Q_ref ({q_ref:.3f})", "Q = 0.50", "Q = 0.70"]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    base = loss_of(6.9, d_grid, q_ref, law, q_ref)
    for q_value, color, label in zip(qualities, colors, labels):
        pred = loss_of(6.9, d_grid, q_value, law, q_ref)
        width = 2.2 if abs(q_value - q_ref) < 1e-8 else 1.4
        axes[0].plot(d_grid, pred, color=color, linewidth=width, label=label)
        axes[1].plot(d_grid, pred - base, color=color, linewidth=width, label=label)
    for ax in axes:
        ax.set_xscale("log")
        clean_axes(ax)
        ax.set_xlabel("数据量 D（十亿 token）")
    axes[0].set_ylabel("预测验证损失")
    axes[0].set_title("(a) 损失水平")
    axes[1].axhline(0, color=INK, linewidth=0.7)
    axes[1].set_ylabel("相对参照质量的损失变化")
    axes[1].set_title("(b) 质量带来的损失差")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("问题二：固定 N = 6.9B 时，数据质量如何改变损失", y=1.03, fontsize=13)
    save_figure(fig, FIG2 / "质量对损失的影响.png")


def _plot_external_validation(checks: pd.DataFrame):
    """外部数据才能说明公式能不能搬走。插值和估算单独标成浅色。"""
    label_map = {
        "B4_跨族收敛点": "B4 跨族真实点",
        "B5_文献标度律": "B5 文献点",
        "B2_Cerebras半合成轨迹": "B2 Cerebras",
        "B3_Pythia插值轨迹": "B3 插值轨迹",
        "B10_大模型预估损失": "B10 估算损失",
    }
    # 真实外部检验用深色；插值和估算损失不能当作新证据。
    evidence = {"B4_跨族收敛点", "B5_文献标度律", "B2_Cerebras半合成轨迹"}
    order = ["B4_跨族收敛点", "B5_文献标度律", "B2_Cerebras半合成轨迹", "B3_Pythia插值轨迹", "B10_大模型预估损失"]
    table = checks.set_index("data").loc[order]
    labels = [label_map[name] for name in order]
    r2 = table["r2"].to_numpy()
    # B2 的 R² 约等于 -5。若按真实长度去画，其他柱子会挤成一条线。
    # 图上把它截到 -1，并用文字写出真实数值。
    shown = np.maximum(r2, -1.0)
    colors = [BLUE if name in evidence else "#C8C8C8" for name in order]
    colors[2] = VERMILION

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    bars = ax.barh(labels[::-1], shown[::-1], color=colors[::-1], height=0.62, zorder=2)
    ax.axvline(0, color=INK, linewidth=0.8)
    clean_axes(ax)
    ax.set_xlabel("R²（B2 的柱已截断，真实值标在文字里）")
    ax.set_xlim(-1.35, 1.25)
    ax.set_title("问题二：经典标度律换到外部数据后的拟合")
    for bar, value in zip(bars, r2[::-1]):
        y = bar.get_y() + bar.get_height() / 2
        if value < -1:
            # 柱子被截断后，真实数值写在 0 的右侧，避免白字压在柱子上看不清。
            ax.text(0.06, y, f"真实 R² = {value:.2f}，柱已截断", va="center", ha="left", fontsize=8)
        else:
            ax.text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=8)
    save_figure(fig, FIG2 / "外部验证.png")


def _plot_gamma(gamma_table: pd.DataFrame):
    """B6 是主结果，B7 是对照，B8 含外推且得不到有用的正指数。"""
    labels = ["B6 半合成基础", "B7 半合成扩展", "B8 含外推"]
    values = gamma_table["gamma"].to_numpy()
    colors = [BLUE, SKY, GRAY]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=2)
    clean_axes(ax)
    ax.set_ylabel("质量指数 γ")
    ax.set_title("问题二：质量指数只从半合成表估计")
    ymax = max(0.45, float(np.max(values)) * 1.25)
    ax.set_ylim(0, ymax)
    for bar, value, rmse in zip(bars, values, gamma_table["rmse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.03,
            f"γ = {value:.2f}\nRMSE = {rmse:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, FIG2 / "质量指数对照.png")


def equivalent_n_ratio(delta_q: float, law: dict) -> float:
    """质量提高 delta_q 后，若仍用原来的质量，参数量要变成原来的多少倍。

    先算出“质量已经提高”时的损失，再反解经典公式里的 N。
    数据项不变，因为比较时 D 和原来的 Q 都固定。
    """
    q_ref = float(law["Q_ref"])
    n0 = 6.9
    d0 = 300.0
    target = float(loss_of(n0, d0, q_ref + delta_q, law, q_ref))
    factor = 1.0  # 仍停留在参照质量
    data_term = law["B"] * (d0 ** (-law["beta"])) * factor
    n_term = target - law["E"] - data_term
    # n_term = A * N^{-alpha}，所以 N = (n_term / A) ^ (-1/alpha)
    n_new = (n_term / law["A"]) ** (-1.0 / law["alpha"])
    return float(n_new / n0)


def _plot_elasticity(law: dict):
    """左图是代表点上的弹性，右图是质量提高对应的参数倍数。"""
    elas = law["elasticity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    ax = axes[0]
    names = ["参数量 N", "数据量 D", "数据质量 Q"]
    values = [elas["eps_N"], elas["eps_D"], elas["eps_Q"]]
    colors = [BLUE, ORANGE, GREEN]
    bars = ax.bar(names, values, color=colors, width=0.62, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    clean_axes(ax)
    ax.set_ylabel("弹性（因素 +1% 时，损失约变化多少百分比）")
    ax.set_title("(a) 代表点上的三种弹性")
    ax.set_ylim(min(values) - 0.012, 0.01)
    for bar, value in zip(bars, values):
        # 数字放在柱子下方的空白处，不压在彩色柱子上。
        ax.text(bar.get_x() + bar.get_width() / 2, value - 0.0015, f"{value:.3f}", ha="center", va="top", fontsize=8)

    ax = axes[1]
    delta = np.linspace(0.0, 0.30, 61)
    ratio = np.array([equivalent_n_ratio(float(item), law) for item in delta])
    ax.plot(delta, ratio, color=BLUE, linewidth=2.0)
    ax.scatter([0.1], [law["quality_delta_0.1"]["N_ratio"]], color=VERMILION, s=40, zorder=3)
    # 文字放在曲线左上方的空白区，箭头再指回质量 +0.1 的那个点。
    ax.annotate(
        f"质量 +0.1\n参数 ×{law['quality_delta_0.1']['N_ratio']:.2f}",
        xy=(0.1, law["quality_delta_0.1"]["N_ratio"]),
        xytext=(0.01, 1.95),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": VERMILION},
        color=VERMILION,
    )
    clean_axes(ax)
    ax.set_xlabel("质量提高量 ΔQ")
    ax.set_ylabel("等价的参数量倍数")
    ax.set_title("(b) 保持损失不变时的参数替代")
    fig.suptitle("问题二：N = 6.9B，D = 300B，Q = Q_ref", y=1.03, fontsize=13)
    save_figure(fig, FIG2 / "弹性与等损失替代.png")


def extra_quality_cost(q_value, q0: float, kind: str):
    """每多 1 个 token，把质量从基线抬到 q_value 要额外花多少 FLOPs。

    三个公式与问题三脚本中的定义相同，来自题目给出的成本假设。
    """
    if kind == "指数型":
        raw = 1e7 * np.exp(6.0 * q_value)
        base = 1e7 * np.exp(6.0 * q0)
    elif kind == "幂函数型":
        raw = 5e9 * np.power(q_value, 4.0)
        base = 5e9 * np.power(q0, 4.0)
    else:
        raw = 2e9 * np.log1p(10.0 * q_value)
        base = 2e9 * np.log1p(10.0 * q0)
    return np.maximum(raw - base, 0.0)


def draw_problem3():
    """问题三：花费结构、规模路径、上下文临界、成本形状、损失下降。"""
    use_paper_style()
    result = pd.read_csv(DATA3 / "三档预算最优配置.csv")
    sens = pd.read_csv(DATA3 / "上下文敏感性.csv")
    summary = read_json(DATA3 / "问题三输出.json")
    _plot_spend_structure(result)
    _plot_scale_path(result)
    _plot_context_threshold(sens, summary["L_ctx_crit"])
    _plot_cost_shapes(summary["Q0"])
    _plot_loss_vs_budget(result)


def _budget_tick(value: float) -> str:
    """把 1e19 写成论文里更易读的 10^19。"""
    exponent = int(round(np.log10(value)))
    return f"10$^{{{exponent}}}$"


def _plot_spend_structure(result: pd.DataFrame):
    """质量放在最底下，才能用 5% 和 15% 两条线读出策略分界。"""
    cost_names = ["指数型", "幂函数型", "对数渐进型"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.3), sharey=True)
    for ax, cost_name in zip(axes, cost_names):
        part = result[result["cost"] == cost_name].sort_values("budget")
        x = np.arange(len(part))
        quality = part["share_quality"].to_numpy()
        train = part["share_train"].to_numpy()
        attn = part["share_attn"].to_numpy()
        ax.bar(x, quality, color=ORANGE, width=0.62, label="质量", zorder=2)
        ax.bar(x, train, bottom=quality, color=BLUE, width=0.62, label="训练", zorder=2)
        ax.bar(x, attn, bottom=quality + train, color=GREEN, width=0.62, label="注意力", zorder=2)
        ax.axhline(0.05, color=INK, linestyle=":", linewidth=0.8)
        ax.axhline(0.15, color=VERMILION, linestyle="--", linewidth=0.8)
        short_strategy = {"规模与质量均衡": "均衡"}
        labels = []
        for _, row in part.iterrows():
            strategy = short_strategy.get(row["strategy"], row["strategy"])
            labels.append(f"{_budget_tick(row['budget'])}\n{strategy}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(cost_name)
        ax.set_ylim(0, 1.08)
        clean_axes(ax)
    axes[0].set_ylabel("算力花费占比")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:.0%}"))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="质量"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="训练"),
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="注意力"),
        Line2D([0], [0], color=VERMILION, linestyle="--", label="质量优先分界 15%"),
        Line2D([0], [0], color=INK, linestyle=":", label="规模优先分界 5%"),
    ]
    # 图例放在三张子图下面，避免挡住最右边那根柱子。
    fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.18)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    fig.suptitle("问题三：三档预算下的算力花费结构（上下文 8192）", y=0.96, fontsize=13)
    save_figure(fig, FIG3 / "花费结构.png")


def _plot_scale_path(result: pd.DataFrame):
    """预算每提高几个数量级，最优参数量和数据量也跟着跳。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    styles = {"指数型": (BLUE, "o"), "幂函数型": (ORANGE, "s"), "对数渐进型": (GREEN, "D")}
    for cost_name, (color, marker) in styles.items():
        part = result[result["cost"] == cost_name].sort_values("budget")
        axes[0].plot(part["budget"], part["N_B"], color=color, marker=marker, label=cost_name)
        axes[1].plot(part["budget"], part["D_B"], color=color, marker=marker, label=cost_name)
    for ax, ylabel, title in [
        (axes[0], "最优参数量 N（十亿）", "(a) 参数量"),
        (axes[1], "最优数据量 D（十亿 token）", "(b) 数据量"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        clean_axes(ax)
        ax.set_xlabel("算力预算（FLOPs）")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    fig.suptitle("问题三：最优规模随预算上升", y=1.03, fontsize=13)
    save_figure(fig, FIG3 / "最优规模随预算.png")


def _plot_context_threshold(sens: pd.DataFrame, lctx_crit: float):
    """窗口一旦超过 6/η，注意力开始吃掉一半以上预算，最优参数量下降。"""
    sens = sens.sort_values("L_ctx")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(sens["L_ctx"], sens["share_attn"], color=GREEN, marker="o", linewidth=2.0, label="注意力占比")
    ax2.plot(sens["L_ctx"], sens["N_B"], color=BLUE, marker="s", linewidth=1.6, label="最优参数量")
    ax1.axvline(lctx_crit, color=VERMILION, linestyle="--", linewidth=1.1)
    # 说明文字放在临界线左下方，避开两条曲线的交叉处。
    ax1.text(lctx_crit * 0.55, 0.06, f"临界长度 {lctx_crit:.0f}", color=VERMILION, fontsize=8, ha="right")
    ax1.set_xscale("log")
    ax1.set_xlabel("上下文长度")
    ax1.set_ylabel("注意力花费占比", color=GREEN)
    ax2.set_ylabel("最优参数量 N（十亿）", color=BLUE)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:.0%}"))
    ax1.set_title("问题三：上下文超过临界长度后，注意力挤压参数量")
    ax1.spines["top"].set_visible(False)
    # 右轴要留给参数量，所以只去掉上边框。
    ax2.spines["top"].set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    save_figure(fig, FIG3 / "上下文临界长度.png")


def _plot_cost_shapes(q0: float):
    """三种成本的绝对大小差很多，所以纵轴用对数。"""
    q_grid = np.linspace(q0 + 0.01, 0.98, 200)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    styles = [("指数型", BLUE), ("幂函数型", ORANGE), ("对数渐进型", GREEN)]
    for kind, color in styles:
        cost = extra_quality_cost(q_grid, q0, kind)
        ax.plot(q_grid, cost, color=color, linewidth=2.0, label=kind)
    ax.set_yscale("log")
    clean_axes(ax)
    ax.set_xlabel("目标数据质量 Q")
    ax.set_ylabel("每个 token 的额外质量成本（FLOPs）")
    ax.set_title(f"问题三：三种质量成本的形状（基线 Q0 = {q0:.3f}）")
    ax.legend(loc="upper left")
    save_figure(fig, FIG3 / "质量成本形状.png")


def _plot_loss_vs_budget(result: pd.DataFrame):
    """预算增大后，三种成本函数找到的最低损失几乎走在一起。"""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    styles = {"指数型": (BLUE, "o"), "幂函数型": (ORANGE, "s"), "对数渐进型": (GREEN, "D")}
    for cost_name, (color, marker) in styles.items():
        part = result[result["cost"] == cost_name].sort_values("budget")
        ax.plot(part["budget"], part["loss"], color=color, marker=marker, linewidth=1.8, label=cost_name)
    # 低预算上三种损失还分得开，不额外写数字。中、高预算几乎重合，每个预算只标一次。
    for budget, part in result.groupby("budget"):
        if part["loss"].max() - part["loss"].min() > 0.02:
            continue
        ax.annotate(
            f"{part['loss'].mean():.2f}",
            (budget, part["loss"].max()),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xscale("log")
    clean_axes(ax)
    ax.set_xlabel("算力预算（FLOPs）")
    ax.set_ylabel("最优预测验证损失")
    ax.set_title("问题三：预算提高后，最优损失随之下移")
    ax.legend(loc="upper right")
    save_figure(fig, FIG3 / "最优损失随预算.png")


def draw_problem4():
    """问题四：贡献分解、宏观规模、损失映射、BBH 分布、前沿情景。"""
    use_paper_style()
    decom = pd.read_csv(DATA4 / "规模与技术进步分解.csv")
    macro = pd.read_csv(DATA4 / "开源模型宏观规模.csv")
    mapping = pd.read_csv(DATA4 / "损失到榜单映射.csv")
    frontier = pd.read_csv(DATA4 / "年度前沿.csv")
    forecast = pd.read_csv(DATA4 / "前沿预测.csv")
    detail = pd.read_csv(DATA4 / "BBH子任务聚合.csv")
    summary = pd.read_csv(DATA4 / "BBH子任务汇总.csv").iloc[0]
    law_path = DATA2 / "问题二输出.json"
    loss_value = None
    if law_path.exists():
        loss_value = float(read_json(law_path)["elasticity"]["L"])
    _plot_contribution(decom)
    _plot_macro_scale(macro)
    _plot_loss_mapping(mapping, loss_value)
    _plot_bbh(detail, summary)
    _plot_forecast(frontier, forecast)


def _plot_contribution(decom: pd.DataFrame):
    """带符号的贡献：预训练模型的参数量后来变小，规模项因此为负。"""
    family_label = {"chat/finetuned": "对话 / 微调", "pretrained": "预训练"}
    labels = [family_label.get(name, name) for name in decom["family"]]
    x = np.arange(len(decom))
    width = 0.36
    scale = decom["scale_contribution"].to_numpy()
    tech = decom["tech_contribution"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2, scale, width, color=BLUE, label="规模扩张", zorder=2)
    ax.bar(x + width / 2, tech, width, color=ORANGE, label="非规模技术进步", zorder=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    clean_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("对综合均分的贡献（分）")
    ax.set_title("问题四：分数变化里，规模和技术各占多少")
    for i, (s_value, t_value) in enumerate(zip(scale, tech)):
        ax.text(i - width / 2, s_value + (0.12 if s_value >= 0 else -0.28), f"{s_value:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, t_value + 0.12, f"{t_value:.2f}", ha="center", fontsize=8)
    # 对话模型的正向占比写在图角，避免和柱顶文字挤在一起。
    chat = decom[decom["family"] == "chat/finetuned"].iloc[0]
    ax.text(
        0.98,
        0.04,
        f"对话/微调的正向占比\n规模 {chat['scale_share']:.0%}，技术 {chat['tech_share']:.0%}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax.legend(loc="upper right")
    save_figure(fig, FIG4 / "规模与技术贡献.png")


def _plot_macro_scale(macro: pd.DataFrame):
    """Epoch 开放权重模型的年度中位数。2026 年样本少，数据量也有缺失。"""
    show = macro[macro["year"] >= 2015].copy()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
    series = [
        ("median_params", "参数量（个）", "(a) 参数量"),
        ("median_flops", "训练算力（FLOPs）", "(b) 训练算力"),
        ("median_tokens", "训练数据量（token）", "(c) 训练数据量"),
    ]
    for ax, (column, ylabel, title) in zip(axes, series):
        part = show.dropna(subset=[column])
        ax.plot(part["year"], part[column], color=BLUE, marker="o", linewidth=1.6)
        ax.set_yscale("log")
        clean_axes(ax)
        ax.set_xlabel("发布年份")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([2016, 2019, 2022, 2025])
    fig.suptitle("问题四：开放权重模型的宏观规模（年度中位数）", y=1.05, fontsize=13)
    save_figure(fig, FIG4 / "开源宏观规模.png")


def _plot_loss_mapping(mapping: pd.DataFrame, loss_value):
    """两条直线必须分开画。高可比样本少且分数整体偏低。"""
    loss_grid = np.linspace(1.6, 3.2, 200)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    styles = {"高可比": BLUE, "中可比": ORANGE}
    for _, row in mapping.iterrows():
        color = styles.get(row["level"], GRAY)
        pred = row["intercept"] + row["slope"] * loss_grid
        ax.plot(loss_grid, pred, color=color, linewidth=2.0, label=f"{row['level']}（n = {int(row['n'])}）")
        ax.fill_between(loss_grid, pred - row["rmse"], pred + row["rmse"], color=color, alpha=0.15)
        if loss_value is not None:
            score = row["intercept"] + row["slope"] * loss_value
            ax.scatter([loss_value], [score], color=color, s=28, zorder=3)
            ax.annotate(f"{score:.1f} 分", (loss_value, score), textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    if loss_value is not None:
        ax.axvline(loss_value, color=GRAY, linestyle="--", linewidth=0.9)
    clean_axes(ax)
    ax.set_xlabel("验证损失")
    ax.set_ylabel("榜单综合均分")
    ax.set_title("问题四：损失到榜单分数必须分层映射")
    ax.legend(loc="upper right")
    save_figure(fig, FIG4 / "损失到分数映射.png")


def _plot_bbh(detail: pd.DataFrame, summary: pd.Series):
    """左图看整体水平，右图看同一个模型内部各子任务差多大。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    ax = axes[0]
    ax.hist(detail["bbh_subtask_mean"], bins=28, color=BLUE, alpha=0.85, edgecolor="white", zorder=2)
    median_mean = float(summary["median_subtask_mean"])
    ax.axvline(median_mean, color=VERMILION, linestyle="--", linewidth=1.2, label=f"中位数 {median_mean:.2f}")
    ax.axvspan(float(summary["p10_subtask_mean"]), float(summary["p90_subtask_mean"]), color=ORANGE, alpha=0.15, label="P10–P90")
    clean_axes(ax)
    ax.set_xlabel("BBH 子任务平均准确率")
    ax.set_ylabel("模型数")
    ax.set_title("(a) 子任务平均分")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.scatter(detail["bbh_subtask_mean"], detail["bbh_subtask_std"], s=12, alpha=0.35, color=BLUE, linewidths=0, zorder=2)
    median_std = float(summary["median_subtask_std"])
    ax.axhline(median_std, color=VERMILION, linestyle="--", linewidth=1.1, label=f"标准差中位数 {median_std:.2f}")
    clean_axes(ax)
    ax.set_xlabel("BBH 子任务平均准确率")
    ax.set_ylabel("同一模型内部的标准差")
    ax.set_title("(b) 总分相近时，内部波动仍然大")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"问题四：{int(summary['n_models'])} 个模型的 BBH 逐任务表现", y=1.03, fontsize=13)
    save_figure(fig, FIG4 / "BBH均值与离散.png")


def _plot_forecast(frontier: pd.DataFrame, forecast: pd.DataFrame):
    """观测只有很短的两年。外推使用分解模型，不用把这两点连成直线。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(frontier["year"], frontier["frontier"], color=GRAY, marker="o", linewidth=1.4, label="开源年度最高均分")
    years = forecast["year"].to_numpy()
    base = forecast["frontier_baseline"].to_numpy()
    low = forecast["low_95"].to_numpy()
    high = forecast["high_95"].to_numpy()
    slow = forecast["frontier_compute_slowdown"].to_numpy()
    # 把 2025 年的观测点接到预测曲线的起点上，避免曲线悬空。
    start_year = float(frontier["year"].max())
    start_score = float(frontier.loc[frontier["year"] == start_year, "frontier"].iloc[0])
    year_line = np.concatenate([[start_year], years])
    ax.fill_between(years, low, high, color=BLUE, alpha=0.15, label="基线 95% 区间")
    ax.plot(year_line, np.concatenate([[start_score], base]), color=BLUE, marker="D", linewidth=1.8, label="基线前沿")
    ax.plot(year_line, np.concatenate([[start_score], slow]), color=ORANGE, marker="s", linewidth=1.6, label="算力增速减半")
    clean_axes(ax)
    ax.set_xlabel("年份")
    ax.set_ylabel("综合能力均分")
    ax.set_title("问题四：未来 12 个月与 24 个月的能力前沿")
    ax.set_xticks([2024, 2025, 2026, 2027])
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.98,
        0.02,
        "外推从 2025 年开源最高分出发\n不把 2024–2025 的回落连成直线",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=GRAY,
    )
    save_figure(fig, FIG4 / "前沿情景对比.png")


def main():
    """四道题的结果都已经在时，一次画出全部补充图。"""
    print("开始根据已有结果补画论文图。")
    draw_problem1()
    draw_problem2()
    draw_problem3()
    draw_problem4()
    print("全部补充图已写入 论文配图/问题一 至 论文配图/问题四。")


if __name__ == "__main__":
    main()
