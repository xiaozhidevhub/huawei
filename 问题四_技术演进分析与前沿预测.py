# -*- coding: utf-8 -*-
"""
问题四：技术演进分析与前沿预测
==============================

把开源大模型能力的上升拆成两块：
1. 规模扩张：参数量变大带来的分数变化；
2. 非规模技术进步：架构、对齐、数据工程等，用“控制参数量之后仍随时间上升”的部分表示。

计量模型（初等线性回归）：
    Score = a + b * ln(参数量) + c * 年份 + 误差
b*Δln(N) 是规模贡献，c*Δ年份 是技术进步贡献。
二者都为正时，贡献占比 = 各自增量 / 两者之和。

综合能力：Open LLM Leaderboard 六维分数的官方均分（Average）。
时间轴：榜单用 Submission Date 的年份；Epoch 宏观数据用 Publication date。
模型类型：Type 中含 pretrained 的记为预训练，含 chat 的记为对话/微调。
开源口径：Hub License 属于可研究复现的开源许可证，并且（若能在 C4 匹配到）
Open model weights 为真。权重不公开的模型不进入前沿预测。

Loss 到 Benchmark 的映射只用桥接表，并按 Loss_Comparability 分成高可比、其余（中可比）。
C8 必做一项逐任务聚合：读取每个模型最新一份可解析 JSON，汇总 BBH 子任务 acc_norm。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments" / "C_efficiency_evolution"
OUT = ROOT / "结果" / "问题四"
OUT.mkdir(parents=True, exist_ok=True)

# 允许研究与复现的常见开源许可证关键词。不包含 unknown、no-license 等无法确认的条目。
OPEN_LICENSE_KEYS = (
    "apache",
    "mit",
    "bsd",
    "llama",
    "gemma",
    "cc-by",
    "openrail",
    "afl",
    "bigscience",
    "academic",
    "mpl",
)


def is_open_license(text) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    if "unknown" in lowered or lowered.strip() in {"", "other", "no license"}:
        return False
    return any(key in lowered for key in OPEN_LICENSE_KEYS)


def model_family(type_text) -> str:
    text = str(type_text).lower()
    if "pretrained" in text:
        return "pretrained"
    if "chat" in text or "finetun" in text or "dpo" in text or "rlhf" in text:
        return "chat/finetuned"
    return "other"


def fit_decomposition(frame: pd.DataFrame) -> dict:
    """Score = a + b ln(N) + c Year。返回系数和贡献占比。"""
    work = frame.dropna(subset=["params_b", "year", "score"]).copy()
    work = work[work["params_b"] > 0]
    if len(work) < 30:
        return {"n": int(len(work)), "note": "样本不足"}
    x = np.column_stack([np.log(work["params_b"].to_numpy()), work["year"].to_numpy()])
    y = work["score"].to_numpy(dtype=np.float64)
    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    # 用样本早期 20% 与晚期 20% 的平均 ln(N)、平均年份做差分，避免单点噪音
    work = work.sort_values("year")
    k = max(20, len(work) // 5)
    early, late = work.iloc[:k], work.iloc[-k:]
    delta_ln = float(np.log(late["params_b"]).mean() - np.log(early["params_b"]).mean())
    delta_year = float(late["year"].mean() - early["year"].mean())
    b_coef = float(model.coef_[0])
    c_coef = float(model.coef_[1])
    scale_part = b_coef * delta_ln
    tech_part = c_coef * delta_year
    # 只把“确实在推高分数”的部分计入占比；下降的部分单独报告，不计入正向贡献。
    positive = []
    if scale_part > 0:
        positive.append(scale_part)
    if tech_part > 0:
        positive.append(tech_part)
    total_pos = sum(positive) if positive else np.nan
    return {
        "n": int(len(work)),
        "a": float(model.intercept_),
        "b_lnN": b_coef,
        "c_year": c_coef,
        "r2": r2,
        "delta_lnN": delta_ln,
        "delta_year": delta_year,
        "scale_contribution": scale_part,
        "tech_contribution": tech_part,
        "scale_share": float(scale_part / total_pos) if scale_part > 0 and total_pos == total_pos else 0.0,
        "tech_share": float(tech_part / total_pos) if tech_part > 0 and total_pos == total_pos else 0.0,
    }


def aggregate_bbh_subtasks(detail_root: Path) -> pd.DataFrame:
    """逐任务聚合：每个模型目录取最新一份能解析的 JSON，汇总 BBH 子任务准确率。

    子任务名形如 leaderboard_bbh_boolean_expressions。
    父任务 leaderboard_bbh 本身是汇总分，不重复计入。
    损坏或截断的 JSON 直接跳过。
    """
    rows = []
    skipped = 0
    used = 0
    for folder in sorted(p for p in detail_root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.json"))
        parsed = None
        for path in reversed(files):  # 文件名含时间，倒序即最新
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, OSError, UnicodeError):
                skipped += 1
        if not isinstance(parsed, dict):
            continue
        results = parsed.get("results") or {}
        scores = []
        for key, value in results.items():
            if not key.startswith("leaderboard_bbh_"):
                continue
            if key == "leaderboard_bbh":
                continue
            if not isinstance(value, dict):
                continue
            number = value.get("acc_norm,none", value.get("acc,none"))
            try:
                number = float(number)
            except (TypeError, ValueError):
                continue
            scores.append(number)
        if not scores:
            continue
        used += 1
        rows.append(
            {
                "model_dir": folder.name,
                "model_name": parsed.get("model_name", folder.name),
                "n_bbh_subtasks": len(scores),
                "bbh_subtask_mean": float(np.mean(scores)),
                "bbh_subtask_std": float(np.std(scores)),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["skipped_bad_json"] = skipped
    table.attrs["used_models"] = used
    return table


def bridge_mapping(bridge: pd.DataFrame) -> pd.DataFrame:
    """按可比性分层，做 均分 = a + b * 验证损失 的一元回归。"""
    rows = []
    bridge = bridge.dropna(subset=["Val_Loss", "LB_Average"]).copy()
    bridge["level"] = np.where(
        bridge["Loss_Comparability"].astype(str).str.startswith("High"),
        "高可比",
        "中可比",
    )
    for level, part in bridge.groupby("level"):
        if len(part) < 5:
            continue
        x = part[["Val_Loss"]].to_numpy(dtype=np.float64)
        y = part["LB_Average"].to_numpy(dtype=np.float64)
        model = LinearRegression().fit(x, y)
        pred = model.predict(x)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rows.append(
            {
                "level": level,
                "n": int(len(part)),
                "intercept": float(model.intercept_),
                "slope": float(model.coef_[0]),
                "rmse": rmse,
                "r2": float(1 - ss_res / ss_tot),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("问题四  技术演进分析与前沿预测")
    print("=" * 72)

    board = pd.read_csv(DATA / "leaderboard_cleaned.csv")
    board["year"] = pd.to_datetime(board["Submission Date"], errors="coerce").dt.year
    board["params_b"] = pd.to_numeric(board["#Params (B)"], errors="coerce")
    board["score"] = pd.to_numeric(board["Average ⬆️"], errors="coerce")
    board["open_license"] = board["Hub License"].map(is_open_license)
    board["family"] = board["Type"].map(model_family)
    open_board = board[board["open_license"] & board["family"].isin(["pretrained", "chat/finetuned"])].copy()
    print(f"C1 全表 {len(board)} 行；开源且类型明确 {len(open_board)} 行。")
    print("开源规则：许可证含 apache/mit/bsd/llama/gemma/cc-by 等，排除 unknown。")
    print("时间轴：榜单按 Submission Date。")

    # C3 时序（混合口径，含历史模型）用于看长期趋势，不与 C1 混在同一个回归里。
    series = pd.read_csv(DATA / "leaderboard_extended_timeseries.csv")
    series["year"] = pd.to_numeric(series["Year"], errors="coerce")
    series["score"] = pd.to_numeric(series["Average"], errors="coerce")
    yearly_hist = series.groupby("year", as_index=False)["score"].max().rename(columns={"score": "hist_frontier"})

    decom_rows = []
    for family, part in open_board.groupby("family"):
        stat = fit_decomposition(part)
        stat["family"] = family
        stat["source"] = "C1_SubmissionDate_开源许可证"
        decom_rows.append(stat)
        print(
            f"{family}: n={stat.get('n')} 规模贡献={stat.get('scale_contribution')} "
            f"技术贡献={stat.get('tech_contribution')} "
            f"占比 规模/技术={stat.get('scale_share')}/{stat.get('tech_share')} R2={stat.get('r2')}"
        )
    decom = pd.DataFrame(decom_rows)
    decom.to_csv(OUT / "规模与技术进步分解.csv", index=False, encoding="utf-8-sig")

    # C4：算力、数据量、是否开源权重。用发布日期，单独做宏观对照。
    epoch = pd.read_csv(DATA / "epoch_all_ai_models.csv")
    epoch["year"] = pd.to_datetime(epoch["Publication date"], errors="coerce").dt.year
    epoch["params"] = pd.to_numeric(epoch["Parameters"], errors="coerce")
    epoch["flops"] = pd.to_numeric(epoch["Training compute (FLOP)"], errors="coerce")
    epoch["tokens"] = pd.to_numeric(epoch["Training dataset size (total)"], errors="coerce")
    open_flag = epoch["Open model weights?"].astype(str).str.lower().isin(["true", "yes", "1"])
    epoch_open = epoch[open_flag & epoch["params"].notna() & epoch["year"].notna()].copy()
    print(f"C4 中标注开放权重的模型 {len(epoch_open)} 个。")
    macro = (
        epoch_open.groupby("year")
        .agg(n=("params", "size"), median_params=("params", "median"), median_flops=("flops", "median"), median_tokens=("tokens", "median"))
        .reset_index()
    )
    macro.to_csv(OUT / "开源模型宏观规模.csv", index=False, encoding="utf-8-sig")

    # 前沿定义：开源模型在提交日期上的年度最高均分。
    # C1 只有大约两个自然年，直接对年份做直线外推会把偶然的高低点连成“能力下降”。
    # 因此预测不用这条短直线，而用上面的分解式：
    #   未来增量 = b * 未来的 ln(参数) 增量 + c * 经过的年数
    # 参数增速取自 C4 开放权重模型的年度中位参数（跨度更长）。
    frontier = open_board.groupby("year", as_index=False)["score"].max().rename(columns={"score": "frontier"})
    frontier = frontier.dropna()
    chat_stat = next(row for row in decom_rows if row.get("family") == "chat/finetuned")
    b_ln = float(chat_stat["b_lnN"])
    c_year = float(chat_stat["c_year"])
    # 回归残差作为不确定性。分解模型已经在 chat 样本上拟合过，这里重新算一次残差标准差。
    chat = open_board[open_board["family"] == "chat/finetuned"].dropna(subset=["params_b", "year", "score"])
    chat = chat[chat["params_b"] > 0]
    x_chat = np.column_stack([np.log(chat["params_b"]), chat["year"]])
    y_chat = chat["score"].to_numpy(dtype=np.float64)
    chat_model = LinearRegression().fit(x_chat, y_chat)
    residual_std = float(np.std(y_chat - chat_model.predict(x_chat), ddof=1))

    growth = macro.dropna(subset=["median_params"]).sort_values("year")
    # 用首尾两年估计每年的对数参数增速。样本年数至少为 1，避免除零。
    year_span = max(float(growth["year"].iloc[-1] - growth["year"].iloc[0]), 1.0)
    log_growth = float(np.log(growth["median_params"].iloc[-1] / growth["median_params"].iloc[0]) / year_span)
    last_year = int(frontier["year"].max())
    last_score = float(frontier.loc[frontier["year"] == last_year, "frontier"].iloc[0])
    forecasts = []
    for months, year_add in [(12, 1.0), (24, 2.0)]:
        # 基线：参数按历史对数速度继续增长。算力放缓：这个速度减半。
        scale_base = b_ln * log_growth * year_add
        scale_slow = b_ln * (0.5 * log_growth) * year_add
        tech = c_year * year_add
        point = last_score + scale_base + tech
        slow = last_score + scale_slow + tech
        forecasts.append(
            {
                "horizon_months": months,
                "year": last_year + year_add,
                "frontier_baseline": point,
                "low_95": point - 1.96 * residual_std,
                "high_95": point + 1.96 * residual_std,
                "frontier_compute_slowdown": slow,
                "residual_std": residual_std,
                "param_log_growth_per_year": log_growth,
            }
        )
        print(
            f"未来 {months} 个月：基线前沿 {point:.2f}，"
            f"95%区间 [{point - 1.96 * residual_std:.2f}, {point + 1.96 * residual_std:.2f}]，"
            f"算力放缓情景 {slow:.2f}"
        )
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(OUT / "前沿预测.csv", index=False, encoding="utf-8-sig")
    frontier.to_csv(OUT / "年度前沿.csv", index=False, encoding="utf-8-sig")

    # Loss-Benchmark 桥接
    bridge = pd.read_csv(DATA / "loss_benchmark_bridge_expanded.csv")
    mapping = bridge_mapping(bridge)
    mapping.to_csv(OUT / "损失到榜单映射.csv", index=False, encoding="utf-8-sig")
    print("\n【Loss → 均分映射】高可比优先；中可比只作敏感性。")
    print(mapping.round(4).to_string(index=False))
    print("斜率应为负：损失越低，榜单分数越高。RMSE 是映射误差，会直接带入分数预测。")

    # 若问题二给出代表点损失，把它翻译成榜单分（只用高可比公式）
    q2_path = ROOT / "结果" / "问题二" / "问题二输出.json"
    if q2_path.exists() and len(mapping):
        law = json.loads(q2_path.read_text(encoding="utf-8"))
        loss_value = float(law["elasticity"]["L"])
        lines = [f"loss={loss_value:.4f}"]
        for _, item in mapping.iterrows():
            score_hat = float(item["intercept"] + item["slope"] * loss_value)
            print(
                f"问题二代表点损失 {loss_value:.3f}，按{item['level']}映射到均分约 {score_hat:.2f}，"
                f"映射 RMSE={item['rmse']:.2f}（n={int(item['n'])}）。"
            )
            lines.append(f"{item['level']}: score={score_hat:.4f}, rmse={item['rmse']:.4f}, n={int(item['n'])}")
        print("高可比样本主要是 Pythia，分数整体偏低，不能代表当前开源前沿。")
        print("中可比样本来源更杂，斜率更陡，但 RMSE 约 10 分，映射误差会明显影响结论。")
        (OUT / "代表点映射.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n【C8 逐任务聚合】正在读取 detailed_results，跳过损坏 JSON……")
    detail = aggregate_bbh_subtasks(DATA / "detailed_results")
    detail.to_csv(OUT / "BBH子任务聚合.csv", index=False, encoding="utf-8-sig")
    print(
        f"成功聚合 {len(detail)} 个模型，BBH 子任务平均分的中位数="
        f"{detail['bbh_subtask_mean'].median():.4f}，"
        f"子任务标准差的中位数={detail['bbh_subtask_std'].median():.4f}"
    )
    print("说明：同一模型的 BBH 子任务分数并不均匀，只看榜单上的 BBH 总分会掩盖任务短板。")
    summary = pd.DataFrame(
        [
            {
                "n_models": int(len(detail)),
                "median_subtask_mean": float(detail["bbh_subtask_mean"].median()),
                "median_subtask_std": float(detail["bbh_subtask_std"].median()),
                "p10_subtask_mean": float(detail["bbh_subtask_mean"].quantile(0.1)),
                "p90_subtask_mean": float(detail["bbh_subtask_mean"].quantile(0.9)),
            }
        ]
    )
    summary.to_csv(OUT / "BBH子任务汇总.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frontier["year"], frontier["frontier"], marker="o", label="开源年度最高均分")
    if len(yearly_hist):
        ax.plot(yearly_hist["year"], yearly_hist["hist_frontier"], marker="s", alpha=0.7, label="C3 历史最高分（混合口径）")
    for _, row in forecast_df.iterrows():
        ax.errorbar(row["year"], row["frontier_baseline"], yerr=1.96 * residual_std, fmt="D", color="#C62828", capsize=4)
    ax.set_xlabel("年份")
    ax.set_ylabel("综合能力均分")
    ax.set_title("问题四：开源模型能力前沿与 12/24 个月预测")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "能力前沿.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(detail["bbh_subtask_std"], bins=30, color="#2F6F8F")
    ax.set_xlabel("同一模型 BBH 子任务分数的标准差")
    ax.set_ylabel("模型数")
    ax.set_title("问题四：BBH 逐任务波动")
    fig.tight_layout()
    fig.savefig(OUT / "BBH子任务波动.png", dpi=150)
    plt.close(fig)

    print(f"\n结果目录：{OUT}")
    # 绘图脚本放在「论文配图」文件夹。先把该文件夹加入搜索路径，才能导入。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "论文配图"))
    from 论文配图 import draw_problem4
    draw_problem4()


if __name__ == "__main__":
    main()
