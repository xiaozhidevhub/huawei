# -*- coding: utf-8 -*-
"""
问题一：数据质量评价、质量冲突消解与领域配比建模
================================================

对应赛题《算力约束下提升大语言模型能力的资源配置建模》。
本文件按华为杯论文常见结构组织：符号、假设、模型、求解、检验、结论。
运行后会在「结果/问题一」写出表格、图片，并给出供问题二、三调用的质量分与最优配比。

依赖（本机已具备）：numpy、pandas、scipy、scikit-learn、matplotlib。
无需 GPU。质量信号用流式方式读入压缩 jsonl，不把原文 content 存进内存。
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 让图中的中文能正常显示（Windows 常见字体）
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 路径：脚本放在 F 题目录，数据在 real_attachments 下
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments"
OUT = ROOT / "结果" / "问题一"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 一、符号与指标方向
# ---------------------------------------------------------------------------
# 22 个质量指标与附件真实字段名一致（数据里有两处拼写与《数据说明》略有不同：
# qurater、rps_doc_unigram_entropy、rps_lines_ending_with_terminal_punctution_mark）。
# 方向：正向 = 原始值越高越好；负向 = 原始值越高越差，稍后做成 1-x。

INDICATORS = [
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
]

# 负向指标：广告、非字母词、重复 n-gram、大写比例、数字比例。处理后改为越高越好。
NEGATIVE = {
    "ad_en",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
}

# 冲突分析用的两组指标。
# 内容价值组：教育性、推理、专业、与优质语料的相似度。
# 文本卫生组：干净、可读、流畅、低广告、低重复、句子完整。
CONTENT_GROUP = [
    "fineweb_edu",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
]
HYGIENE_GROUP = [
    "modernbert_cleanliness",
    "modernbert_readability",
    "fluency_en",
    "ad_en",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_ending_with_terminal_punctution_mark",
]

# 冲突阈值：两组“越高越好”得分相差超过 0.30，视为显著不一致。
CONFLICT_GAP = 0.30


def compress_to_scalar(value):
    """把一个质量字段压成一个数。

    标量直接使用。列表是分类打分的一组 logits：
    用 softmax 把它们变成概率，再求“期望类别序号”。
    序号越大，表示模型越倾向更高档的评价。
    只有 1 个数的列表（如 fineweb_edu）就取这个数本身。
    """
    if isinstance(value, list):
        if len(value) == 0:
            return np.nan
        arr = np.asarray(value, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return float(arr[0])
        # 减去最大值，避免 exp 溢出。这是 softmax 的标准稳定写法。
        shifted = arr - np.max(arr)
        weights = np.exp(np.clip(shifted, -50, 50))
        weights = weights / weights.sum()
        levels = np.arange(arr.size, dtype=np.float64)
        return float(np.dot(weights, levels))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    return number


def stream_quality(path: Path, domain_name: str | None, split_name: str):
    """逐行读取 xz 压缩的 jsonl，只保留 22 个指标和领域名。

    domain_name 不为空时（扩展集没有 _source_domain），用文件所属领域。
    原文 content 读完即丢，避免占用大量内存。
    """
    rows = []
    domains = []
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            domain = domain_name if domain_name is not None else str(obj.get("_source_domain", "unknown"))
            values = [compress_to_scalar(obj.get(name)) for name in INDICATORS]
            rows.append(values)
            domains.append(domain)
    frame = pd.DataFrame(rows, columns=INDICATORS)
    frame["domain"] = domains
    frame["split"] = split_name
    return frame


def direction_unify(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """缩尾、最小-最大归一化，并把负向指标改成越高越好。

    缩尾：把两端 1% 的极端值拉回到分位点，减轻广告比例等长尾的影响。
    返回统一后的表，以及每个指标的熵权。
    """
    unified = pd.DataFrame(index=frame.index)
    for name in INDICATORS:
        series = frame[name].astype(float)
        # 缺失用中位数填补，保证每一行都能打分
        filled = series.fillna(series.median())
        low, high = np.nanpercentile(filled, [1, 99])
        if high <= low:
            high = low + 1e-6
        clipped = filled.clip(low, high)
        scaled = (clipped - low) / (high - low)
        if name in NEGATIVE:
            scaled = 1.0 - scaled
        unified[name] = scaled.clip(0.0, 1.0)

    # 熵权法：哪个指标在样本之间差异越大，权重越高。
    # 公式：p_ij = x_ij / sum_i x_ij，e_j = -sum p ln p / ln(n)，w_j 正比于 1-e_j。
    matrix = unified.to_numpy(dtype=np.float64) + 1e-12
    proportion = matrix / matrix.sum(axis=0, keepdims=True)
    n_rows = matrix.shape[0]
    entropy = -(proportion * np.log(proportion)).sum(axis=0) / np.log(n_rows)
    diversity = 1.0 - entropy
    diversity = np.clip(diversity, 0.0, None)
    if diversity.sum() <= 0:
        weights = np.full(len(INDICATORS), 1.0 / len(INDICATORS))
    else:
        weights = diversity / diversity.sum()
    weight_table = pd.Series(weights, index=INDICATORS, name="entropy_weight")
    return unified, weight_table


def score_quality(unified: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """计算样本质量分，并按冲突规则消解。

    综合分 Q_entropy 是熵权加权平均，范围约在 0 到 1，越高越好。
    冲突定义：内容价值组均值与文本卫生组均值之差的绝对值大于 0.30。
    直观例子：教育价值很高，但广告/重复也很高（卫生分被压低）。
    消解规则：冲突越强，越向“短板”（两组中的较小值）靠拢。
        Q = (1-λ) * Q_entropy + λ * min(内容分, 卫生分)
        λ = min(1, (差距-0.30)/0.40)，没有冲突时 λ=0。
    这样一份“看起来很专业但充满广告”的文本不会拿到虚高的质量分。
    """
    content_score = unified[CONTENT_GROUP].mean(axis=1)
    hygiene_score = unified[HYGIENE_GROUP].mean(axis=1)
    base_score = unified.to_numpy(dtype=np.float64) @ weights.to_numpy(dtype=np.float64)
    gap = (content_score - hygiene_score).abs()
    conflict = gap > CONFLICT_GAP
    lam = np.where(conflict, np.minimum(1.0, (gap - CONFLICT_GAP) / 0.40), 0.0)
    short_board = np.minimum(content_score, hygiene_score)
    resolved = (1.0 - lam) * base_score + lam * short_board
    return pd.DataFrame(
        {
            "Q_entropy": base_score,
            "Q_content": content_score,
            "Q_hygiene": hygiene_score,
            "conflict": conflict.astype(int),
            "lambda_conflict": lam,
            "Q": resolved,
        }
    )


def domain_summary(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """样本分按领域取平均，得到领域级质量分。同时统计冲突比例。"""
    temp = pd.concat([frame[["domain", "split"]], scores], axis=1)
    grouped = (
        temp.groupby("domain", as_index=False)
        .agg(
            n=("Q", "size"),
            Q=("Q", "mean"),
            Q_entropy=("Q_entropy", "mean"),
            conflict_rate=("conflict", "mean"),
        )
        .sort_values("Q", ascending=False)
    )
    return grouped


def compare_sample_and_extended(raw: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """抽样集与 arxiv/github 扩展全量的领域级 Q 对照，并用 KS 检验看分布是否接近。"""
    temp = pd.concat([raw[["domain", "split"]], scores[["Q"]]], axis=1)
    rows = []
    for domain, extended_split in [("arxiv", "extended_arxiv"), ("github", "extended_github")]:
        sample_q = temp.loc[(temp["domain"] == domain) & (temp["split"] == "sample"), "Q"]
        extended_q = temp.loc[temp["split"] == extended_split, "Q"]
        if len(sample_q) == 0 or len(extended_q) == 0:
            continue
        stat, pvalue = ks_2samp(sample_q, extended_q)
        rows.append(
            {
                "domain": domain,
                "sample_n": int(len(sample_q)),
                "sample_Q": float(sample_q.mean()),
                "extended_n": int(len(extended_q)),
                "extended_Q": float(extended_q.mean()),
                "Q_diff": float(extended_q.mean() - sample_q.mean()),
                "ks_stat": float(stat),
                "ks_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def load_mixture_pair(mix_name: str, loss_name: str) -> pd.DataFrame:
    """读入一组配比表和损失表，按 index 对齐。目标损失取 13 个验证域的平均。"""
    mix = pd.read_csv(DATA / "A_data_value" / "regmix_tables" / mix_name)
    loss = pd.read_csv(DATA / "A_data_value" / "regmix_tables" / loss_name)
    merged = mix.merge(loss, on="index", how="inner")
    ratio_cols = [c for c in mix.columns if c.startswith("train_the_pile_")]
    loss_cols = [c for c in loss.columns if c != "index"]
    merged["loss_mean"] = merged[loss_cols].mean(axis=1)
    return merged, ratio_cols, loss_cols


def fit_ridge(train: pd.DataFrame, ratio_cols: list[str]):
    """用岭回归建立 L ≈ a + b·p。

    17 个配比加起来约等于 1，普通最小二乘会不稳定。
    岭回归给系数加一个小的平方惩罚，适合这种“和为 1”的配比向量。
    alpha 用 5 折交叉验证在一组候选值里挑选。
    """
    x = train[ratio_cols].to_numpy(dtype=np.float64)
    y = train["loss_mean"].to_numpy(dtype=np.float64)
    candidates = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
    best_alpha, best_score = candidates[0], -1e9
    folder = KFold(n_splits=5, shuffle=True, random_state=23)
    for alpha in candidates:
        scores = []
        for train_idx, valid_idx in folder.split(x):
            model = Ridge(alpha=alpha)
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[valid_idx])
            scores.append(r2_score(y[valid_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_alpha, best_score = alpha, mean_score
    final = Ridge(alpha=best_alpha)
    final.fit(x, y)
    return final, best_alpha, best_score


def evaluate_split(model, frame, ratio_cols, split_name: str) -> dict:
    """在某一张配比-损失表上计算 RMSE、R^2 和 Spearman 相关。"""
    pred = model.predict(frame[ratio_cols].to_numpy(dtype=np.float64))
    y = frame["loss_mean"].to_numpy(dtype=np.float64)
    spearman = pd.Series(pred).corr(pd.Series(y), method="spearman")
    return {
        "split": split_name,
        "n": int(len(frame)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearman),
    }


# 这 4 个领域只有训练配比，没有自己的验证损失。
# 它们的回归系数主要反映“挤占其他领域”的间接作用，不能单独当成推荐语料。
NO_LOSS_DOMAINS = {"nih_exporter", "enron_emails", "europarl", "philpapers"}


def optimal_mixture(model, ratio_cols: list[str], train: pd.DataFrame) -> pd.Series:
    """在单纯形上最小化预测损失。

    线性函数在单纯形上的最小值会出现在顶点（100% 投给一个领域）。
    因此规定：单个有损失观测的领域不超过 30%。
    没有损失列的 4 个领域固定为训练集中位数比例，只解释间接影响，不参与“冲到上限”。
    求解器用 SLSQP，适合带等式和不等式约束的光滑问题。
    """
    coef = model.coef_
    intercept = float(model.intercept_)
    n_dim = len(ratio_cols)
    fixed = np.zeros(n_dim)
    free = []
    for i, col in enumerate(ratio_cols):
        domain = col.replace("train_the_pile_", "")
        if domain in NO_LOSS_DOMAINS:
            fixed[i] = float(train[col].median())
        else:
            free.append(i)
    fixed_mass = float(fixed.sum())
    rest = max(1.0 - fixed_mass, 0.05)

    def objective(free_p):
        p = fixed.copy()
        p[free] = free_p
        return float(intercept + np.dot(coef, p))

    start = np.full(len(free), rest / len(free))
    bounds = [(0.0, 0.30)] * len(free)
    constraints = {"type": "eq", "fun": lambda free_p: np.sum(free_p) - rest}
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    free_p = result.x if result.success else start
    free_p = np.clip(free_p, 0.0, None)
    free_p = free_p / free_p.sum() * rest
    p = fixed.copy()
    p[free] = free_p
    p = p / p.sum()
    return pd.Series(p, index=ratio_cols)


def main():
    print("=" * 72)
    print("问题一  数据质量评价、质量冲突消解与领域配比建模")
    print("=" * 72)
    print("【假设】")
    print("1. 列表型指标是分类 logits，用 softmax 期望档位压成标量。")
    print("2. 负向指标做 1-MinMax，使全部指标都是越高越好。")
    print("3. 领域质量分等于该领域样本质量分的算术平均。")
    print("4. 4 个没有验证损失的训练域仍保留在配比模型中，其影响体现在系数上。")

    sample_path = DATA / "A_data_value" / "slimpajama_quality_signal_sample.jsonl.xz"
    arxiv_path = next((DATA / "A_data_value" / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz"))
    github_path = next((DATA / "A_data_value" / "slimpajama_quality_extended").glob("github_*.jsonl.xz"))

    print("\n【读取质量信号】抽样集 + arxiv 扩展 + github 扩展，全文记录，不保留原文。")
    sample = stream_quality(sample_path, None, "sample")
    arxiv = stream_quality(arxiv_path, "arxiv", "extended_arxiv")
    github = stream_quality(github_path, "github", "extended_github")
    raw = pd.concat([sample, arxiv, github], ignore_index=True)
    print(f"记录数：抽样 {len(sample)}，arxiv 扩展 {len(arxiv)}，github 扩展 {len(github)}，合计 {len(raw)}")

    # 归一化用全量记录一起做，保证抽样集和扩展集的分数可以比较
    unified, weights = direction_unify(raw)
    scores = score_quality(unified, weights)
    weights.sort_values(ascending=False).to_csv(OUT / "熵权.csv", encoding="utf-8-sig")
    print("\n【熵权前 8】")
    print(weights.sort_values(ascending=False).head(8).round(4).to_string())

    by_domain = domain_summary(raw, scores)
    by_domain.to_csv(OUT / "领域质量分.csv", index=False, encoding="utf-8-sig")
    print("\n【领域级质量分 Q】")
    print(by_domain.round(4).to_string(index=False))

    contrast = compare_sample_and_extended(raw, scores)
    contrast.to_csv(OUT / "抽样与扩展对照.csv", index=False, encoding="utf-8-sig")
    print("\n【抽样集 vs 扩展全量】")
    print(contrast.round(4).to_string(index=False))

    # 冲突是否在扩展集上仍然成立：比较冲突比例
    tagged = pd.concat([raw[["split"]], scores[["conflict"]]], axis=1)
    conflict_rate = tagged.groupby("split")["conflict"].mean()
    print("\n【冲突比例】")
    print(conflict_rate.round(4).to_string())
    conflict_rate.to_csv(OUT / "冲突比例.csv", encoding="utf-8-sig")

    # 只把抽样集的领域分，和“抽样+该域扩展”合并后的领域分都保存。
    # 配比建模使用：arxiv/github 用扩展全量均值，其余域用抽样集均值。
    sample_domain = domain_summary(sample, scores.loc[sample.index])
    q_map = {}
    for _, row in by_domain.iterrows():
        q_map[row["domain"]] = float(row["Q"])
    # 扩展集把 arxiv/github 的样本和扩展混在 by_domain 里了，上面 groupby 会把同名域合在一起。
    # 这里改成显式：arxiv、github 用扩展集，其他域用抽样集。
    q_for_mix = {row["domain"]: float(row["Q"]) for _, row in sample_domain.iterrows()}
    q_for_mix["arxiv"] = float(scores.loc[raw["split"] == "extended_arxiv", "Q"].mean())
    q_for_mix["github"] = float(scores.loc[raw["split"] == "extended_github", "Q"].mean())

    # -------------------------------------------------------------------
    # 配比模型
    # -------------------------------------------------------------------
    print("\n【领域配比模型】岭回归：平均验证损失 ~ 17 维配比")
    train, ratio_cols, _ = load_mixture_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    model, alpha, cv_r2 = fit_ridge(train, ratio_cols)
    print(f"交叉验证选中的岭参数 alpha={alpha}，训练折内平均 R^2={cv_r2:.4f}")

    coef_table = pd.DataFrame({"domain": [c.replace("train_the_pile_", "") for c in ratio_cols], "coef": model.coef_})
    coef_table["effect"] = np.where(coef_table["coef"] < 0, "增大配比有利于降低损失", "增大配比会抬高损失")
    coef_table = coef_table.sort_values("coef")
    coef_table.to_csv(OUT / "配比系数.csv", index=False, encoding="utf-8-sig")
    print(coef_table.round(4).to_string(index=False))
    print("系数越小，越应该多分配该领域的训练比例。")

    checks = [("train_1m", "train_mixture_1m.csv", "train_pile_loss_1m.csv")]
    checks += [
        ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv"),
        ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
        ("test_1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv"),
        ("est_10b", "est_mixture_10b.csv", "est_pile_loss_10b.csv"),
        ("est_70b", "est_mixture_70b.csv", "est_pile_loss_70b.csv"),
    ]
    metric_rows = []
    for name, mix_name, loss_name in checks:
        frame, _, _ = load_mixture_pair(mix_name, loss_name)
        row = evaluate_split(model, frame, ratio_cols, name)
        if name.startswith("est_"):
            row["note"] = "外推损失不是直接观测。绝对 RMSE 不可比，主要看 Spearman 排序"
        elif name == "train_1m" or name == "test_1m":
            row["note"] = "与训练同一参数规模，R^2 可直接比较"
        else:
            row["note"] = "损失水平随参数规模下移，绝对 R^2 会很差；排序能力看 Spearman"
        metric_rows.append(row)
        print(f"{name}: RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}, Spearman={row['spearman']:.4f}")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "配比模型检验.csv", index=False, encoding="utf-8-sig")

    best_p = optimal_mixture(model, ratio_cols, train)
    pretty_p = pd.DataFrame(
        {
            "domain": [c.replace("train_the_pile_", "") for c in best_p.index],
            "p": best_p.to_numpy(),
        }
    ).sort_values("p", ascending=False)
    pretty_p.to_csv(OUT / "最优配比.csv", index=False, encoding="utf-8-sig")
    print("\n【最优配比】无损失列的 4 个域固定为训练中位数，其余单域不超过 30%")
    print(pretty_p.round(4).to_string(index=False))
    print(f"预测平均损失 = {float(model.intercept_ + np.dot(model.coef_, best_p.to_numpy())):.4f}")

    # 质量分不能再作为线性模型的新自变量：Q_mix = q·p 已经是 p 的线性组合。
    # 这里改为检验：映射得上的领域，其质量分与损失系数是否同向。
    guide = pd.read_csv(DATA / "A_data_value" / "domain_mapping_guide.csv")
    guide["quality_domain"] = guide["quality_domain"].replace("(none)", np.nan)
    coef_table = coef_table.merge(guide[["mixture_domain", "quality_domain", "mapping_type"]], left_on="domain", right_on="mixture_domain", how="left")
    coef_table["Q_domain"] = coef_table["quality_domain"].map(q_for_mix)
    linked = coef_table.dropna(subset=["Q_domain"])
    if len(linked) >= 3:
        corr = float(linked["Q_domain"].corr(linked["coef"], method="spearman"))
    else:
        corr = float("nan")
    print(f"\n【质量与配比的关系】可映射领域上，领域质量分与损失系数的 Spearman 相关 = {corr:.4f}")
    print("相关为负表示：质量更高的领域，增大其配比更有利于降低损失。")
    print("线性模型里不再把 q·p 当作新特征，因为它与 p 完全共线，无法单独识别。")
    coef_table.to_csv(OUT / "质量与系数对照.csv", index=False, encoding="utf-8-sig")

    # 供后续问题使用的参照质量：最优配比下的加权质量。映射不上的域用可映射域的平均质量。
    known_q = [q_for_mix[d] for d in q_for_mix if d in set(guide["quality_domain"].dropna())]
    fallback_q = float(np.mean(known_q)) if known_q else 0.5
    q_vector = []
    for domain in pretty_p["domain"]:
        # pretty_p 的 domain 是配方域。能直接对上质量域的用该域分数，否则用后备分数。
        mapped = guide.loc[guide["mixture_domain"] == domain, "quality_domain"]
        if len(mapped) and pd.notna(mapped.iloc[0]) and mapped.iloc[0] in q_for_mix:
            q_vector.append(q_for_mix[mapped.iloc[0]])
        else:
            q_vector.append(fallback_q)
    q_ref = float(np.dot(pretty_p["p"].to_numpy(), np.asarray(q_vector)))

    payload = {
        "Q_ref": q_ref,
        "Q_fallback": fallback_q,
        "ridge_alpha": alpha,
        "ridge_intercept": float(model.intercept_),
        "conflict_gap": CONFLICT_GAP,
        "n_sample": int(len(sample)),
        "n_arxiv": int(len(arxiv)),
        "n_github": int(len(github)),
        "conflict_rate": {k: float(v) for k, v in conflict_rate.items()},
        "spearman_Q_vs_coef": corr,
        "domain_Q": {row["domain"]: float(row["Q"]) for _, row in by_domain.iterrows()},
        "optimal_p": {row["domain"]: float(row["p"]) for _, row in pretty_p.iterrows()},
    }
    (OUT / "问题一输出.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图：领域质量分
    plot_df = by_domain.sort_values("Q")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["domain"], plot_df["Q"], color="#2F6F8F")
    ax.set_xlabel("领域质量分 Q（越高越好）")
    ax.set_title("问题一：各质量领域的综合质量分")
    fig.tight_layout()
    fig.savefig(OUT / "领域质量分.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    show = coef_table.sort_values("coef")
    colors = ["#2E7D32" if v < 0 else "#C62828" for v in show["coef"]]
    ax.barh(show["domain"], show["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("岭回归系数（负值表示增加该域配比可降低损失）")
    ax.set_title("问题一：17 个领域对平均验证损失的边际影响")
    fig.tight_layout()
    fig.savefig(OUT / "配比系数.png", dpi=150)
    plt.close(fig)

    print("\n【结论摘要】")
    print(f"1. 全量质量信号已打分，参照配比下的综合质量 Q_ref = {q_ref:.4f}。")
    print("2. 冲突按“内容价值与文本卫生显著背离”定义，并用短板加权消解。")
    print("3. 配比-损失模型在 1M/60M/1B 检验集上给出 RMSE 与 R^2；10B/70B 只作排序稳健性讨论。")
    print(f"结果目录：{OUT}")
    # 绘图脚本放在「论文配图」文件夹。先把该文件夹加入搜索路径，才能导入。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "论文配图"))
    from 论文配图 import draw_problem1
    draw_problem1()


if __name__ == "__main__":
    main()
