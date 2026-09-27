# -*- coding: utf-8 -*-
"""
论文附录：四个问题的核心代码
============================

本文件只收录论文四个问题的求解核心，代码摘自仓库里已经跑通的四个脚本，
公式、阈值和约束都没有另写一套。作图、打印和结果表仍留在原脚本中，本文件不改那些文件。

对照关系：
    问题一  问题一_数据质量评价、质量冲突消解与领域配比建模.py
    问题二  问题二_跨维度数据融合与广义标度律.py
    问题三  问题三_算力约束下的多维资源联合优化与结构性转移.py
    问题四  问题四_技术演进分析与前沿预测.py

阅读顺序与论文一致：先给每条文本一个质量分，再拟合损失如何随规模和质量变化，
然后在算力预算内搜索参数量、数据量和质量，最后把能力上升拆成规模与技术进步，并做前沿预测。

依赖：numpy、pandas、scipy、scikit-learn。不需要 GPU。
完整读数请仍按顺序运行上面四个脚本；本附录中的函数可单独核对公式。
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 路径与四个原脚本相同：脚本放在题目目录，数据在 real_attachments 下。
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments"


# =============================================================================
# 问题一  数据质量评价、质量冲突消解与领域配比
# =============================================================================
# 这一问做两件事：
#   1. 把 22 个质量指标变成 0 到 1 的领域质量分，并处理“内容很好但文本很脏”的冲突；
#   2. 用 17 个领域的配比去解释平均验证损失，再在“单域不超过 30%”的约束下求最优配比。

# 22 个指标与附件字段名一致。数据里有三处拼写与说明文档略有不同，这里按真实字段名书写。
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

# 负向指标：原始值越高越差。后面会改成 1-x，使全部指标都变成“越高越好”。
NEGATIVE = {
    "ad_en",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
}

# 内容价值：教育性、推理、专业程度，以及和优质语料的相似度。
CONTENT_GROUP = [
    "fineweb_edu",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
]
# 文本卫生：干净、可读、流畅、少广告、少重复、句子完整。
HYGIENE_GROUP = [
    "modernbert_cleanliness",
    "modernbert_readability",
    "fluency_en",
    "ad_en",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_ending_with_terminal_punctution_mark",
]

# 两组分数相差超过 0.30，就认为“内容”和“卫生”明显不一致。
CONFLICT_GAP = 0.30

# 这 4 个领域只有训练配比，没有自己的验证损失。
# 它们的回归系数主要反映“挤占其他领域”，不能单独当成推荐语料。
NO_LOSS_DOMAINS = {"nih_exporter", "enron_emails", "europarl", "philpapers"}


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


def build_quality_for_mixture(sample: pd.DataFrame, raw: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """配比模型使用的领域质量：arxiv、github 用扩展全量均值，其余域用抽样集均值。"""
    sample_domain = domain_summary(sample, scores.loc[sample.index])
    q_for_mix = {row["domain"]: float(row["Q"]) for _, row in sample_domain.iterrows()}
    q_for_mix["arxiv"] = float(scores.loc[raw["split"] == "extended_arxiv", "Q"].mean())
    q_for_mix["github"] = float(scores.loc[raw["split"] == "extended_github", "Q"].mean())
    return q_for_mix


def load_mixture_pair(mix_name: str, loss_name: str):
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


def reference_quality(pretty_p: pd.DataFrame, q_for_mix: dict, guide: pd.DataFrame) -> tuple[float, float]:
    """最优配比下的加权质量 Q_ref。映射不上的域用可映射域的平均质量。

    Q_ref 是问题二、问题三的参照点：质量正好等于它时，广义标度律退回经典标度律。
    """
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
    return q_ref, fallback_q


# =============================================================================
# 问题二  跨维度数据融合与广义标度律
# =============================================================================
# 经典标度律只含参数量 N 和数据量 D：
#     L(N, D) = E + A * N^{-α} + B * D^{-β}
# 加入质量后：
#     L(N, D, Q) = E + A * N^{-α} + B * D^{-β} * (Q_ref / Q)^{γ}
# 当 Q = Q_ref 时，质量因子等于 1，新公式退回经典公式。
# N、D 的单位都是“十亿”。γ 只在半合成表 B6 上估计，B7、B8 只做对照。


def classic_law(nd, E, A, alpha, B, beta):
    """经典五参数标度律。nd 的第 0 行是 N（十亿参数），第 1 行是 D（十亿 token）。"""
    n_value, d_value = nd
    return E + A * np.power(n_value, -alpha) + B * np.power(d_value, -beta)


def generalized_law(n_value, d_value, q_value, params, q_ref):
    """广义标度律。params = (E, A, alpha, B, beta, gamma)。"""
    E, A, alpha, B, beta, gamma = params
    q_safe = np.maximum(q_value, 1e-3)
    quality_factor = np.power(q_ref / q_safe, gamma)
    return E + A * np.power(n_value, -alpha) + B * np.power(d_value, -beta) * quality_factor


def fit_classic(frame: pd.DataFrame):
    """在 Pythia 真实轨迹上用非线性最小二乘估计经典标度律。"""
    n_value = frame["N_params_B"].to_numpy(dtype=np.float64)
    d_value = frame["D_tokens_B"].to_numpy(dtype=np.float64)
    y = frame["val_loss"].to_numpy(dtype=np.float64)
    # 初值参考 Hoffmann 等人的数量级，但 N、D 用“十亿”作单位，系数会不同。
    p0 = [1.5, 2.0, 0.3, 2.0, 0.3]
    lower = [0.2, 1e-6, 0.01, 1e-6, 0.01]
    upper = [4.0, 50.0, 1.5, 50.0, 1.5]
    fitted, covariance = curve_fit(
        classic_law,
        (n_value, d_value),
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )
    pred = classic_law((n_value, d_value), *fitted)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return fitted, covariance, rmse, r2


def fit_gamma(frame: pd.DataFrame, classic_params, q_ref_b6: float = 1.0):
    """只在半合成 N-D-Q 表上估计 γ。经典参数保持不变。

    B6 的 Q_score 以 1 为高质量端，因此这里的参照点取 1，
    与问题一的 Q_ref 分开。问题三使用时，用“当前配比质量”作为参照，
    只借用 γ 这个弹性指数。这是跨数据集的可检验假设，不是直接观测。
    """
    n_value = frame["N_params_B"].to_numpy(dtype=np.float64)
    d_value = frame["D_tokens_B"].to_numpy(dtype=np.float64)
    q_value = frame["Q_score"].to_numpy(dtype=np.float64)
    y = frame["val_loss"].to_numpy(dtype=np.float64)
    E, A, alpha, B, beta = classic_params

    def model(ndq, gamma):
        return generalized_law(ndq[0], ndq[1], ndq[2], (E, A, alpha, B, beta, gamma), q_ref_b6)

    fitted, _ = curve_fit(
        model,
        (n_value, d_value, q_value),
        y,
        p0=[0.5],
        bounds=([0.0], [3.0]),
        maxfev=10000,
    )
    gamma = float(fitted[0])
    pred = model((n_value, d_value, q_value), gamma)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return gamma, rmse


def regression_metrics(y_true, y_pred) -> dict:
    """外部验证用的 RMSE 和 R^2。样本内贴合不够，还要看其他表能不能对上。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "r2": r2, "n": int(len(y_true))}


def elasticities(params, n_value, d_value, q_value, q_ref):
    """在一个代表点上计算损失对 N、D、Q 的弹性。

    弹性 ε_X = (∂L/∂X) * (X/L)。
    用中心差分计算导数，避免手写复杂导数时出错，也便于初学者核对。
    """
    base = float(generalized_law(n_value, d_value, q_value, params, q_ref))
    step = 1e-3

    def one_side(kind):
        if kind == "N":
            plus = generalized_law(n_value * (1 + step), d_value, q_value, params, q_ref)
            minus = generalized_law(n_value * (1 - step), d_value, q_value, params, q_ref)
            derivative = (plus - minus) / (2 * step * n_value)
            return derivative * n_value / base
        if kind == "D":
            plus = generalized_law(n_value, d_value * (1 + step), q_value, params, q_ref)
            minus = generalized_law(n_value, d_value * (1 - step), q_value, params, q_ref)
            derivative = (plus - minus) / (2 * step * d_value)
            return derivative * d_value / base
        plus = generalized_law(n_value, d_value, q_value * (1 + step), params, q_ref)
        minus = generalized_law(n_value, d_value, q_value * (1 - step), params, q_ref)
        derivative = (plus - minus) / (2 * step * q_value)
        return derivative * q_value / base

    return {"L": base, "eps_N": one_side("N"), "eps_D": one_side("D"), "eps_Q": one_side("Q")}


def quality_equivalent_params(params, n_value, d_value, q_value, q_ref, delta_q=0.1):
    """求质量提高 delta_q 后，损失不变所对应的新参数量。

    做法：先算出提高质量后的损失，再对参数量做一维搜索，
    使 L(N_new, D, Q) = L(N, D, Q+delta_q)。
    """
    target = float(generalized_law(n_value, d_value, q_value + delta_q, params, q_ref))
    # 质量提高会降低损失。要在原来的质量下达到同样低的损失，需要更大的参数量。
    # 所以搜索区间放在当前参数量的右侧。这就是“质量 +0.1 相当于参数增加多少”。
    grid = np.geomspace(n_value, n_value * 80.0, 8000)
    losses = generalized_law(grid, d_value, q_value, params, q_ref)
    idx = int(np.argmin(np.abs(losses - target)))
    n_new = float(grid[idx])
    return {
        "Q": q_value,
        "Q_new": q_value + delta_q,
        "L_target": target,
        "N_old": n_value,
        "N_new": n_new,
        "N_ratio": n_new / n_value,
        "equivalent_param_increase_B": n_new - n_value,
    }


# =============================================================================
# 问题三  算力约束下的多维资源联合优化与结构性转移
# =============================================================================
# 决策变量是参数量 N、数据量 D、数据质量 Q。领域配比固定为问题一的最优配比。
# 总算力：
#     C = C_train + C_quality + C_attn
#     C_train   = 6 N D
#     C_quality = D * max(g(Q) - g(Q0), 0)
#     C_attn    = η N D L_ctx ，η = 2e-4
# 注意力开销等于训练开销时，L_ctx_crit = 6 / η = 30000。
# 三档预算是 10^19、10^22、10^24 FLOPs。网格搜索步骤透明，适合入门实现。

ETA = 2e-4
LCTX_CRIT = 6.0 / ETA  # 30000
BUDGETS = [1e19, 1e22, 1e24]


def g_exp(q):
    """指数型质量成本，附录 B：γ=1e7，λ=6。"""
    return 1e7 * np.exp(6.0 * q)


def g_power(q):
    """幂函数型：γ=5e9，λ=4。"""
    return 5e9 * np.power(q, 4.0)


def g_log(q):
    """对数渐进型：γ=2e9，λ=10。"""
    return 2e9 * np.log1p(10.0 * q)


COST_FUNCS = {"指数型": g_exp, "幂函数型": g_power, "对数渐进型": g_log}


def predict_loss(n_billion, d_billion, q_value, params, q_ref):
    """问题二的广义标度律。N、D 仍用十亿作单位，与拟合时一致。"""
    E, A, alpha, B, beta, gamma = params
    factor = (q_ref / max(q_value, 1e-3)) ** gamma
    return E + A * (n_billion ** (-alpha)) + B * (d_billion ** (-beta)) * factor


def search_one(budget, g_func, lctx, q0, q_ref, params):
    """在一张网格上找损失最小且总算力不超过预算的点。"""
    # 参数量从 0.07B 到 400B，数据量从 1B 到 2e4 B token，覆盖三档预算。
    n_grid = np.geomspace(0.07, 400.0, 28)
    d_grid = np.geomspace(1.0, 20000.0, 28)
    q_grid = np.linspace(q0, min(0.98, max(q0 + 0.02, 0.98)), 13)

    best = None
    for n_b in n_grid:
        n_count = n_b * 1e9
        for d_b in d_grid:
            d_count = d_b * 1e9
            c_train = 6.0 * n_count * d_count
            c_attn = ETA * n_count * d_count * lctx
            # 训练+注意力已经超预算时，提高质量只会更贵，可以直接跳过
            if c_train + c_attn > budget:
                continue
            for q_value in q_grid:
                c_quality = d_count * max(float(g_func(q_value) - g_func(q0)), 0.0)
                total = c_train + c_quality + c_attn
                if total > budget:
                    continue
                loss = predict_loss(n_b, d_b, q_value, params, q_ref)
                if best is None or loss < best["loss"]:
                    best = {
                        "loss": float(loss),
                        "N_B": float(n_b),
                        "D_B": float(d_b),
                        "Q": float(q_value),
                        "C_train": float(c_train),
                        "C_quality": float(c_quality),
                        "C_attn": float(c_attn),
                        "C_total": float(total),
                    }
    if best is None:
        return None
    total = best["C_total"]
    best["share_train"] = best["C_train"] / total
    best["share_quality"] = best["C_quality"] / total
    best["share_attn"] = best["C_attn"] / total
    # 策略标签：质量花费占比决定“规模优先”还是“质量优先”
    if best["share_quality"] < 0.05:
        best["strategy"] = "规模优先"
    elif best["share_quality"] >= 0.15:
        best["strategy"] = "质量优先"
    else:
        best["strategy"] = "规模与质量均衡"
    return best


def structural_shift(low_row, high_row) -> str:
    """结构性转移的数学定义。

    记两档预算的最优策略标签为 s1、s2，质量花费占比为 r1、r2。
    若 s1 ≠ s2，或 |r2-r1| > 0.15，则称从低预算到高预算发生了结构性转移。
    这是“支出结构发生质变”，不是损失数值的普通下降。
    """
    label_change = low_row["strategy"] != high_row["strategy"]
    share_change = abs(high_row["share_quality"] - low_row["share_quality"]) > 0.15
    if label_change or share_change:
        return "发生结构性转移"
    return "未发生结构性转移"


def context_lengths_from_architecture(arch: pd.DataFrame) -> list[int]:
    """上下文长度只使用架构表里真实出现过的窗口，不自行假设一个连续区间。"""
    return sorted(int(v) for v in arch["max_position_embeddings"].dropna().unique())


def choose_base_context(lctx_values: list[int]) -> int:
    """主结果优先用 8192；表中没有它时，用长度列表的中位值。"""
    if 8192 in lctx_values:
        return 8192
    return lctx_values[len(lctx_values) // 2]


# =============================================================================
# 问题四  技术演进分析与前沿预测
# =============================================================================
# 计量模型：
#     Score = a + b * ln(参数量) + c * 年份 + 误差
# b*Δln(N) 是规模贡献，c*Δ年份 是非规模技术进步贡献。
# 综合能力用榜单六维均分。前沿预测不用只有两年的短直线，而用上面的分解式外推。

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
    """许可证文本里出现可复现关键词，且不是 unknown，才记为开源。"""
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    if "unknown" in lowered or lowered.strip() in {"", "other", "no license"}:
        return False
    return any(key in lowered for key in OPEN_LICENSE_KEYS)


def model_family(type_text) -> str:
    """Type 中含 pretrained 记为预训练，含 chat 或常见对齐字样记为对话/微调。"""
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


def param_log_growth(macro: pd.DataFrame) -> float:
    """用开放权重模型首尾两年的中位参数，估计每年的对数参数增速。"""
    growth = macro.dropna(subset=["median_params"]).sort_values("year")
    # 样本年数至少为 1，避免除零。
    year_span = max(float(growth["year"].iloc[-1] - growth["year"].iloc[0]), 1.0)
    return float(np.log(growth["median_params"].iloc[-1] / growth["median_params"].iloc[0]) / year_span)


def forecast_frontier(last_year, last_score, b_ln, c_year, log_growth, residual_std):
    """未来增量 = b * 未来的 ln(参数) 增量 + c * 经过的年数。

    基线情景按历史对数速度增长；算力放缓情景把这个速度减半。
    95% 区间用回归残差标准差的 1.96 倍。这里只预测 12 个月和 24 个月。
    """
    forecasts = []
    for months, year_add in [(12, 1.0), (24, 2.0)]:
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
    return forecasts


def map_loss_to_score(mapping: pd.DataFrame, loss_value: float) -> list[dict]:
    """把一个验证损失按高可比、中可比两条直线分别翻译成榜单均分。"""
    rows = []
    for _, item in mapping.iterrows():
        score_hat = float(item["intercept"] + item["slope"] * loss_value)
        rows.append(
            {
                "level": item["level"],
                "score": score_hat,
                "rmse": float(item["rmse"]),
                "n": int(item["n"]),
            }
        )
    return rows


def _self_check() -> None:
    """用几个手算例子核对核心公式。不读取附件，也不写出论文结果。"""
    # 只有一个数的列表，就取这个数本身。
    assert compress_to_scalar([3.5]) == 3.5
    # 两档 logits 里第二档远大于第一档时，期望档位应接近 1。
    level = compress_to_scalar([0.0, 20.0])
    assert abs(level - 1.0) < 1e-6

    # 注意力开销与训练开销相等的临界长度由 6/η 直接推出。
    assert abs(LCTX_CRIT - 30000.0) < 1e-6

    # Q 等于参照质量时，广义标度律必须退回经典标度律。
    classic_params = (1.69, 0.354, 0.34, 1.24, 0.28)
    params = classic_params + (0.29,)
    q_ref = 0.333
    classic = float(classic_law((6.9, 300.0), *classic_params))
    general = float(generalized_law(6.9, 300.0, q_ref, params, q_ref))
    assert abs(classic - general) < 1e-8

    # 策略标签从“质量优先”变成“规模优先”，应判定为结构性转移。
    low = {"strategy": "质量优先", "share_quality": 0.22}
    high = {"strategy": "规模优先", "share_quality": 0.01}
    assert structural_shift(low, high) == "发生结构性转移"

    # 冲突消解：内容 0.9、卫生 0.2，差距 0.7，λ 被截到 1，质量分应落到短板 0.2。
    row = {name: 0.5 for name in INDICATORS}
    for name in CONTENT_GROUP:
        row[name] = 0.9
    for name in HYGIENE_GROUP:
        row[name] = 0.2
    unified = pd.DataFrame([row])
    weights = pd.Series(1.0 / len(INDICATORS), index=INDICATORS)
    scored = score_quality(unified, weights)
    assert abs(float(scored.loc[0, "Q"]) - 0.2) < 1e-8

    print("附录核心公式自检通过。")
    print("完整求解仍请按顺序运行四个问题脚本；本文件不改动它们，也不重写结果表。")


if __name__ == "__main__":
    _self_check()
