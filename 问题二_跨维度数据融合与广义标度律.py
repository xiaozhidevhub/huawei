# -*- coding: utf-8 -*-
"""
问题二：跨维度数据融合与广义标度律
================================

经典标度律只含参数量 N 和数据量 D：
    L(N, D) = E + A * N^{-α} + B * D^{-β}
本问在此基础上加入数据质量 Q。设计原则：当 Q 等于参照质量 Q_ref 时，
新公式退回经典公式。参照质量来自问题一的最优配比加权质量分。

    L(N, D, Q) = E + A * N^{-α} + B * D^{-β} * (Q_ref / Q)^{γ}

γ>0 时，质量越高，数据项越小，损失越低。
附件 B6–B8 是半合成数据，只用于估计 γ，论文中不把它当成真实实验。

弹性：损失对某个因素变化 1% 时，自己大约变化百分之几。
替代：质量提高 0.1，相当于参数量变为原来的多少倍，才能保持损失不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments" / "B_scaling_laws"
OUT = ROOT / "结果" / "问题二"
OUT.mkdir(parents=True, exist_ok=True)
Q1_JSON = ROOT / "结果" / "问题一" / "问题一输出.json"


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


def main():
    if not Q1_JSON.exists():
        raise SystemExit("请先运行问题一，生成结果/问题一/问题一输出.json")
    problem1 = json.loads(Q1_JSON.read_text(encoding="utf-8"))
    q_ref = float(problem1["Q_ref"])
    optimal_p = problem1["optimal_p"]

    print("=" * 72)
    print("问题二  跨维度数据融合与广义标度律")
    print("=" * 72)
    print("【可检验假设】")
    print("H1：Pythia 轨迹服从经典标度律，且该参数可迁移到其他模型族（用 B2/B4/B5 检验）。")
    print("H2：质量只缩放数据项；Q=Q_ref 时广义律退化为经典律。")
    print("H3：领域之间的替代由问题一的配比系数符号描述：系数为负的领域相互替代，")
    print("    一正一负则表现为互补（必须同时保留，否则损失上升）。")
    print(f"问题一给出的参照质量 Q_ref = {q_ref:.4f}")

    pythia = pd.read_csv(DATA / "pythia_training_log_existing.csv")
    pythia = pythia.replace([np.inf, -np.inf], np.nan).dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    pythia = pythia[(pythia["N_params_B"] > 0) & (pythia["D_tokens_B"] > 0) & (pythia["val_loss"] > 0)]
    classic_params, _, rmse_b1, r2_b1 = fit_classic(pythia)
    names = ["E", "A", "alpha", "B", "beta"]
    print("\n【B1 经典标度律】单位：N、D 均为十亿")
    for name, value in zip(names, classic_params):
        print(f"  {name} = {value:.6f}")
    print(f"  拟合 RMSE={rmse_b1:.4f}, R^2={r2_b1:.4f}")

    # 半合成质量实验：估计 gamma。B6 为主，B7/B8 做对照，并标明不是直接观测。
    gamma_rows = []
    gamma_main = None
    for label, filename in [
        ("B6_半合成基础", "supplementary_NQ_experiment.csv"),
        ("B7_半合成扩展", "supplementary_NQ_experiment_expanded.csv"),
        ("B8_半合成大规模含外推", "supplementary_NQ_experiment_large.csv"),
    ]:
        table = pd.read_csv(DATA / filename).dropna()
        gamma, rmse = fit_gamma(table, classic_params, q_ref_b6=1.0)
        gamma_rows.append({"data": label, "gamma": gamma, "rmse": rmse, "n": len(table), "credibility": "半合成，非直接观测"})
        print(f"{label}: gamma={gamma:.4f}, RMSE={rmse:.4f}, n={len(table)}")
        if gamma_main is None:
            gamma_main = gamma
    pd.DataFrame(gamma_rows).to_csv(OUT / "质量指数估计.csv", index=False, encoding="utf-8-sig")

    params = (
        float(classic_params[0]),
        float(classic_params[1]),
        float(classic_params[2]),
        float(classic_params[3]),
        float(classic_params[4]),
        float(gamma_main),
    )

    # 验证：B2 族外，B3 插值轨迹，B4 跨族，B5 文献。这些表没有独立的 Q，
    # 按假设取 Q=Q_ref，此时广义律等于经典律。
    checks = []
    cerebras = pd.read_csv(DATA / "cerebras_training_log.csv").dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    pred = classic_law((cerebras["N_params_B"], cerebras["D_tokens_B"]), *classic_params)
    row = regression_metrics(cerebras["val_loss"], pred)
    row.update({"data": "B2_Cerebras半合成轨迹", "credibility": "半合成"})
    checks.append(row)

    traj_files = sorted((DATA / "training_trajectories").glob("*.csv"))
    traj = pd.concat([pd.read_csv(path) for path in traj_files], ignore_index=True)
    traj = traj.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    pred = classic_law((traj["N_params_B"], traj["D_tokens_B"]), *classic_params)
    row = regression_metrics(traj["val_loss"], pred)
    row.update({"data": "B3_Pythia插值轨迹", "credibility": "插值，不是新实验"})
    checks.append(row)

    for label, filename, credibility in [
        ("B4_跨族收敛点", "scaling_baseline.csv", "真实公开点"),
        ("B5_文献标度律", "published_scaling_data.csv", "文献整理"),
        ("B10_大模型预估损失", "supplementary_large_baseline.csv", "估算值，不能当作观测来证明模型"),
    ]:
        table = pd.read_csv(DATA / filename).dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
        table = table[(table["N_params_B"] > 0) & (table["D_tokens_B"] > 0)]
        pred = classic_law((table["N_params_B"], table["D_tokens_B"]), *classic_params)
        row = regression_metrics(table["val_loss"], pred)
        row.update({"data": label, "credibility": credibility})
        checks.append(row)
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT / "标度律验证.csv", index=False, encoding="utf-8-sig")
    print("\n【验证】Q 取参照值时退化为经典律")
    print(check_df.round(4).to_string(index=False))

    # 百亿以上：B9 给出真实参数规模，B10 的损失是估算。只报告模型外推与估算表的偏差。
    large = pd.read_csv(DATA / "supplementary_large_models.csv")
    print(f"\n【B9】百亿以上模型元数据 {len(large)} 行，用于确认外推区间，不直接提供验证损失。")

    # 代表点：约 7B 参数、300B token、质量等于问题一的参照质量
    n_star, d_star, q_star = 6.9, 300.0, q_ref
    elas = elasticities(params, n_star, d_star, q_star, q_ref)
    equiv = quality_equivalent_params(params, n_star, d_star, q_star, q_ref, delta_q=0.1)
    print("\n【代表点弹性】N=6.9B, D=300B, Q=Q_ref")
    print(f"  预测损失 L={elas['L']:.4f}")
    print(f"  ε_N={elas['eps_N']:.4f}  ε_D={elas['eps_D']:.4f}  ε_Q={elas['eps_Q']:.4f}")
    print("  弹性为负：增大该因素会降低损失。绝对值越大，越敏感。")
    print("\n【质量提升 0.1 的参数替代】")
    print(f"  原参数 {equiv['N_old']:.2f}B；质量提高 0.1 后，等价的参数量约为 {equiv['N_new']:.2f}B")
    print(f"  参数比 N_new/N_old = {equiv['N_ratio']:.4f}")
    print("  条件：固定 D 与其余参数，令 L(N_new, D, Q)=L(N, D, Q+0.1)。")

    # 领域替代/互补：问题一系数
    coef_path = ROOT / "结果" / "问题一" / "配比系数.csv"
    coef = pd.read_csv(coef_path)
    helpful = coef.loc[coef["coef"] < 0, "domain"].tolist()
    harmful = coef.loc[coef["coef"] >= 0, "domain"].tolist()
    print("\n【领域替代与互补】")
    print("降低损失的领域（可相互替代，优先分配）：", "、".join(helpful[:8]))
    print("抬高损失的领域（与前者互补约束：比例过高会抵消收益）：", "、".join(harmful[:8]))
    print("最优配比（来自问题一）前三项：")
    top_p = sorted(optimal_p.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for domain, share in top_p:
        print(f"  {domain}: {share:.3f}")

    pd.DataFrame([elas | equiv | {"gamma": gamma_main, "Q_ref": q_ref}]).to_csv(
        OUT / "弹性与替代.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "E": params[0],
        "A": params[1],
        "alpha": params[2],
        "B": params[3],
        "beta": params[4],
        "gamma": params[5],
        "Q_ref": q_ref,
        "rmse_B1": rmse_b1,
        "r2_B1": r2_b1,
        "elasticity": elas,
        "quality_delta_0.1": equiv,
        "note": "gamma 来自半合成 B6；Q=Q_ref 时退化为经典律。N 与 D 的单位是十亿。",
    }
    (OUT / "问题二输出.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图：Pythia 上预测值与真实值
    pred_b1 = classic_law((pythia["N_params_B"], pythia["D_tokens_B"]), *classic_params)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pythia["val_loss"], pred_b1, s=12, alpha=0.5, c="#2F6F8F")
    low = min(pythia["val_loss"].min(), pred_b1.min())
    high = max(pythia["val_loss"].max(), pred_b1.max())
    ax.plot([low, high], [low, high], color="#C62828", linewidth=1)
    ax.set_xlabel("真实验证损失")
    ax.set_ylabel("经典标度律预测")
    ax.set_title("问题二：Pythia 轨迹拟合")
    fig.tight_layout()
    fig.savefig(OUT / "经典标度律拟合.png", dpi=150)
    plt.close(fig)

    print(f"\n结果目录：{OUT}")
    # 绘图脚本放在「论文配图」文件夹。先把该文件夹加入搜索路径，才能导入。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "论文配图"))
    from 论文配图 import draw_problem2
    draw_problem2()


if __name__ == "__main__":
    main()
