# -*- coding: utf-8 -*-
"""
问题三：算力约束下的多维资源联合优化与结构性转移
================================================

决策变量：参数量 N、数据量 D、数据质量 Q。
领域配比 p 固定为问题一的最优配比，不再和 N、D 一起搜索。
理由：配比改变的是算力在领域间的分配效果，不增加总 FLOPs；
把它提前定下来，本问才能看清算力预算如何在 N、D、Q 之间转移。

上下文长度 L_ctx 是外生变量，取值来自附件 C7 里真实出现过的窗口长度。

总算力：
    C = C_train + C_quality + C_attn
    C_train   = 6 N D                         （N、D 用“个数”，不是十亿）
    C_quality = D * max(g(Q) - g(Q0), 0)
    C_attn    = η N D L_ctx ，η = 2e-4

注意力开销与训练开销相等时：
    η N D L_ctx = 6 N D  =>  L_ctx_crit = 6 / η = 30000
这个临界值由公式直接推出，不依赖主观假设。

预算三档：10^19、10^22、10^24 FLOPs。
目标：在 C ≤ 预算 时，用问题二的广义标度律把验证损失降到最低。
求解方法：对 N、D、Q 做对数/线性网格搜索。网格搜索步骤透明，适合入门实现。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结果" / "问题三"
OUT.mkdir(parents=True, exist_ok=True)

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


def main():
    q2_path = ROOT / "结果" / "问题二" / "问题二输出.json"
    q1_path = ROOT / "结果" / "问题一" / "问题一输出.json"
    if not q2_path.exists() or not q1_path.exists():
        raise SystemExit("请先依次运行问题一和问题二。")
    law = json.loads(q2_path.read_text(encoding="utf-8"))
    params = (law["E"], law["A"], law["alpha"], law["B"], law["beta"], law["gamma"])
    q_ref = float(law["Q_ref"])
    # 基线质量：不做额外清洗时，就用问题一配比已经达到的质量
    q0 = q_ref

    arch = pd.read_csv(ROOT / "real_attachments" / "C_efficiency_evolution" / "model_architecture_metadata.csv")
    lctx_values = sorted(int(v) for v in arch["max_position_embeddings"].dropna().unique())
    print("=" * 72)
    print("问题三  算力约束下的多维资源联合优化与结构性转移")
    print("=" * 72)
    print(f"C7 中出现的上下文长度：{lctx_values}")
    print(f"临界上下文长度 L_ctx_crit = 6/η = {LCTX_CRIT:.0f}")
    print("当 L_ctx 小于临界值时，注意力开销小于训练开销；大于临界值时则反过来。")
    print(f"基线质量 Q0 = Q_ref = {q0:.4f}。配比固定为问题一的最优配比。")

    # 主结果：把上下文放在 C7 的中位附近。若 8192 存在就用它，否则用列表中第一个 ≥2048 的值。
    if 8192 in lctx_values:
        base_lctx = 8192
    else:
        base_lctx = lctx_values[len(lctx_values) // 2]

    rows = []
    for cost_name, g_func in COST_FUNCS.items():
        for budget in BUDGETS:
            found = search_one(budget, g_func, base_lctx, q0, q_ref, params)
            if found is None:
                print(f"{cost_name} 预算 {budget:.0e} 无可行解")
                continue
            found["cost"] = cost_name
            found["budget"] = budget
            found["L_ctx"] = base_lctx
            rows.append(found)
            print(
                f"{cost_name} | C={budget:.0e} | N={found['N_B']:.2f}B D={found['D_B']:.1f}B "
                f"Q={found['Q']:.3f} L={found['loss']:.3f} 策略={found['strategy']} "
                f"花费比 训练/质量/注意力="
                f"{found['share_train']:.2f}/{found['share_quality']:.2f}/{found['share_attn']:.2f}"
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "三档预算最优配置.csv", index=False, encoding="utf-8-sig")

    print("\n【结构性转移】同一成本函数下，比较 1e19 与 1e24")
    shift_rows = []
    for cost_name in COST_FUNCS:
        part = result[result["cost"] == cost_name].sort_values("budget")
        if len(part) < 2:
            continue
        low = part.iloc[0]
        high = part.iloc[-1]
        judgement = structural_shift(low, high)
        shift_rows.append(
            {
                "cost": cost_name,
                "low_strategy": low["strategy"],
                "high_strategy": high["strategy"],
                "delta_quality_share": float(high["share_quality"] - low["share_quality"]),
                "judgement": judgement,
            }
        )
        print(f"  {cost_name}: {low['strategy']} -> {high['strategy']} ，{judgement}")
    pd.DataFrame(shift_rows).to_csv(OUT / "结构性转移判定.csv", index=False, encoding="utf-8-sig")

    # 敏感性：固定中档预算和对数成本（通常最可能投资质量），扫描 C7 的全部窗口
    print("\n【上下文长度敏感性】预算 1e22，对数渐进型成本")
    sens_rows = []
    for lctx in lctx_values:
        found = search_one(1e22, g_log, lctx, q0, q_ref, params)
        if found is None:
            continue
        found["L_ctx"] = lctx
        found["above_critical"] = int(lctx >= LCTX_CRIT)
        sens_rows.append(found)
        print(
            f"  L_ctx={lctx:<7} N={found['N_B']:.2f}B D={found['D_B']:.1f}B "
            f"Q={found['Q']:.3f} 注意力占比={found['share_attn']:.3f} 策略={found['strategy']}"
        )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "上下文敏感性.csv", index=False, encoding="utf-8-sig")

    # 图：三种成本下质量花费占比随预算变化
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cost_name, part in result.groupby("cost"):
        part = part.sort_values("budget")
        ax.plot(part["budget"], part["share_quality"], marker="o", label=cost_name)
    ax.set_xscale("log")
    ax.set_xlabel("算力预算（FLOPs）")
    ax.set_ylabel("质量提升花费占比")
    ax.set_title(f"问题三：质量投入占比（L_ctx={base_lctx}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "质量花费占比.png", dpi=150)
    plt.close(fig)

    payload = {
        "L_ctx_crit": LCTX_CRIT,
        "eta": ETA,
        "base_L_ctx": base_lctx,
        "Q0": q0,
        "budgets": BUDGETS,
        "feasible_L_ctx": lctx_values,
        "definition": "策略标签改变或质量花费占比变化超过 0.15，即结构性转移",
    }
    (OUT / "问题三输出.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果目录：{OUT}")
    # 绘图脚本放在「论文配图」文件夹。先把该文件夹加入搜索路径，才能导入。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "论文配图"))
    from 论文配图 import draw_problem3
    draw_problem3()


if __name__ == "__main__":
    main()
