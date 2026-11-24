"""小状态系统的精确有限时域首次溢出概率（马尔可夫链逐时隙转移核对）。

状态 = (队列长度, 到达相位, 服务相位)。每个时隙：
  到达 a 使 q+a>C -> 累入首次溢出吸收质量；
  否则完成服务 c 后 q'=max(0,q+a-c)，相位按各自转移律推进。
只适用于状态空间较小的场景；超限会拒绝计算而不是硬跑。
"""

from __future__ import annotations

from .models import Scenario, MmppArrival, StallingService, DeterministicService

MAX_STATES = 6000


def _arrival_phases(scenario: Scenario) -> list[int]:
    if isinstance(scenario.arrival, MmppArrival):
        return [0, 1]
    return [0]


def _service_phases(scenario: Scenario) -> list[int]:
    if isinstance(scenario.service, StallingService):
        return [0, 1]
    return [0]


def exact_overflow(scenario: Scenario, tilt: float = 1.0,
                   verbose: bool = False) -> dict:
    """返回 horizon 内首次溢出的精确概率与时隙级首次命中分布。"""
    arrival, service = scenario.arrival, scenario.service
    c, horizon = scenario.capacity, scenario.horizon
    a_phases = _arrival_phases(scenario)
    s_phases = _service_phases(scenario)

    states = [(q, ap, sp) for q in range(c + 1)
              for ap in a_phases for sp in s_phases]
    if len(states) > MAX_STATES:
        raise ValueError(
            f"状态空间 {len(states)} 超过精确求解上限 {MAX_STATES}，请缩小容量/相位数")

    index = {s: i for i, s in enumerate(states)}
    amax = arrival.max_arrivals()

    # 预构建每个状态的 (存活转移 dict, 溢出质量)
    live_rows: list[dict[int, float]] = [dict() for _ in states]
    overflow_p = [0.0] * len(states)
    for (q, ap, sp), i in index.items():
        # 到达相位转移（MMPP 的相位切换在抽样到达前发生，律不随倾斜变）
        if isinstance(arrival, MmppArrival):
            ap_options = arrival.phase_transition(ap)
        else:
            ap_options = [(ap, 1.0)]
        # 服务相位转移
        if isinstance(service, StallingService):
            sp_options = service.transition({"svc_phase": sp})
        else:
            sp_options = [({"svc_phase": sp}, 1.0)]

        for ap2, pap in ap_options:
            for a in range(amax + 1):
                pa = arrival.pmf(a, tilt=tilt, phase=ap2)
                if pa <= 0.0:
                    continue
                if q + a > c:
                    overflow_p[i] += pap * pa
                    continue
                q_after_arr = q + a
                for sp_dict, psp in sp_options:
                    sp2 = sp_dict.get("svc_phase", sp)
                    # 完成数支持（确定性服务需受积压截断）
                    comp_options = service.probs({"svc_phase": sp2})
                    for comp, pc in comp_options:
                        comp = min(comp, q_after_arr)
                        q2 = max(0, q_after_arr - comp)
                        j = index[(q2, ap2, sp2)]
                        live_rows[i][j] = live_rows[i].get(j, 0.0) \
                            + pap * pa * psp * pc

    # 行和归一化检查（截断误差监测）
    row_sums = [sum(r.values()) + overflow_p[i] for i, r in enumerate(live_rows)]
    norm_err = max(abs(1.0 - s) for s in row_sums)

    init_ap = arrival.initial_state().get("phase", 0)
    init_sp = service.initial_state().get("svc_phase", 0)
    v = [0.0] * len(states)
    v[index[(0, init_ap, init_sp)]] = 1.0

    first_passage = []
    cumulative = 0.0
    alive_mass = 1.0
    for _t in range(horizon):
        hit_mass = sum(v[i] * overflow_p[i] for i in range(len(states)) if v[i])
        nv = [0.0] * len(states)
        for i, mass in enumerate(v):
            if mass == 0.0:
                continue
            row = live_rows[i]
            for j, p in row.items():
                nv[j] += mass * p
        v = nv
        alive_mass -= hit_mass
        cumulative += hit_mass
        first_passage.append(hit_mass)

    return {
        "probability": cumulative,
        "first_passage_by_slot": first_passage,
        "alive_mass": alive_mass,
        "state_count": len(states),
        "tilt": tilt,
        "row_sum_max_error": norm_err,
        "note": "截断到达支撑导致的行和偏差已报告；其余为精确浮点求和",
    }


def check_estimate(scenario: Scenario, estimate: dict, method: str) -> dict:
    """把模拟估计与精确概率对照（z 偏差、CI 覆盖）。"""
    exact = exact_overflow(scenario)
    p = exact["probability"]
    est, se = estimate["estimate"], estimate["se"]
    z = (est - p) / se if se > 0 else None
    covered = estimate["ci_low"] <= p <= estimate["ci_high"]
    return {
        "method": method,
        "exact": p,
        "estimate": est,
        "absolute_error": abs(est - p),
        "relative_error_vs_exact": abs(est - p) / p if p > 0 else None,
        "z_deviation": z,
        "ci_covers_exact": covered,
        "exact_state_count": exact["state_count"],
        "exact_row_sum_max_error": exact["row_sum_max_error"],
    }
