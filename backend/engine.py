"""模拟引擎：普通模拟（Crude MC）与重要抽样（IS）。

关键设计：
- 重要抽样只倾斜到达机制，似然比按路径上每个时隙的实际抽样结果逐段累加，
  与队列状态无关，绝不按最终队列长度事后补权；
- 权重在对数空间累加，数值稳定；在线维护一阶/二阶矩、有效样本量；
- 自适应选偏置只用独立试运行（pilot），正式估计阶段偏置倍率固定；
- 长计算按 shard 分片，每片结束回调持久化（随机流状态 + 已终止路径统计 +
  权重和）；取消时当前未终止路径直接丢弃，不计入任何统计。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .rng import Xoshiro256
from .models import Scenario

Z96 = 1.959963984540054  # 正态 97.5% 分位


# ---------------------------------------------------------------- 单条路径
def run_one_path(scenario: Scenario, rng: Xoshiro256, tilt: float,
                 record_trace: bool = False):
    """执行一条路径直到首次溢出或达到 horizon。

    返回 dict：hit、log_weight（整条路径对数似然比）、slots、trace。
    未终止路径不会产生结果——调用方在分片内只对完整路径记账。
    """
    arrival, service = scenario.arrival, scenario.service
    q = 0
    astate = arrival.initial_state()
    sstate = service.initial_state()
    log_w = 0.0
    trace = [] if record_trace else None

    for t in range(scenario.horizon):
        a, astate_new = arrival.sample(rng, astate, tilt)
        # 似然比按本时隙实际到达抽样逐段累计；对 MMPP 等带相位模型，
        # 权重使用切换后的发射相位（相位切换律本身两测度相同，贡献为 1）
        log_factor = arrival.log_lr_factor(a, astate_new, tilt)
        log_w += log_factor
        q_before = q
        overflow = q + a > scenario.capacity

        if record_trace:
            trace.append({
                "t": t,
                "arrival_phase": astate_new.get("phase"),
                "service_phase": sstate.get("svc_phase"),
                "arrivals": a,
                "queue_before": q_before,
                "overflow": overflow,
                "log_lr": log_factor,
                "log_weight_cum": log_w,
            })

        if overflow:  # 先于服务判定，路径立即终止
            return {"hit": True, "log_weight": log_w, "slots": t + 1,
                    "trace": trace}

        q = q + a
        sstate_new = service.sample_state(rng, sstate)
        c, _ = service.completions(rng, sstate_new, q)
        q = max(0, q - c)
        if record_trace:
            trace[-1]["service"] = c
            trace[-1]["queue_after"] = q
        astate, sstate = astate_new, sstate_new

    return {"hit": False, "log_weight": log_w, "slots": scenario.horizon,
            "trace": trace}


# ---------------------------------------------------------------- 权重累加器
class WeightAccumulator:
    """对数权重的在线累加，支持分片合并与完整序列化。

    维护统一尺度 m = 最大 log 权重：
      s1 = Σ exp(logw - m)，s2 = Σ exp(2(logw - m))，
    命中路径单独记 s1_hit / s2_hit（估计量只含命中项）。
    """

    def __init__(self):
        self.n = 0
        self.hits = 0
        self.slots = 0
        self.m = -math.inf
        self.s1_all = 0.0
        self.s2_all = 0.0
        self.s1_hit = 0.0
        self.s2_hit = 0.0
        self.max_hit_lw = -math.inf
        self.min_hit_lw = math.inf
        self.hit_logw: list[float] = []      # 截断保存，供直方图
        self.traces: list[dict] = []         # 截断保存，供网页查看
        self._hist_cap = 500
        self._trace_cap = 5

    def add(self, result: dict):
        lw = result["log_weight"]
        self.n += 1
        self.slots += result["slots"]
        if lw > self.m:
            if math.isfinite(self.m):
                shift = math.exp(self.m - lw)
                self.s1_all *= shift
                self.s2_all *= shift * shift
                self.s1_hit *= shift
                self.s2_hit *= shift * shift
            else:
                self.s1_all = self.s2_all = self.s1_hit = self.s2_hit = 0.0
            self.m = lw
        e1 = math.exp(lw - self.m)
        e2 = e1 * e1
        self.s1_all += e1
        self.s2_all += e2
        if result["hit"]:
            self.hits += 1
            self.s1_hit += e1
            self.s2_hit += e2
            self.max_hit_lw = max(self.max_hit_lw, lw)
            self.min_hit_lw = min(self.min_hit_lw, lw)
            if len(self.hit_logw) < self._hist_cap:
                self.hit_logw.append(lw)
            if len(self.traces) < self._trace_cap and result.get("trace"):
                self.traces.append(result["trace"])

    def to_dict(self) -> dict:
        return {
            "n": self.n, "hits": self.hits, "slots": self.slots,
            "m": None if not math.isfinite(self.m) else self.m,
            "s1_all": self.s1_all, "s2_all": self.s2_all,
            "s1_hit": self.s1_hit, "s2_hit": self.s2_hit,
            "max_hit_lw": None if not math.isfinite(self.max_hit_lw) else self.max_hit_lw,
            "min_hit_lw": None if not math.isfinite(self.min_hit_lw) else self.min_hit_lw,
            "hit_logw": self.hit_logw, "traces": self.traces,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightAccumulator":
        acc = cls()
        acc.n = d["n"]
        acc.hits = d["hits"]
        acc.slots = d["slots"]
        acc.m = -math.inf if d["m"] is None else d["m"]
        acc.s1_all, acc.s2_all = d["s1_all"], d["s2_all"]
        acc.s1_hit, acc.s2_hit = d["s1_hit"], d["s2_hit"]
        acc.max_hit_lw = -math.inf if d["max_hit_lw"] is None else d["max_hit_lw"]
        acc.min_hit_lw = math.inf if d["min_hit_lw"] is None else d["min_hit_lw"]
        acc.hit_logw = list(d.get("hit_logw", []))[: acc._hist_cap]
        acc.traces = list(d.get("traces", []))[: acc._trace_cap]
        return acc

    def estimates(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        mean = math.exp(self.m) * self.s1_hit / self.n if self.hits else 0.0
        second = math.exp(2.0 * self.m) * self.s2_hit / self.n if self.hits else 0.0
        var = max(second - mean * mean, 0.0) / self.n
        se = math.sqrt(var)
        ess_all = (self.s1_all ** 2 / self.s2_all) if self.s2_all > 0 else 0.0
        ess_hits = ((self.s1_hit ** 2 / self.s2_hit)
                    if self.hits and self.s2_hit > 0 else 0.0)
        return {
            "n": self.n, "hits": self.hits, "slots": self.slots,
            "estimate": mean, "variance": var, "se": se,
            "ci_low": max(mean - Z96 * se, 0.0),
            "ci_high": mean + Z96 * se,
            "rel_halfwidth": (Z96 * se / mean) if mean > 0 else None,
            "ess_all": ess_all,
            "ess_all_ratio": ess_all / self.n,
            "ess_hits": ess_hits,
            "ess_hits_ratio": (ess_hits / self.hits) if self.hits else None,
            "max_hit_logw": self.max_hit_lw if math.isfinite(self.max_hit_lw) else None,
            "min_hit_logw": self.min_hit_lw if math.isfinite(self.min_hit_lw) else None,
        }


# ---------------------------------------------------------------- 可靠性判定
def reliability(est: dict, method: str, collapse_threshold: float = 0.01) -> dict:
    """检查权重塌缩 / 零命中 / 区间失效，返回原因列表。"""
    reasons = []
    n = est["n"]
    if n == 0:
        return {"reliable": False, "reasons": ["尚无完成路径"]}
    if est["hits"] == 0:
        if method == "crude":
            reasons.append("零命中：普通模拟无法给出非零估计，仅能报 3/n 经验上界")
        else:
            reasons.append("零命中：重要抽样也未触发事件，估计为 0 且无信息量")
        return {"reliable": False, "reasons": reasons}
    if est["ess_all_ratio"] < collapse_threshold:
        reasons.append(
            f"权重塌缩：全体有效样本占比 {est['ess_all_ratio']:.2e} < {collapse_threshold:g}")
    if est["ess_hits_ratio"] is not None and est["ess_hits_ratio"] < collapse_threshold:
        reasons.append(
            f"权重塌缩：命中路径内部有效样本占比 {est['ess_hits_ratio']:.2e} < {collapse_threshold:g}")
    spread = (est["max_hit_logw"] - est["min_hit_logw"]) \
        if est["max_hit_logw"] is not None else 0.0
    if spread > 20.0:
        reasons.append(f"极端权重：命中路径对数权重跨度 {spread:.1f}（约 {math.exp(spread):.1e} 倍）")
    if est["rel_halfwidth"] is not None and est["rel_halfwidth"] > 1.0:
        reasons.append("95% 置信区间相对半宽超过 100%，区间不可靠")
    return {"reliable": not reasons, "reasons": reasons}


# ---------------------------------------------------------------- 分片运行
@dataclass
class MethodState:
    """可持久化的单方法运行状态。"""
    method: str
    tilt: float
    rng: Xoshiro256
    acc: WeightAccumulator
    elapsed: float = 0.0
    convergence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method, "tilt": self.tilt,
            "rng": self.rng.getstate(), "acc": self.acc.to_dict(),
            "elapsed": self.elapsed, "convergence": self.convergence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MethodState":
        rng = Xoshiro256(0)
        rng.setstate(d["rng"])
        return cls(method=d["method"], tilt=d["tilt"], rng=rng,
                   acc=WeightAccumulator.from_dict(d["acc"]),
                   elapsed=d.get("elapsed", 0.0),
                   convergence=d.get("convergence", []))


def run_shards(scenario: Scenario, state: MethodState, n_total: int,
               shard_size: int, is_cancelled=None, on_shard=None) -> MethodState:
    """运行至 n_total 条**已终止**路径；每片结束回调 on_shard(state)。

    取消语义：分片边界检查取消标志；被取消时当前分片内未跑完的路径不记账，
    已完成路径与随机流状态保持一致（rng 已推进，恢复后随机流不重复）。
    """
    while state.acc.n < n_total:
        if is_cancelled and is_cancelled():
            break
        budget = min(shard_size, n_total - state.acc.n)
        t0 = time.perf_counter()
        for _ in range(budget):
            want_trace = state.acc.hits < state.acc._trace_cap
            result = run_one_path(scenario, state.rng, state.tilt,
                                  record_trace=want_trace and state.method == "is")
            state.acc.add(result)
        state.elapsed += time.perf_counter() - t0
        est = state.acc.estimates()
        state.convergence.append({
            "n": est["n"], "estimate": est["estimate"], "se": est["se"],
            "hits": est["hits"],
        })
        if on_shard:
            on_shard(state)
    return state


# ---------------------------------------------------------------- 自适应试运行
def run_pilot(scenario: Scenario, n_pilot: int, tilt_grid: list[float],
              master_seed: int, shard_size: int = 1000) -> dict:
    """独立试运行：每个候选偏置倍率用独立随机流，按相对误差选最优。

    试运行结果不进入正式估计；返回候选表与选择结果。
    """
    from .rng import derive_seed

    table = []
    best = None
    for r in tilt_grid:
        rng = Xoshiro256(derive_seed(master_seed, f"pilot-r{r:.4f}"))
        st = MethodState(method="is", tilt=r, rng=rng, acc=WeightAccumulator())
        run_shards(scenario, st, n_pilot, shard_size)
        est = st.acc.estimates()
        rel = (est["se"] / est["estimate"]) if est["hits"] and est["estimate"] > 0 else None
        row = {"tilt": r, "hits": est["hits"], "estimate": est["estimate"],
               "se": est["se"], "rel_error": rel,
               "ess_all_ratio": est["ess_all_ratio"]}
        table.append(row)
        if rel is not None and (best is None or rel < best["rel_error"]):
            best = row
    chosen = best["tilt"] if best else max(tilt_grid)
    return {"grid": table, "chosen_tilt": chosen,
            "inconclusive": best is None, "n_pilot": n_pilot}


# ---------------------------------------------------------------- 费用与解释
def cost_metrics(state: MethodState) -> dict:
    est = state.acc.estimates()
    elapsed = state.elapsed
    calls = state.rng.calls
    var = est.get("variance", 0.0)
    fom_time = 1.0 / (var * elapsed) if var > 0 and elapsed > 0 else None
    fom_calls = 1.0 / (var * calls) if var > 0 and calls > 0 else None
    return {
        "elapsed_seconds": elapsed,
        "rng_calls": calls,
        "slots_simulated": est.get("slots", 0),
        "fom_time": fom_time,
        "fom_rng": fom_calls,
        "rel_error": (est["se"] / est["estimate"])
        if est.get("hits") and est["estimate"] > 0 else None,
    }
