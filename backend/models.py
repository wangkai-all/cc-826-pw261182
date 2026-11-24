"""虚构排队系统的到达流与服务过程模型。

模型仅用于教学性质的罕见事件模拟，不对应任何真实系统。

离散时间约定（每个时隙 t）：
1. 产生本时隙到达 a；
2. 若 q + a > 容量 C -> 本时隙发生首次溢出，路径终止（先于服务判定）；
3. 否则 q += a，再产生本时隙服务完成数 c，q <- max(0, q - c)。

重要抽样采用指数倾斜：到达速率整体乘以 r = e^theta (theta>=0)，
到达之外的随机机制（MMPP 相位切换、服务/停顿相位切换）概率律不变，
因此这些切换对似然比贡献恒为 1；似然比按**每个时隙的实际抽样结果**
逐段累计，包括触发溢出的那一次到达抽样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rng import Xoshiro256


# ---------------------------------------------------------------- 到达流
class ArrivalModel:
    kind = "base"

    def initial_state(self) -> dict:
        return {}

    def sample(self, rng: Xoshiro256, state: dict, tilt: float) -> tuple[int, dict]:
        """在倾斜倍率 tilt 下抽样一个时隙的到达数，返回 (到达数, 新状态)。"""
        raise NotImplementedError

    def log_lr_factor(self, arrivals: int, state: dict, tilt: float) -> float:
        """单次到达抽样的对数似然比 log P_nom(a)/P_tilt(a)。"""
        raise NotImplementedError

    def pmf(self, arrivals: int, tilt: float = 1.0, phase: int | None = None) -> float:
        """单时隙到达数的概率质量（精确求解器使用），自动截断到 max_arrivals。"""
        raise NotImplementedError

    def max_arrivals(self) -> int:
        raise NotImplementedError

    def params(self) -> dict:
        raise NotImplementedError


class PoissonArrival(ArrivalModel):
    """齐次泊松到达，速率 lam。倾斜后速率为 lam * tilt。"""

    kind = "poisson"

    def __init__(self, lam: float, max_arrivals: int | None = None):
        self.lam = float(lam)
        self._max = max_arrivals

    def initial_state(self) -> dict:
        return {}

    def sample(self, rng, state, tilt):
        return rng.poisson(self.lam * tilt), dict(state)

    def log_lr_factor(self, arrivals, state, tilt):
        # 正确方向 dP_nom/dP_tilt = e^{-lam+lam*r} * r^{-a}
        return self.lam * (tilt - 1.0) - arrivals * math.log(tilt)

    def pmf(self, arrivals, tilt=1.0, phase=None):
        rate = self.lam * tilt
        return math.exp(-rate) * rate ** arrivals / math.factorial(arrivals)

    def max_arrivals(self):
        if self._max is not None:
            return self._max
        # 精确求解器用的安全截断（超出质量 < 1e-12 量级）
        cap = max(8, int(self.lam * 6) + 6)
        return cap

    def params(self):
        return {"kind": self.kind, "lam": self.lam}


class MmppArrival(ArrivalModel):
    """两状态马尔可夫调制泊松过程（突发到达）。

    phase=0 平静期速率 lam0，phase=1 突发期速率 lam1；
    相位按转移概率 P 切换，切换律在倾斜测度下保持不变。
    """

    kind = "mmpp"

    def __init__(self, lam0: float, lam1: float, p01: float, p10: float,
                 max_arrivals: int | None = None):
        self.lam0, self.lam1 = float(lam0), float(lam1)
        self.p01, self.p10 = float(p01), float(p10)
        self._max = max_arrivals

    def initial_state(self):
        return {"phase": 0}

    def _rate(self, phase: int, tilt: float) -> float:
        return (self.lam1 if phase == 1 else self.lam0) * tilt

    def sample(self, rng, state, tilt):
        # 先按（两测度共同的）相位转移律切换，再在新相位上发射泊松到达
        phase = state.get("phase", 0)
        switch_p = self.p01 if phase == 0 else self.p10
        if rng.chance(switch_p):
            phase = 1 - phase
        a = rng.poisson(self._rate(phase, tilt))
        return a, {"phase": phase}

    def log_lr_factor(self, arrivals, state, tilt):
        # 相位切换律不变 -> 只计泊松发射项；state 是切换后的发射相位
        rate = self.lam1 if state.get("phase", 0) == 1 else self.lam0
        return rate * (tilt - 1.0) - arrivals * math.log(tilt)

    def pmf(self, arrivals, tilt=1.0, phase=None):
        if phase is None:
            # 稳态混合分布（仅供参考；精确求解器按相位分别建状态）
            pi1 = self.p01 / (self.p01 + self.p10) if self.p01 + self.p10 > 0 else 0.0
            return (1 - pi1) * self._pmf_poisson(self.lam0 * tilt, arrivals) \
                + pi1 * self._pmf_poisson(self.lam1 * tilt, arrivals)
        rate = self._rate(phase, tilt)
        return self._pmf_poisson(rate, arrivals)

    @staticmethod
    def _pmf_poisson(rate, a):
        return math.exp(-rate) * rate ** a / math.factorial(a)

    def phase_transition(self, phase: int) -> list[tuple[int, float]]:
        """返回 [(新相位, 概率), ...]。"""
        if phase == 0:
            return [(0, 1 - self.p01), (1, self.p01)]
        return [(0, self.p10), (1, 1 - self.p10)]

    def max_arrivals(self):
        if self._max is not None:
            return self._max
        return max(8, int(max(self.lam0, self.lam1) * 6) + 6)

    def params(self):
        return {"kind": self.kind, "lam0": self.lam0, "lam1": self.lam1,
                "p01": self.p01, "p10": self.p10}


class MixtureArrival(ArrivalModel):
    """独立同分布的混合泊松到达：每时隙以 p_high 概率走高速率流。

    混合权重在倾斜测度下不变，似然比无法写成简单指数形式，
    因此直接用两个 pmf 之比计算——用于检验通用似然比累计逻辑。
    """

    kind = "mixture"

    def __init__(self, lam_low: float, lam_high: float, p_high: float,
                 max_arrivals: int | None = None):
        self.lam_low, self.lam_high = float(lam_low), float(lam_high)
        self.p_high = float(p_high)
        self._max = max_arrivals

    def initial_state(self):
        return {}

    def _pmf_poisson(self, rate, a):
        return math.exp(-rate) * rate ** a / math.factorial(a)

    def pmf(self, arrivals, tilt=1.0, phase=None):
        pl = self._pmf_poisson(self.lam_low * tilt, arrivals)
        ph = self._pmf_poisson(self.lam_high * tilt, arrivals)
        return (1 - self.p_high) * pl + self.p_high * ph

    def sample(self, rng, state, tilt):
        if rng.chance(self.p_high):
            return rng.poisson(self.lam_high * tilt), {}
        return rng.poisson(self.lam_low * tilt), {}

    def log_lr_factor(self, arrivals, state, tilt):
        # 混合权重不变，似然比为两个混合泊松 pmf 之比（dP_nom/dP_tilt）
        nom = self.pmf(arrivals, 1.0)
        tilted = self.pmf(arrivals, tilt)
        if nom <= 0.0 or tilted <= 0.0:
            # 数值下溢保护
            return -700.0 if tilted <= 0.0 else 700.0
        return math.log(nom / tilted)

    def max_arrivals(self):
        if self._max is not None:
            return self._max
        return max(8, int(self.lam_high * 6) + 6)

    def params(self):
        return {"kind": self.kind, "lam_low": self.lam_low,
                "lam_high": self.lam_high, "p_high": self.p_high}


# ---------------------------------------------------------------- 服务过程
class ServiceModel:
    kind = "base"

    def initial_state(self) -> dict:
        return {}

    def completions(self, rng: Xoshiro256, state: dict, backlog: int) -> tuple[int, dict]:
        raise NotImplementedError

    def support(self) -> list[int]:
        """单时隙可能的完成数（精确求解器用）。"""
        raise NotImplementedError

    def probs(self, state: dict) -> list[tuple[int, float]]:
        """给定服务状态下 [(完成数, 概率), ...]。"""
        raise NotImplementedError

    def transition(self, state: dict) -> list[tuple[dict, float]]:
        """服务内部相位（如停顿）的状态转移，默认无内部状态。"""
        return [(dict(state), 1.0)]

    def sample_state(self, rng: Xoshiro256, state: dict) -> dict:
        """推进服务内部相位（如停顿）；无内部状态时原样返回。"""
        return dict(state)

    def params(self) -> dict:
        raise NotImplementedError


class BernoulliService(ServiceModel):
    """每时隙以概率 p 完成 1 个作业，否则完成 0 个。"""

    kind = "bernoulli"

    def __init__(self, p: float):
        self.p = float(p)

    def completions(self, rng, state, backlog):
        return (1 if rng.chance(self.p) else 0), dict(state)

    def support(self):
        return [0, 1]

    def probs(self, state):
        return [(0, 1 - self.p), (1, self.p)]

    def params(self):
        return {"kind": self.kind, "p": self.p}


class DeterministicService(ServiceModel):
    """每时隙固定完成 rate 个作业（受实际积压截断）。"""

    kind = "deterministic"

    def __init__(self, rate: int):
        self.rate = int(rate)

    def completions(self, rng, state, backlog):
        return min(self.rate, backlog), dict(state)

    def support(self):
        return list(range(self.rate + 1))

    def probs(self, state):
        return [(self.rate, 1.0)]

    def params(self):
        return {"kind": self.kind, "rate": self.rate}


class StallingService(ServiceModel):
    """带服务停顿的两相位马尔可夫服务。

    up 相位每时隙完成 s_up 个（确定值），down 相位完成 s_down（通常 0）；
    p_ud / p_du 为相位切换概率。服务律不做倾斜，似然比贡献为 1。
    """

    kind = "stalling"

    def __init__(self, s_up: int, s_down: int, p_ud: float, p_du: float):
        self.s_up, self.s_down = int(s_up), int(s_down)
        self.p_ud, self.p_du = float(p_ud), float(p_du)

    def initial_state(self):
        return {"svc_phase": 1}  # 从 up 开始

    def transition(self, state):
        phase = state.get("svc_phase", 1)
        if phase == 1:
            return [({"svc_phase": 1}, 1 - self.p_ud),
                    ({"svc_phase": 0}, self.p_ud)]
        return [({"svc_phase": 1}, self.p_du),
                ({"svc_phase": 0}, 1 - self.p_du)]

    def sample_state(self, rng, state):
        phase = state.get("svc_phase", 1)
        switch_p = self.p_ud if phase == 1 else self.p_du
        if rng.chance(switch_p):
            return {"svc_phase": 1 - phase}
        return {"svc_phase": phase}

    def completions(self, rng, state, backlog):
        phase = state.get("svc_phase", 1)
        c = self.s_up if phase == 1 else self.s_down
        return min(c, backlog), dict(state)

    def support(self):
        return sorted({self.s_up, self.s_down})

    def probs(self, state):
        phase = state.get("svc_phase", 1)
        c = self.s_up if phase == 1 else self.s_down
        return [(c, 1.0)]

    def params(self):
        return {"kind": self.kind, "s_up": self.s_up, "s_down": self.s_down,
                "p_ud": self.p_ud, "p_du": self.p_du}


# ---------------------------------------------------------------- 场景
@dataclass
class Scenario:
    name: str
    capacity: int
    horizon: int
    arrival: ArrivalModel
    service: ServiceModel
    description: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "horizon": self.horizon,
            "description": self.description,
            "arrival": self.arrival.params(),
            "service": self.service.params(),
            "extra": self.extra,
        }


def build_arrival(cfg: dict) -> ArrivalModel:
    kind = cfg["kind"]
    if kind == "poisson":
        return PoissonArrival(cfg["lam"], cfg.get("max_arrivals"))
    if kind == "mmpp":
        return MmppArrival(cfg["lam0"], cfg["lam1"], cfg["p01"], cfg["p10"],
                           cfg.get("max_arrivals"))
    if kind == "mixture":
        return MixtureArrival(cfg["lam_low"], cfg["lam_high"], cfg["p_high"],
                              cfg.get("max_arrivals"))
    raise ValueError(f"未知到达模型: {kind}")


def build_service(cfg: dict) -> ServiceModel:
    kind = cfg["kind"]
    if kind == "bernoulli":
        return BernoulliService(cfg["p"])
    if kind == "deterministic":
        return DeterministicService(cfg["rate"])
    if kind == "stalling":
        return StallingService(cfg["s_up"], cfg["s_down"], cfg["p_ud"], cfg["p_du"])
    raise ValueError(f"未知服务模型: {kind}")


def build_scenario(cfg: dict) -> Scenario:
    return Scenario(
        name=cfg["name"],
        capacity=int(cfg["capacity"]),
        horizon=int(cfg["horizon"]),
        arrival=build_arrival(cfg["arrival"]),
        service=build_service(cfg["service"]),
        description=cfg.get("description", ""),
        extra=cfg.get("extra", {}),
    )
