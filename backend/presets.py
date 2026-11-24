"""内置虚构场景与默认模拟参数。

所有场景均为教学构造，不代表任何真实系统；不构成容量保证。
"""

from __future__ import annotations

from .models import build_scenario

DEFAULT_TILT_GRID = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

PRESETS = {
    "baseline": {
        "label": "基准：泊松到达 + 伯努利服务（稀有溢出）",
        "config": {
            "name": "baseline",
            "capacity": 14,
            "horizon": 25,
            "description": "低强度泊松流，单时隙到达超过容量余量的概率很小，"
                           "普通模拟常出现零命中。",
            "arrival": {"kind": "poisson", "lam": 0.8},
            "service": {"kind": "bernoulli", "p": 0.9},
            "extra": {},
        },
        "default_n": 20000,
        "pilot_n": 4000,
        "tilt_grid": DEFAULT_TILT_GRID,
        "exact_eligible": True,
    },
    "bursty_mmpp": {
        "label": "突发到达：两状态 MMPP",
        "config": {
            "name": "bursty_mmpp",
            "capacity": 32,
            "horizon": 25,
            "description": "平静期速率 0.4，突发期速率 3.5；突发相位很少进入但持续，"
                           "溢出主要由突发段累积造成。",
            "arrival": {"kind": "mmpp", "lam0": 0.4, "lam1": 3.5,
                        "p01": 0.04, "p10": 0.25},
            "service": {"kind": "bernoulli", "p": 0.95},
            "extra": {},
        },
        "default_n": 20000,
        "pilot_n": 4000,
        "tilt_grid": DEFAULT_TILT_GRID,
        "exact_eligible": True,
    },
    "service_stall": {
        "label": "服务停顿：两相位可停顿服务",
        "config": {
            "name": "service_stall",
            "capacity": 26,
            "horizon": 25,
            "description": "服务偶发进入停顿相位且恢复缓慢，停顿期间队列只增不减；"
                           "服务机制不参与倾斜，似然比完全来自到达。",
            "arrival": {"kind": "poisson", "lam": 0.9},
            "service": {"kind": "stalling", "s_up": 1, "s_down": 0,
                        "p_ud": 0.03, "p_du": 0.10},
            "extra": {},
        },
        "default_n": 20000,
        "pilot_n": 4000,
        "tilt_grid": DEFAULT_TILT_GRID,
        "exact_eligible": True,
    },
    "extreme_weights": {
        "label": "极端权重：混合流 + 强偏置（权重塌缩样例）",
        "config": {
            "name": "extreme_weights",
            "capacity": 12,
            "horizon": 30,
            "description": "90% 时隙为低速率流、10% 为高速率流；若强偏置（倍率 3），"
                           "名义/倾斜测度差异被路径长度指数放大，权重分布极度不均，"
                           "用于演示有效样本量塌缩与不可靠区间。",
            "arrival": {"kind": "mixture", "lam_low": 0.5, "lam_high": 3.0,
                        "p_high": 0.10},
            "service": {"kind": "deterministic", "rate": 1},
            "extra": {},
        },
        "default_n": 20000,
        "pilot_n": 4000,
        "tilt_grid": [1.2, 1.5, 2.0, 3.0],
        "forced_tilt": 3.0,          # 演示用：强制强偏置跑一条塌缩对照
        "exact_eligible": True,
    },
    "tiny_exact": {
        "label": "小状态核对场（精确转移可算）",
        "config": {
            "name": "tiny_exact",
            "capacity": 3,
            "horizon": 12,
            "description": "容量与时域都很小，精确马尔可夫链几秒内可算，"
                           "用于核对普通模拟与重要抽样的偏差和区间覆盖率。",
            "arrival": {"kind": "poisson", "lam": 0.6, "max_arrivals": 8},
            "service": {"kind": "bernoulli", "p": 0.5},
            "extra": {},
        },
        "default_n": 10000,
        "pilot_n": 3000,
        "tilt_grid": [1.5, 2.0, 3.0],
        "exact_eligible": True,
    },
}


def preset_scenario(preset_id: str):
    p = PRESETS[preset_id]
    return build_scenario(p["config"])
