"""伪随机数生成器（从零实现，不依赖 numpy / random 模块）。

核心算法：
- splitmix64：用于从种子派生独立子流；
- xoshiro256**：主生成器，状态可完整导出/恢复，供分片断点续算；
- 泊松采样：小速率 Knuth 连乘，大速率 transformed rejection（poidev 思路）。

所有分布只依赖本模块的均匀随机源，保证任务恢复后随机流严格延续。
"""

from __future__ import annotations

import math

_U64_MASK = (1 << 64) - 1
_U53_DIV = 1 << 53


def splitmix64(x: int) -> tuple[int, int]:
    """splitmix64 步进，返回 (新状态, 混合输出)。"""
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    z = z ^ (z >> 31)
    return x, z


def derive_seed(master_seed: int, label: str) -> int:
    """由主种子与文本标签派生出互不相关的 64 位子流种子。"""
    h = master_seed & _U64_MASK
    for ch in label.encode("utf-8"):
        h, z = splitmix64(h ^ ch)
    _, z = splitmix64(h)
    return z


class Xoshiro256:
    """xoshiro256** 生成器，状态为 4 个 uint64。"""

    def __init__(self, seed: int):
        self._s = [0, 0, 0, 0]
        state = seed & _U64_MASK
        for i in range(4):
            state, z = splitmix64(state)
            self._s[i] = z
        self._poisson_cache: dict[float, tuple[float, float, float]] = {}
        self.calls = 0  # 基础均匀随机数取用次数（费用计量，随状态持久化）

    # ---- 状态持久化 -------------------------------------------------
    def getstate(self) -> dict:
        return {"s": list(self._s), "calls": self.calls}

    def setstate(self, state: dict) -> None:
        self._s = [int(x) & _U64_MASK for x in state["s"]]
        if len(self._s) != 4:
            raise ValueError("xoshiro 状态必须包含 4 个 uint64")
        self.calls = int(state.get("calls", 0))

    # ---- 基础均匀随机 -----------------------------------------------
    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) | (x >> (64 - k))) & _U64_MASK

    def next_uint64(self) -> int:
        self.calls += 1
        s = self._s
        result = (self._rotl((s[1] * 5) & _U64_MASK, 7) * 9) & _U64_MASK
        t = (s[1] << 17) & _U64_MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)
        return result

    def random(self) -> float:
        """[0, 1) 上的 53 位均匀随机数。"""
        return (self.next_uint64() >> 11) / _U53_DIV

    def randint(self, low: int, high: int) -> int:
        """[low, high] 闭区间均匀整数。"""
        span = high - low + 1
        return low + int(self.random() * span)

    def chance(self, p: float) -> bool:
        return self.random() < p

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.random()

    # ---- 常用连续分布 -----------------------------------------------
    def expovariate(self, rate: float) -> float:
        u = self.random()
        # 避免 log(0)
        if u == 0.0:
            u = 1.0 / _U53_DIV
        return -math.log(u) / rate

    # ---- 离散分布 ---------------------------------------------------
    def geometric(self, p: float) -> int:
        """支持 {1, 2, ...} 的几何分布，P(X=k)=p(1-p)^(k-1)，均值 1/p。"""
        u = self.random()
        if u == 0.0:
            u = 1.0 / _U53_DIV
        return 1 + int(math.log(u) / math.log(1.0 - p))

    def poisson(self, lam: float) -> int:
        """泊松采样。lam<12 用 Knuth，否则用 transformed rejection（poidev）。"""
        if lam < 12.0:
            return self._poisson_knuth(lam)
        return self._poisson_rejection(lam)

    def _poisson_knuth(self, lam: float) -> int:
        limit = math.exp(-lam)
        count = 0
        prod = 1.0
        while True:
            count += 1
            prod *= self.random()
            if prod <= limit:
                return count - 1

    def _poisson_rejection(self, lam: float) -> int:
        # 自包含拒绝采样：以众数 m=floor(λ) 为中心的离散拉普拉斯提议
        #   g(k) ∝ exp(-|k-m|/b)，b 取 sqrt(λ)；
        # 拉普拉斯是指数尾，缩放系数 b 自适应放大直到其在整个支撑上
        # 严格覆盖泊松 pmf（对数凹 + 指数尾保证必然可覆盖）。
        # 包络常数 M=max f/g 显式求出，接受率 f/(M·g) 处处 <= 1。
        cache = self._poisson_cache.get(lam)
        if cache is None:
            m = int(math.floor(lam))
            log_lam = math.log(lam)
            b = math.sqrt(lam)
            width = max(20, int(8 * b) + 10)
            for _ in range(30):
                ks = list(range(max(0, m - width), m + width + 1))
                log_f = [(-lam + k * log_lam - math.lgamma(k + 1)) for k in ks]
                log_g = [-abs(k - m) / b for k in ks]
                log_g_norm_const = math.log(sum(math.exp(v) for v in log_g))
                log_g_normed = [v - log_g_norm_const for v in log_g]
                log_m = max(log_f[i] - log_g_normed[i]
                            for i in range(len(ks))) + 1e-12
                ok = all(log_f[i] <= log_m + log_g_normed[i] + 1e-9
                         for i in range(len(ks)))
                # 右边缘（扩展侧）概率质量低于阈值；左边缘 k=0 是自然边界
                right_edge = math.exp(log_f[-1])
                total_mass = sum(math.exp(v) for v in log_f)
                if ok and right_edge < 1e-14 and total_mass > 1 - 1e-10:
                    break
                b *= 1.25
                width = int(width * 1.2) + 5
            else:
                raise RuntimeError("泊松拒绝采样支撑扩展失败")
            g_unnorm = [math.exp(v) for v in log_g]
            g_norm = sum(g_unnorm)
            probs = [v / g_norm for v in g_unnorm]
            log_accept = [log_f[i] - log_g_normed[i] - log_m
                          for i in range(len(ks))]
            index = {k: i for i, k in enumerate(ks)}
            cache = (ks, probs, index, log_accept)
            self._poisson_cache[lam] = cache
        ks, probs, index, log_accept = cache
        while True:
            u = self.random()
            cum = 0.0
            k = ks[-1]
            for kk, p in zip(ks, probs):
                cum += p
                if u <= cum:
                    k = kk
                    break
            if math.log(self.random()) <= log_accept[index[k]]:
                return k
