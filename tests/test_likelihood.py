import math
import unittest

from backend.engine import MethodState, WeightAccumulator, run_one_path, run_shards
from backend.exact import check_estimate, exact_overflow
from backend.models import (BernoulliService, DeterministicService, MmppArrival,
                            MixtureArrival, PoissonArrival, Scenario,
                            StallingService)
from backend.rng import Xoshiro256, derive_seed


def make_scenario(arrival, service, cap=6, horizon=10):
    return Scenario(name="t", capacity=cap, horizon=horizon,
                    arrival=arrival, service=service)


class TestLikelihoodRatioIdentity(unittest.TestCase):
    """重加权恒等式：倾斜测度下 E_Q[L] = E_P[1] = 1（L=dP/dQ）。"""

    def test_poisson_single_slot_and_path(self):
        rng = Xoshiro256(1)
        lam, r = 1.5, 2.0
        arr = PoissonArrival(lam)
        # 单时隙：对抽样到的 a 重加权
        total = 0.0
        n = 200_000
        for _ in range(n):
            a, st = arr.sample(rng, {}, r)
            total += math.exp(arr.log_lr_factor(a, st, r))
        self.assertAlmostEqual(total / n, 1.0, delta=0.02)

    def test_full_path_weight_normalizes(self):
        sc = make_scenario(PoissonArrival(0.8), BernoulliService(0.7))
        rng = Xoshiro256(99)
        n = 40_000
        total = 0.0
        for _ in range(n):
            res = run_one_path(sc, rng, tilt=2.0)
            total += math.exp(res["log_weight"])
        self.assertAlmostEqual(total / n, 1.0, delta=0.03)

    def test_mmpp_weight_normalizes(self):
        sc = make_scenario(
            MmppArrival(0.4, 3.0, 0.1, 0.3), BernoulliService(0.8))
        rng = Xoshiro256(123)
        n = 40_000
        total = 0.0
        for _ in range(n):
            res = run_one_path(sc, rng, tilt=1.8)
            total += math.exp(res["log_weight"])
        self.assertAlmostEqual(total / n, 1.0, delta=0.05)

    def test_mixture_weight_normalizes(self):
        sc = make_scenario(
            MixtureArrival(0.5, 3.0, 0.1), DeterministicService(1))
        rng = Xoshiro256(55)
        n = 40_000
        total = 0.0
        for _ in range(n):
            res = run_one_path(sc, rng, tilt=2.0)
            total += math.exp(res["log_weight"])
        self.assertAlmostEqual(total / n, 1.0, delta=0.05)


class TestEstimatorsVsExact(unittest.TestCase):
    def _run_both(self, sc, n=12000, tilt=2.0, seed=20261125):
        crude = MethodState("crude", 1.0,
                            Xoshiro256(derive_seed(seed, "c")), WeightAccumulator())
        ist = MethodState("is", tilt,
                          Xoshiro256(derive_seed(seed, "i")), WeightAccumulator())
        run_shards(sc, crude, n, 2000)
        run_shards(sc, ist, n, 2000)
        return crude, ist

    def test_poisson_estimate_matches_exact(self):
        sc = make_scenario(PoissonArrival(0.7), BernoulliService(0.8),
                           cap=8, horizon=15)
        exact = exact_overflow(sc)
        crude, ist = self._run_both(sc)
        for st in (crude, ist):
            chk = check_estimate(sc, st.acc.estimates(), st.method)
            self.assertTrue(abs(chk["z_deviation"]) < 3.0,
                            msg=f"{st.method} z={chk['z_deviation']}")
            self.assertTrue(chk["ci_covers_exact"])

    def test_mmpp_bursty_matches_exact(self):
        sc = make_scenario(MmppArrival(0.4, 3.5, 0.04, 0.25),
                           BernoulliService(0.95), cap=12, horizon=15)
        crude, ist = self._run_both(sc, tilt=1.5)
        for st in (crude, ist):
            chk = check_estimate(sc, st.acc.estimates(), st.method)
            self.assertTrue(abs(chk["z_deviation"]) < 3.0)
            self.assertTrue(chk["ci_covers_exact"])

    def test_stalling_service_matches_exact(self):
        sc = make_scenario(PoissonArrival(1.0),
                           StallingService(1, 0, 0.05, 0.2),
                           cap=10, horizon=15)
        crude, ist = self._run_both(sc, tilt=1.5)
        for st in (crude, ist):
            chk = check_estimate(sc, st.acc.estimates(), st.method)
            self.assertTrue(abs(chk["z_deviation"]) < 3.0)
            self.assertTrue(chk["ci_covers_exact"])

    def test_is_reduces_variance_in_rare_regime(self):
        sc = make_scenario(PoissonArrival(0.7), BernoulliService(0.9),
                           cap=12, horizon=18)
        exact = exact_overflow(sc)
        self.assertLess(exact["probability"], 0.05)
        crude, ist = self._run_both(sc, n=30000, tilt=1.8)
        # 普通模拟即便命中也很稀少；IS 命中数显著更多且方差更小
        self.assertGreater(ist.acc.hits, crude.acc.hits * 3)
        if crude.acc.hits > 0:
            self.assertGreater(crude.acc.estimates()["variance"]
                               / ist.acc.estimates()["variance"], 3.0)


if __name__ == "__main__":
    unittest.main()
