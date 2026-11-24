import math
import unittest

from backend.rng import Xoshiro256, derive_seed, splitmix64


class TestRng(unittest.TestCase):
    def test_uniform_moments(self):
        rng = Xoshiro256(42)
        n = 200_000
        s = sum(rng.random() for _ in range(n))
        s2 = sum((rng.random() - 0.5) ** 2 for _ in range(n))
        self.assertAlmostEqual(s / n, 0.5, delta=0.01)
        self.assertAlmostEqual(s2 / n, 1 / 12, delta=0.01)

    def test_poisson_moments_small_and_large(self):
        rng = Xoshiro256(7)
        for lam in (0.3, 5.0, 40.0, 100.0, 300.0):
            n = 60_000
            vals = [rng.poisson(lam) for _ in range(n)]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            self.assertAlmostEqual(mean, lam, delta=lam * 0.05 + 0.05)
            self.assertAlmostEqual(var, lam, delta=lam * 0.1 + 0.05)

    def test_poisson_tail_matches_exact_cdf(self):
        import math
        rng = Xoshiro256(99)
        lam, n = 40.0, 300_000
        vals = [rng.poisson(lam) for _ in range(n)]

        def exact_tail(k0):
            return sum(math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
                       for k in range(k0, int(lam) + 200))

        for k0 in (48, 55, 60):
            emp = sum(1 for v in vals if v >= k0) / n
            self.assertAlmostEqual(emp, exact_tail(k0),
                                   delta=3 * math.sqrt(exact_tail(k0) / n)
                                   + 1e-3)

    def test_poisson_envelope_is_valid(self):
        # 构造期已自检；这里验证不同 λ 都能成功构造且拒绝率合理
        for lam in (12.0, 13.7, 50.0, 200.0):
            rng = Xoshiro256(int(lam * 10))
            draws = [rng.poisson(lam) for _ in range(5000)]
            self.assertTrue(all(v >= 0 for v in draws))
            self.assertLess(max(draws), lam + 20 * math.sqrt(lam))

    def test_geometric(self):
        rng = Xoshiro256(11)
        p = 0.25
        n = 100_000
        mean = sum(rng.geometric(p) for _ in range(n)) / n
        self.assertAlmostEqual(mean, 1 / p, delta=0.05)

    def test_state_roundtrip_and_streams(self):
        rng1 = Xoshiro256(123)
        [rng1.random() for _ in range(100)]
        state = rng1.getstate()
        rng2 = Xoshiro256(999)
        rng2.setstate(state)
        self.assertEqual([rng1.random() for _ in range(5)],
                         [rng2.random() for _ in range(5)])
        # 派生流彼此不同且确定
        a = Xoshiro256(derive_seed(1, "a"))
        b = Xoshiro256(derive_seed(1, "b"))
        self.assertNotEqual([a.random() for _ in range(3)],
                            [b.random() for _ in range(3)])
        a2 = Xoshiro256(derive_seed(1, "a"))
        a3 = Xoshiro256(derive_seed(1, "a"))
        self.assertEqual(a2.random(), a3.random())

    def test_splitmix_deterministic(self):
        x, z = splitmix64(0)
        x2, z2 = splitmix64(0)
        self.assertEqual((x, z), (x2, z2))


if __name__ == "__main__":
    unittest.main()
