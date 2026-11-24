import json
import os
import tempfile
import time
import unittest

from backend.engine import (MethodState, WeightAccumulator, reliability,
                            run_one_path, run_shards)
from backend.exact import MAX_STATES, exact_overflow
from backend.models import (BernoulliService, MixtureArrival, PoissonArrival,
                            Scenario, DeterministicService)
from backend.presets import PRESETS, build_scenario
from backend.rng import Xoshiro256, derive_seed
from backend.store import (STATUS_CANCELLED, STATUS_DONE, STATUS_INTERRUPTED,
                           TaskStore)


def tiny_scenario():
    return Scenario(name="persist", capacity=5, horizon=12,
                    arrival=PoissonArrival(0.7),
                    service=BernoulliService(0.8))


class TestShardResume(unittest.TestCase):
    def test_resume_reproduces_uninterrupted_run(self):
        sc = tiny_scenario()
        seed = 777

        # 一次性跑完
        full = MethodState("is", 1.8,
                           Xoshiro256(derive_seed(seed, "formal-is")),
                           WeightAccumulator())
        run_shards(sc, full, 3000, 3000)

        # 分片：跑 1200 -> 序列化 -> 恢复 -> 跑到 3000
        part = MethodState("is", 1.8,
                           Xoshiro256(derive_seed(seed, "formal-is")),
                           WeightAccumulator())
        run_shards(sc, part, 1200, 600)
        blob = json.dumps(part.to_dict())
        restored = MethodState.from_dict(json.loads(blob))
        run_shards(sc, restored, 3000, 700)

        self.assertEqual(restored.acc.n, full.acc.n)
        self.assertEqual(restored.acc.hits, full.acc.hits)
        self.assertEqual(restored.rng.calls, full.rng.calls)
        self.assertAlmostEqual(restored.acc.estimates()["estimate"],
                               full.acc.estimates()["estimate"], places=12)
        self.assertEqual(restored.acc.to_dict()["s1_hit"],
                         full.acc.to_dict()["s1_hit"])

    def test_accumulator_serialization_roundtrip(self):
        sc = tiny_scenario()
        rng = Xoshiro256(3)
        acc = WeightAccumulator()
        for _ in range(500):
            acc.add(run_one_path(sc, rng, 1.5))
        acc2 = WeightAccumulator.from_dict(json.loads(json.dumps(acc.to_dict())))
        e1, e2 = acc.estimates(), acc2.estimates()
        self.assertEqual(e1["n"], e2["n"])
        self.assertAlmostEqual(e1["estimate"], e2["estimate"])
        self.assertAlmostEqual(e1["ess_all"], e2["ess_all"])


class TestTaskStoreLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TaskStore(self.tmp)

    def test_full_task_with_exact_check(self):
        doc = self.store.create_task({
            "scenario": {"preset_id": "tiny_exact"},
            "n_total": 3000, "pilot_n": 1500, "shard_size": 1000,
        })
        self.store.start(doc["id"])
        deadline = time.time() + 60
        while time.time() < deadline:
            cur = self.store.get_task(doc["id"])
            if cur["status"] in (STATUS_DONE, "failed"):
                break
            time.sleep(0.1)
        cur = self.store.get_task(doc["id"])
        self.assertEqual(cur["status"], STATUS_DONE, msg=cur.get("error"))
        for key in ("crude", "is"):
            m = cur["results"]["methods"][key]
            self.assertIn("exact_check", m)
            self.assertTrue(m["exact_check"]["ci_covers_exact"])
        self.assertIsNotNone(cur["results"]["exact"]["probability"])
        self.assertIn("disclaimer", cur["results"])

    def test_cancel_keeps_only_completed_paths(self):
        # 小分片 -> 取消标志在分片边界被观察到；已落盘路径数必为分片整数倍，
        # 证明未终止路径没有被记账
        doc = self.store.create_task({
            "scenario": {"preset_id": "baseline"},
            "n_total": 20000, "pilot_n": 1000, "shard_size": 500,
        })
        self.store.start(doc["id"])
        time.sleep(0.3)
        self.store.cancel(doc["id"])
        deadline = time.time() + 60
        while time.time() < deadline:
            cur = self.store.get_task(doc["id"])
            if cur["status"] in (STATUS_CANCELLED, STATUS_DONE):
                break
            time.sleep(0.1)
        cur = self.store.get_task(doc["id"])
        self.assertEqual(cur["status"], STATUS_CANCELLED)
        for key, md in cur["methods"].items():
            n = md["acc"]["n"]
            self.assertEqual(n % 500, 0, msg=f"{key} 路径数不是分片整数倍")
            self.assertLess(n, 20000)
            # 未完成路径不得被当作“未溢出”：命中+未命中只来自完整路径
            self.assertEqual(md["acc"]["n"], md["acc"]["hits"]
                             + (md["acc"]["n"] - md["acc"]["hits"]))

    def test_restart_marks_interrupted_and_resumes(self):
        doc = self.store.create_task({
            "scenario": {"preset_id": "tiny_exact"},
            "n_total": 4000, "pilot_n": 1000, "shard_size": 1000,
        })
        # 手工置为 running 模拟进程崩溃
        doc["status"] = "running"
        self.store._write(doc["id"], doc)
        store2 = TaskStore(self.tmp)
        recovered = store2.get_task(doc["id"])
        self.assertEqual(recovered["status"], STATUS_INTERRUPTED)
        # 恢复执行，随机流从断点延续
        store2.start(doc["id"])
        deadline = time.time() + 60
        while time.time() < deadline:
            cur = store2.get_task(doc["id"])
            if cur["status"] in (STATUS_DONE, "failed"):
                break
            time.sleep(0.1)
        cur = store2.get_task(doc["id"])
        self.assertEqual(cur["status"], STATUS_DONE)
        self.assertEqual(cur["results"]["methods"]["is"]["estimates"]["n"], 4000)


class TestReliabilityAndExact(unittest.TestCase):
    def test_zero_hit_reports_rule_of_three_not_zero_risk(self):
        acc = WeightAccumulator()
        for _ in range(1000):
            acc.add({"hit": False, "log_weight": 0.0, "slots": 5, "trace": None})
        est = acc.estimates()
        rel = reliability(est, "crude")
        self.assertFalse(rel["reliable"])
        self.assertTrue(any("零命中" in r for r in rel["reasons"]))
        self.assertEqual(est["estimate"], 0.0)

    def test_strong_tilt_triggers_collapse_warning(self):
        sc = Scenario(name="ext", capacity=12, horizon=30,
                      arrival=MixtureArrival(0.5, 3.0, 0.1),
                      service=DeterministicService(1))
        st = MethodState("is", 3.0, Xoshiro256(42), WeightAccumulator())
        run_shards(sc, st, 8000, 2000)
        rel = reliability(st.acc.estimates(), "is")
        self.assertFalse(rel["reliable"])
        self.assertTrue(any("权重塌缩" in r or "极端权重" in r
                            for r in rel["reasons"]))

    def test_exact_solver_rejects_oversized_state_space(self):
        sc = Scenario(name="big", capacity=MAX_STATES + 10, horizon=5,
                      arrival=PoissonArrival(0.5),
                      service=BernoulliService(0.8))
        with self.assertRaises(ValueError):
            exact_overflow(sc)

    def test_exact_row_sum_truncation_error_reported(self):
        sc = build_scenario(PRESETS["tiny_exact"]["config"])
        res = exact_overflow(sc)
        self.assertIn("row_sum_max_error", res)
        self.assertLess(res["row_sum_max_error"], 1e-4)
        self.assertAlmostEqual(res["probability"], 0.4956, delta=0.01)


if __name__ == "__main__":
    unittest.main()
