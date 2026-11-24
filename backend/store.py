"""持久化任务存储与后台工作线程。

持久化内容（JSON，原子写 tmp+rename）：
- 配置、状态、取消标志；
- 每个方法的随机流完整状态、已终止路径的权重累加器（含权重和）、
  收敛快照、费用计量；
- 试运行候选表与最终选定的偏置倍率。

取消语义：取消标志只在分片边界生效；未跑完的路径不记账，已终止路径与
随机流状态保持一致。进程重启后 running 任务转为 interrupted，可继续恢复，
恢复时随机流从断点延续，绝不把未完成路径计为未溢出或已溢出。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict

from .engine import (MethodState, WeightAccumulator, cost_metrics,
                     reliability, run_pilot, run_shards)
from .exact import check_estimate, exact_overflow
from .models import build_scenario
from .presets import PRESETS
from .rng import Xoshiro256, derive_seed

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"


class TaskStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._recover_on_start()

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._lock_guard:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[task_id] = lock
            return lock

    def _paths(self, task_id: str):
        d = os.path.join(self.data_dir, task_id)
        return d, os.path.join(d, "task.json"), os.path.join(d, "cancel.flag")

    # ------------------------------------------------------------ IO
    def _write(self, task_id: str, doc: dict):
        d, path, _ = self._paths(task_id)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "task.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _read(self, task_id: str) -> dict:
        _, path, _ = self._paths(task_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_tasks(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.data_dir):
            return out
        for name in sorted(os.listdir(self.data_dir)):
            try:
                doc = self._read(name)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            out.append(self._summary(doc))
        out.sort(key=lambda d: d["created_at"], reverse=True)
        return out

    def get_task(self, task_id: str) -> dict:
        return self._read(task_id)

    @staticmethod
    def _summary(doc: dict) -> dict:
        return {
            "id": doc["id"], "status": doc["status"],
            "scenario": doc["scenario"], "created_at": doc["created_at"],
            "updated_at": doc.get("updated_at"),
            "error": doc.get("error"),
            "n_total": doc["config"]["n_total"],
        }

    def _recover_on_start(self):
        for name in os.listdir(self.data_dir) if os.path.isdir(self.data_dir) else []:
            try:
                doc = self._read(name)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if doc["status"] == STATUS_RUNNING:
                doc["status"] = STATUS_INTERRUPTED
                doc["updated_at"] = time.time()
                self._write(name, doc)

    # ------------------------------------------------------------ 创建
    def create_task(self, config: dict) -> dict:
        scenario_cfg = config["scenario"]
        preset_id = scenario_cfg.get("preset_id")
        if preset_id:
            preset = PRESETS[preset_id]
            merged = json.loads(json.dumps(preset["config"]))
            overrides = scenario_cfg.get("overrides") or {}
            self._deep_update(merged, overrides)
            scenario_cfg = merged
        scenario = build_scenario(scenario_cfg)  # 校验失败直接抛错

        task_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        preset_defaults = PRESETS.get(preset_id, {}) if preset_id else {}
        doc = {
            "id": task_id,
            "status": STATUS_QUEUED,
            "created_at": time.time(),
            "updated_at": time.time(),
            "scenario": scenario.to_dict(),
            "preset_id": preset_id,
            "config": {
                "n_total": int(config.get("n_total",
                                          preset_defaults.get("default_n", 10000))),
                "pilot_n": int(config.get("pilot_n",
                                          preset_defaults.get("pilot_n", 3000))),
                "shard_size": int(config.get("shard_size", 2000)),
                "tilt_grid": list(config.get(
                    "tilt_grid", preset_defaults.get("tilt_grid", [1.5, 2.0, 3.0]))),
                "manual_tilt": config.get("manual_tilt"),
                "forced_tilt": preset_defaults.get("forced_tilt"),
                "seed": int(config.get("seed", 20261125)),
            },
            "pilot": None,
            "methods": {},
            "results": None,
            "exact": None,
            "error": None,
        }
        self._write(task_id, doc)
        return doc

    @staticmethod
    def _deep_update(base: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                TaskStore._deep_update(base[k], v)
            else:
                base[k] = v

    # ------------------------------------------------------------ 控制
    def start(self, task_id: str):
        with self._lock_for(task_id):
            doc = self._read(task_id)
            if doc["status"] not in (STATUS_QUEUED, STATUS_INTERRUPTED,
                                     STATUS_CANCELLED):
                return
            doc["status"] = STATUS_RUNNING
            self._write(task_id, doc)
        t = threading.Thread(target=self._run, args=(task_id,), daemon=True)
        self._threads[task_id] = t
        t.start()

    def cancel(self, task_id: str):
        d, _, flag = self._paths(task_id)
        os.makedirs(d, exist_ok=True)
        with open(flag, "w", encoding="utf-8") as f:
            f.write(str(time.time()))

    def is_cancelled(self, task_id: str) -> bool:
        return os.path.exists(self._paths(task_id)[2])

    def _clear_cancel(self, task_id: str):
        _, _, flag = self._paths(task_id)
        if os.path.exists(flag):
            os.remove(flag)

    # ------------------------------------------------------------ 主流程
    def _run(self, task_id: str):
        try:
            doc = self._read(task_id)
            scenario = build_scenario(doc["scenario"])
            cfg = doc["config"]
            n_total, shard = cfg["n_total"], cfg["shard_size"]
            seed = cfg["seed"]
            cancelled = lambda: self.is_cancelled(task_id)

            def persist(states: dict, status=None, pilot=None, results=None):
                cur = self._read(task_id)
                cur["methods"] = {k: v.to_dict() for k, v in states.items()}
                cur["updated_at"] = time.time()
                if status:
                    cur["status"] = status
                if pilot is not None:
                    cur["pilot"] = pilot
                if results is not None:
                    cur["results"] = results
                self._write(task_id, cur)

            states: dict[str, MethodState] = {}
            for key, md in doc["methods"].items():  # 断点恢复
                states[key] = MethodState.from_dict(md)

            # 1) 自适应试运行（仅 IS 需要；独立随机流，结果不入正式估计）
            if "is" not in states:
                if cfg.get("manual_tilt"):
                    chosen = float(cfg["manual_tilt"])
                    pilot = {"grid": [], "chosen_tilt": chosen,
                             "inconclusive": False, "n_pilot": 0,
                             "note": "用户手动指定偏置倍率，跳过试运行"}
                elif cfg.get("forced_tilt"):
                    chosen = float(cfg["forced_tilt"])
                    pilot = {"grid": [], "chosen_tilt": chosen,
                             "inconclusive": False, "n_pilot": cfg["pilot_n"],
                             "note": "教学演示强制偏置（用于展示权重塌缩），"
                                     "试运行仍执行但不改变选择"}
                    pilot_full = run_pilot(scenario, cfg["pilot_n"],
                                           cfg["tilt_grid"], seed, shard)
                    pilot["grid"] = pilot_full["grid"]
                else:
                    pilot = run_pilot(scenario, cfg["pilot_n"],
                                      cfg["tilt_grid"], seed, shard)
                    chosen = pilot["chosen_tilt"]
                doc["pilot"] = pilot
                doc["methods"]["is"] = MethodState(
                    "is", chosen,
                    Xoshiro256(derive_seed(seed, "formal-is")),
                    WeightAccumulator()).to_dict()
                self._write(task_id, doc)
                states["is"] = MethodState.from_dict(doc["methods"]["is"])
            else:
                chosen = states["is"].tilt

            # 2) 普通模拟（独立随机流）
            if "crude" not in states:
                states["crude"] = MethodState(
                    "crude", 1.0,
                    Xoshiro256(derive_seed(seed, "formal-crude")),
                    WeightAccumulator())

            order = ["crude", "is"]
            for key in order:
                st = states[key]
                run_shards(scenario, st, n_total, shard,
                           is_cancelled=cancelled,
                           on_shard=lambda _s, st2=states: persist(st2))
                if cancelled():
                    persist(states, status=STATUS_CANCELLED)
                    self._clear_cancel(task_id)
                    return

            # 3) 汇总结果与可靠性
            results = self._build_results(doc, states, scenario)
            persist(states, status=STATUS_DONE, pilot=doc.get("pilot"),
                    results=results)
        except Exception as exc:  # noqa: BLE001 - 任务级失败需要落盘
            doc = self._read(task_id)
            doc["status"] = STATUS_FAILED
            doc["error"] = repr(exc)
            doc["updated_at"] = time.time()
            self._write(task_id, doc)

    def _build_results(self, doc: dict, states: dict, scenario) -> dict:
        results = {"methods": {}}
        for key, st in states.items():
            est = st.acc.estimates()
            rel = reliability(est, key)
            cost = cost_metrics(st)
            results["methods"][key] = {
                "tilt": st.tilt,
                "estimates": est,
                "reliability": rel,
                "cost": cost,
                "convergence": st.convergence,
                "traces": st.acc.traces,
                "hit_logw_sample": st.acc.hit_logw,
            }
        # 精确核对（小状态场景）
        try:
            exact = exact_overflow(scenario)
            results["exact"] = exact
            for key, st in states.items():
                est = st.acc.estimates()
                if est["n"]:
                    results["methods"][key]["exact_check"] = check_estimate(
                        scenario, est, key)
        except ValueError as exc:
            results["exact"] = {"available": False, "reason": str(exc)}

        # 两方法对照指标
        c = results["methods"]["crude"]
        i = results["methods"]["is"]
        ce, ie = c["estimates"], i["estimates"]
        if ce["hits"] and ie["hits"] and ce["se"] > 0 and ie["se"] > 0:
            results["comparison"] = {
                "variance_reduction": ce["variance"] / ie["variance"],
                "se_reduction": ce["se"] / ie["se"],
                "time_ratio_crude_over_is": c["cost"]["elapsed_seconds"]
                / i["cost"]["elapsed_seconds"]
                if i["cost"]["elapsed_seconds"] > 0 else None,
                "fom_ratio_is_over_crude_time": (
                    (i["cost"]["fom_time"] / c["cost"]["fom_time"])
                    if i["cost"]["fom_time"] and c["cost"]["fom_time"] else None),
            }
        else:
            results["comparison"] = None
        results["rule_of_three_bound"] = 3.0 / ce["n"]
        results["disclaimer"] = (
            "结果基于虚构到达流与服务过程的教学模拟，仅用于方法对照，"
            "不代表任何真实系统，也不构成容量或可靠性保证。")
        return results
