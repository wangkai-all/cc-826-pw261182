#!/usr/bin/env python3
"""命令行入口：python -m backend.cli presets|run|exact|serve。"""

from __future__ import annotations

import argparse
import json
import sys

from .engine import MethodState, WeightAccumulator, reliability, run_pilot, run_shards
from .exact import check_estimate, exact_overflow
from .models import build_scenario
from .presets import PRESETS, preset_scenario
from .rng import Xoshiro256, derive_seed


def cmd_presets(_args):
    for pid, p in PRESETS.items():
        print(f"{pid:16s} {p['label']}")


def cmd_run(args):
    if args.preset:
        preset = PRESETS[args.preset]
        cfg = json.loads(json.dumps(preset["config"]))
        scenario = build_scenario(cfg)
        grid = args.tilt_grid or preset["tilt_grid"]
        pilot_n = args.pilot_n or preset["pilot_n"]
    else:
        with open(args.scenario_file, "r", encoding="utf-8") as f:
            scenario = build_scenario(json.load(f))
        grid = args.tilt_grid or [1.5, 2.0, 3.0]
        pilot_n = args.pilot_n or 3000

    seed = args.seed
    print(f"场景: {scenario.name}  容量={scenario.capacity}  时域={scenario.horizon}")
    if args.tilt:
        chosen = args.tilt
        pilot_info = {"chosen_tilt": chosen, "grid": [], "inconclusive": False}
        print(f"手动指定偏置倍率 r={chosen}，跳过试运行")
    else:
        print(f"自适应试运行（独立随机流，n={pilot_n}）...")
        pilot_info = run_pilot(scenario, pilot_n, grid, seed, args.shard)
        chosen = pilot_info["chosen_tilt"]
        for row in pilot_info["grid"]:
            re_ = "NA" if row["rel_error"] is None else f"{row['rel_error']:.3f}"
            print(f"  r={row['tilt']:<4.2f} 命中={row['hits']:<6d} "
                  f"p̂={row['estimate']:.3e} 相对误差={re_} "
                  f"ESS占比={row['ess_all_ratio']:.3e}")
        print(f"选定偏置倍率 r={chosen}（正式阶段固定）")

    crude = MethodState("crude", 1.0,
                        Xoshiro256(derive_seed(seed, "formal-crude")),
                        WeightAccumulator())
    isst = MethodState("is", chosen,
                       Xoshiro256(derive_seed(seed, "formal-is")),
                       WeightAccumulator())
    run_shards(scenario, crude, args.n, args.shard)
    run_shards(scenario, isst, args.n, args.shard)

    for st in (crude, isst):
        est = st.acc.estimates()
        rel = reliability(est, st.method)
        print(f"\n[{st.method.upper()}] tilt={st.tilt}")
        if est["hits"]:
            print(f"  p̂={est['estimate']:.4e}  SE={est['se']:.4e} "
                  f"95%CI=[{est['ci_low']:.4e}, {est['ci_high']:.4e}]")
            print(f"  命中={est['hits']}/{est['n']}  "
                  f"ESS(全体)={est['ess_all']:.1f} ({est['ess_all_ratio']:.2e}) "
                  f"ESS(命中内)={est['ess_hits']:.1f}")
        else:
            print(f"  零命中！仅能给经验上界 3/n = {3.0 / est['n']:.2e}")
        print(f"  可靠={rel['reliable']} {('；'.join(rel['reasons']) or '')}")
        print(f"  费用: 时隙={est['slots']} RNG调用={st.rng.calls} "
              f"耗时={st.elapsed:.3f}s")

    try:
        exact = exact_overflow(scenario)
        print(f"\n精确首次溢出概率 = {exact['probability']:.6e} "
              f"(状态数={exact['state_count']}, 截断行和误差="
              f"{exact['row_sum_max_error']:.2e})")
        for st in (crude, isst):
            chk = check_estimate(scenario, st.acc.estimates(), st.method)
            print(f"  {st.method}: 绝对误差={chk['absolute_error']:.3e} "
                  f"z偏差={chk['z_deviation']:.2f} CI覆盖精确值={chk['ci_covers_exact']}")
    except ValueError as exc:
        print(f"\n精确核对跳过: {exc}")

    print("\n声明：基于虚构模型的教学模拟，不代表真实系统，不构成容量保证。")


def cmd_exact(args):
    with open(args.scenario_file, "r", encoding="utf-8") as f:
        scenario = build_scenario(json.load(f))
    res = exact_overflow(scenario, tilt=args.tilt)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_serve(args):
    from .server import main
    main(args.host, args.port)


def main(argv=None):
    p = argparse.ArgumentParser(description="罕见事件首次溢出概率分析")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets").set_defaults(func=cmd_presets)

    pr = sub.add_parser("run", help="试运行+正式估计对照")
    pr.add_argument("--preset")
    pr.add_argument("--scenario-file")
    pr.add_argument("--n", type=int, default=10000)
    pr.add_argument("--pilot-n", type=int)
    pr.add_argument("--shard", type=int, default=2000)
    pr.add_argument("--seed", type=int, default=20261125)
    pr.add_argument("--tilt", type=float)
    pr.add_argument("--tilt-grid", type=float, nargs="+")
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("exact", help="精确转移计算")
    pe.add_argument("scenario_file")
    pe.add_argument("--tilt", type=float, default=1.0)
    pe.set_defaults(func=cmd_exact)

    ps = sub.add_parser("serve", help="启动 Web 应用")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    ps.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
