"""平面伸长流中的分子个体性：Wi=2，64 条独立的 N=30 FENE 链。

无排除体积（EV）和流体力学相互作用。所有链使用相同参数与应变率，
从无流 FENE 平衡分布独立抽取初始构象，再突然开启流场。
FENE 键长上限为 Q0=sqrt(Q0_square)，整链最大轮廓长度为 (N-1)*Q0。

用理想 Hookean Rouse 链的解析第一模态时间
tau_R=gamma/[4H*sin(pi/(2N))**2] 作参考；沿用哑铃的二阶矩口径，
lambda=tau_R/2，Wi=strain_rate*lambda。tau_R 不是 FENE 链的精确松弛时间。
FENE 平衡初始构象可直接采样；另取独立样本估计共同的平衡平均伸展量 x_eq。

伸展量 x(t)=max_i(r_ix)-min_i(r_ix)，不是端到端距离。
extension.png 将全部 64 条原始伸展量 x(t) 曲线画在同一张图上；
原始 x、供后续分析的 x/x_eq、真实时间和 Hencky 应变均保存，默认推进到应变 6。
不画构象图。trajectories.npz 中只保存累计应变 0、1、2、3、4、5、6
附近积分步的全部链、全部珠子的三维坐标。对应的实际应变和时间也一并保存。

运行（需 NumPy 和 Matplotlib）：
    python ElongationFlow.py
    python ElongationFlow.py --workers 1 --output my_run
输出默认保存在本文件旁的 ElongationalFlow_result1/，非空目录不覆盖。

先从 extension.png 选择典型链，再从 trajectories.npz 的 snapshots 中
按 [应变序号, 链号减一, 珠子序号, xyz] 读取坐标画构象。
此后可在 classification.csv 填一个主路径标签及可选备注：
folded / kinked / dumbbell / half-dumbbell / coiled / other / unclassified。
参考 Smith & Chu (1998), Science 281, 1335, Fig. 1B 和 Fig. 3，比较
应变 2.3 附近及前后的暂态形态；不能只从曲线斜率或末态判断路径。
类别是连续形态的简化；kinked 不等于拓扑结，本模型不保证链段不穿越。
标注后执行（只统计已有结果，不重跑）：
    python ElongationFlow.py --summarize
各类比例以全部 64 条为分母，包含未分类项，不预先编造类别或比例。
全部完成标注且至少辨认三种形态才报告分类自查完成；不保证一次出现三类。

默认 H=gamma=kBT=1，Q0_square=100，dt=1e-4，seed=2026，最多 8 进程。
使用现有半隐式 FENE 积分器；正式物理结论仍需检查减半 dt 的统计。
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

import necklace_chain as nc
import observables as ob
import time_stepping as ts


LABELS = ("folded", "kinked", "dumbbell", "half-dumbbell", "coiled", "other",
          "unclassified")


def advance(r, gradient, config, rng, context):
    """复用半隐式 FENE 积分器，并检查键长约束。"""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            new = ts.semi_implicit_fene_step(
                r, gradient, config["kBT"], config["gamma"], config["H"],
                config["Q0_square"], config["dt"], 0.0, 0.0, rng,
            )
    except (FloatingPointError, RuntimeError) as error:
        raise RuntimeError(f"{context}: FENE 积分失败，请减小 dt 并重跑。") from error
    q2 = np.sum(nc.connection_vector(new)**2, axis=-1)
    if not np.all(np.isfinite(new)) or not np.all(q2 < config["Q0_square"]):
        raise RuntimeError(f"{context}: 非有限位置或 FENE 键长越界；请减小 dt。")
    # 在线性均匀流中移去质心平移，不影响内部构象或跨度。
    return new - new.mean(axis=1, keepdims=True)


def run_batch(start, count, config, initial):
    """每批最多 8 条互不作用的副本；固定随机流不依赖进程完成顺序。"""
    rng = np.random.default_rng(np.random.SeedSequence(
        config["seed"], spawn_key=(1, start),
    ))
    r = np.array(initial, copy=True)
    if r.shape != (count, config["beads"], 3):
        raise ValueError("初始构象的链数或珠子数与配置不符。")
    gradient = np.diag([config["strain_rate"], -config["strain_rate"], 0.0])
    steps = config["flow_steps"]
    every = max(1, int(round(0.01 / (config["strain_rate"] * config["dt"]))))
    snapshot_steps = config["snapshot_steps"]
    snapshot_lookup = {step: index for index, step in enumerate(snapshot_steps)}
    snapshots = np.empty((len(snapshot_steps), count, config["beads"], 3),
                         dtype=np.float64)
    time, extension = [], []

    def record(step):
        time.append(step * config["dt"])
        extension.append(np.ptp(r[..., 0], axis=1))

    snapshots[snapshot_lookup[0]] = r
    record(0)
    for step in range(1, steps + 1):
        r = advance(r, gradient, config, rng, f"flow batch={start} step={step}")
        if step in snapshot_lookup:
            snapshots[snapshot_lookup[step]] = r
        if step % every == 0 or step == steps or step in snapshot_lookup:
            record(step)
    return {
        "start": start, "time": np.asarray(time), "extension": np.asarray(extension),
        "snapshots": snapshots,
    }


def collect_batches(config, executor, initial):
    tasks = [(start, min(8, config["chains"] - start))
             for start in range(0, config["chains"], 8)]
    if executor is None:
        batches = [run_batch(start, count, config, initial[start:start + count])
                   for start, count in tasks]
    else:
        futures = [executor.submit(
            run_batch, start, count, config, initial[start:start + count],
        ) for start, count in tasks]
        batches = []
        for future in as_completed(futures):
            batches.append(future.result())
            print(f'flow: {len(batches)}/{len(tasks)} batches done', flush=True)
    batches.sort(key=lambda batch: batch["start"])
    if not all(np.array_equal(b["time"], batches[0]["time"]) for b in batches):
        raise RuntimeError("各批次采样时刻不同。")
    return batches


def sample_equilibrium(config):
    """从无 EV 的 FENE 平衡分布抽取初始链和独立的长度标尺样本。"""
    def draw(count, spawn_key):
        rng = np.random.default_rng(np.random.SeedSequence(
            config["seed"], spawn_key=spawn_key,
        ))
        return nc.initialize_chain(
            "FENE", count, config["beads"], config["kBT"], config["H"],
            epsilon=0.0, sigma=0.0, rng=rng,
            Q0_square=config["Q0_square"],
        )

    initial = draw(config["chains"], (0,))
    reference = draw(config["equilibrium_samples"], (2,))
    equilibrium_extension = np.ptp(reference[..., 0], axis=1)
    return initial, {
        "equilibrium_extension": equilibrium_extension,
        "x_eq": float(equilibrium_extension.mean()),
        "initial_mean_extension": float(np.ptp(initial[..., 0], axis=1).mean()),
    }


def plot_results(time, extension, config, output, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("turbo")(np.linspace(0, 1, config["chains"]))
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for chain in range(config["chains"]):
        ax.plot(time, extension[:, chain], color=colors[chain], alpha=0.6, lw=0.8)
    ax.plot(time, extension.mean(axis=1), color="black", lw=2, label="Ensemble mean")
    ax.set(xlabel="Time t (simulation units)", ylabel=r"Extension $x(t)$ (simulation length units)",
           title=f'Wi={config["wi"]:g}, N={config["beads"]}, '
                 f'{config["chains"]} independent chains')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "extension.png", dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def summarize(output):
    with np.load(output / "trajectories.npz") as data:
        count = data["extension"].shape[1]
    with (output / "classification.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if sorted(int(row["chain_id"]) for row in rows) != list(range(1, count + 1)):
        raise ValueError("分类表必须包含每条链一次，不能重复或遗漏。")
    counts = dict.fromkeys(LABELS, 0)
    for row in rows:
        label = row["label"].strip() or "unclassified"
        if label not in counts:
            raise ValueError(f'chain {row["chain_id"]}: 未知标签 {label!r}，可用 {LABELS}')
        counts[label] += 1
    distinct = sum(counts[label] > 0 for label in LABELS
                   if label not in ("other", "unclassified"))
    result = {
        "chains": count, "counts": counts,
        "percent_of_all_chains": {label: 100 * n / count for label, n in counts.items()},
        "identified_shape_classes": distinct,
        "manual_check_complete": counts["unclassified"] == 0 and distinct >= 3,
        "note": "根据人工标签计数；不把曲线差异自动视作形态分类，也不以 other 凑三类。",
    }
    write_json(output / "classification_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beads", type=int, default=30)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--wi", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--strain", type=float, default=6.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--kBT", type=float, default=1.0)
    parser.add_argument("--Q0-square", type=float, default=100.0,
                        help="FENE 单根弹簧最大键长的平方")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "ElongationalFlow_result2")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    if not all(np.isfinite(v) and v > 0 for v in (
            args.wi, args.dt, args.H, args.gamma, args.kBT,
            args.Q0_square, args.strain)):
        parser.error("物理参数、Q0-square、dt 和 strain 必须为有限正数。")
    if args.beads < 3 or not 2 <= args.chains <= 64 or args.workers < 1 or args.seed < 0:
        parser.error("需要 beads>=3、2<=chains<=64、workers>=1、seed>=0。")
    if args.dt >= args.gamma / (2 * args.H):
        parser.error("需要 dt<gamma/(2H)；FENE 越界时还需减小 dt。")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("输出目录非空，请用 --output 指定新目录，避免覆盖已有结果。")
    config = {key: value for key, value in vars(args).items()
              if key not in ("output", "show", "summarize")}
    config["workers"] = min(args.workers, (args.chains + 7) // 8)
    config["epsilon_ev"] = 0.0
    config["equilibrium_samples"] = 4096
    config["tau_R_theory"] = float(ob.theoretical_relaxation_time(
        args.beads, args.H, args.gamma,
    ))
    config["lambda"] = config["tau_R_theory"] / 2
    config["strain_rate"] = args.wi / config["lambda"]
    config["max_contour_length"] = (args.beads - 1) * np.sqrt(args.Q0_square)
    nominal_steps = args.strain / (config["strain_rate"] * args.dt)
    rounded_steps = int(round(nominal_steps))
    roundoff_tolerance = 8 * np.finfo(float).eps * max(1.0, nominal_steps)
    config["flow_steps"] = (rounded_steps if abs(nominal_steps - rounded_steps)
                            <= roundoff_tolerance else int(np.ceil(nominal_steps)))
    snapshot_target_strain = np.arange(int(np.floor(args.strain)) + 1, dtype=np.float64)
    snapshot_steps = np.rint(snapshot_target_strain /
                             (config["strain_rate"] * args.dt)).astype(np.int64)
    if len(set(snapshot_steps)) != len(snapshot_steps):
        parser.error("dt 过大，相邻整数应变对应同一积分步；请减小 dt。")
    config["snapshot_target_strain"] = snapshot_target_strain.tolist()
    config["snapshot_steps"] = snapshot_steps.tolist()
    started = perf_counter()
    initial, calibration = sample_equilibrium(config)
    sampling_seconds = perf_counter() - started
    simulation_started = perf_counter()
    config["x_eq"] = calibration["x_eq"]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'FENE no EV: N={args.beads}, M={args.chains}, '
          f'Q0^2={args.Q0_square:g}, dt={args.dt:g}, '
          f'workers={config["workers"]}', flush=True)
    print(f'tau_R(theory)={config["tau_R_theory"]:.6g}, '
          f'lambda=tau_R/2={config["lambda"]:.6g}, '
          f'strain_rate={config["strain_rate"]:.6g}, '
          f'Wi={args.wi:g}, x_eq={config["x_eq"]:.6g}', flush=True)
    print(f'Saving coordinates at target Hencky strains: '
          f'{config["snapshot_target_strain"]}', flush=True)
    executor = (ProcessPoolExecutor(max_workers=config["workers"])
                if config["workers"] > 1 else None)
    try:
        flow = collect_batches(config, executor, initial)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    time = flow[0]["time"]
    extension = np.concatenate([b["extension"] for b in flow], axis=1)
    snapshots = np.concatenate([b["snapshots"] for b in flow], axis=1)
    snapshot_time = snapshot_steps * args.dt
    np.savez_compressed(
        args.output / "trajectories.npz", time=time, strain=time * config["strain_rate"],
        extension=extension, normalized_extension=extension / config["x_eq"],
        x_eq=config["x_eq"], snapshots=snapshots,
        snapshot_target_strain=snapshot_target_strain,
        snapshot_strain=snapshot_time * config["strain_rate"],
        snapshot_time=snapshot_time, snapshot_steps=snapshot_steps,
    )
    np.savez_compressed(
        args.output / "calibration.npz", **calibration,
        tau_R_theory=config["tau_R_theory"], lambda_reference=config["lambda"],
    )
    with (args.output / "classification.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["chain_id", "label", "notes"])
        writer.writerows((chain + 1, "unclassified", "") for chain in range(args.chains))
    simulation_seconds = perf_counter() - simulation_started
    plot_results(time, extension, config, args.output, args.show)
    scalar_calibration = {key: value for key, value in calibration.items()
                          if not isinstance(value, np.ndarray)}
    write_json(args.output / "summary.json", {
        "config": config, "calibration": scalar_calibration,
        "equilibrium_sampling_wall_seconds": sampling_seconds,
        "simulation_wall_seconds": simulation_seconds,
        "total_wall_seconds": perf_counter() - started,
        "extension_records_per_chain": len(time),
        "coordinate_frames_per_chain": len(snapshot_steps),
        "actual_flow_time": float(time[-1]),
        "actual_final_strain": float(time[-1] * config["strain_rate"]),
        "classification_status": "pending_manual_review",
        "notes": [
            "Independent FENE chains without EV or HI; initial bonds are drawn exactly from the no-flow FENE equilibrium distribution.",
            "Flow batches use SeedSequence(seed, spawn_key=(1, start)); changing worker count preserves random streams.",
            "Wi uses half the analytical longest-mode time of the ideal Hookean Rouse chain, not the actual FENE relaxation time.",
            "The equilibrium length scale comes from separate equilibrium FENE samples; compare smaller dt before physical conclusions.",
            "Plots show raw extension x(t); normalized_extension is saved separately using the shared x_eq.",
            "Only integer-Hencky-strain snapshots of all bead positions are stored as float64 in trajectories.npz; actual snapshot strains may differ from targets by up to half a time step. No shape plots are generated.",
            "FENE bonds have a maximum length but without EV the chain can cross itself; kinked is not a topological knot.",
        ],
    })
    print(f'Done in {(perf_counter() - started) / 3600:.3f} h: {args.output.resolve()}', flush=True)
    print("Inspect extension.png and trajectories.npz snapshots; select representative chains later.")


if __name__ == "__main__":
    main()
