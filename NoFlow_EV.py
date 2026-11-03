"""
    Hookean 珠簧链，启用 WCA 排除体积，无流动、无流体动力学相互作用。

    默认扫描 N=20,30,...,60，M=64；先预平衡再记录尺寸和第一模态相关函数。
    默认 dt=5e-5；通过 Numba 加速保留较小步长，WCA 失稳时仍会报错。
    默认 --backend numba，--backend numpy 可运行原 NumPy 实现作对照。
    预平衡默认 3*tau_Hooke，正式采样 3*tau_Hooke；
    相关时间在0.2*tau_Hooke 到 1.5*tau_Hooke 内拟合。

    同一 N 的独立样本按每批 8 条链分配到多个进程，默认最多 8 个进程；
    不同批次使用独立随机流，合并全部样本的模态后统一计算相关函数和拟合 tau。

    两子图分别绘制 log(Rg)-log(N)、log(tau)-log(N)。
    tau_Hooke 仅安排时长、采样和拟合窗口，不是含排除体积链的理论相关时间；
    不绘制解析对比曲线，不用理论值替代未测出的相关时间。
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

import Necklace_chain as nc
import observables as ob
import parameters as p
import time_stepping as ts


def fit_power_law(N, values, theory_slope):
    """拟合 log-log 斜率，偏差定义为拟合斜率减去理论指数。"""
    x = np.log(np.asarray(N, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    coefficient, covariance = np.polyfit(x, y, 1, cov=True)
    return {
        "slope": float(coefficient[0]),
        "intercept": float(coefficient[1]),
        "slope_se": float(np.sqrt(covariance[0, 0])),
        "theory_slope": float(theory_slope),
        "slope_deviation": float(coefficient[0] - theory_slope),
    }


def simulate_chain_batch(N, config, sample_start, sample_count):
    """推进一批独立链，返回逐样本尺寸和未归一化的模态轨迹。"""
    M, dt = sample_count, config["dt"]
    compiled_step = None
    if config.get("backend", "numba") == "numba":
        try:
            from ev_acceleration import hookean_wca_step
        except ImportError as error:
            raise RuntimeError("Numba 未安装，请安装 requirements_ev.txt，或使用 --backend numpy。") from error
        compiled_step = hookean_wca_step
    kBT, H, gamma = config["kBT"], config["H"], config["gamma"]
    epsilon_ev, sigma_ev = config["epsilon_ev"], config["sigma_ev"]
    rng = np.random.default_rng(
        np.random.SeedSequence(config["seed"], spawn_key=(int(N), int(sample_start)))
    )
    r = nc.initialize_chain("Hookean", M, N, kBT, H, epsilon=epsilon_ev, sigma=sigma_ev, rng=rng)
    velocity_gradient = np.zeros((3, 3))
    tau_theory = float(ob.theoretical_relaxation_time(N, H, gamma))
    n_equilibration = int(np.ceil(config["equilibration"] * tau_theory / dt))
    n_steps = int(np.ceil(config["duration"] * tau_theory / dt))
    save_every = max(1, int(round(tau_theory / (config["samples_per_tau"] * dt))))

    def advance(step, phase):
        nonlocal r
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                if compiled_step is None:
                    new_position = ts.euler_maruyama_step(
                        "Hookean", r, velocity_gradient, kBT, gamma, H,
                        1.0, dt, epsilon_ev, sigma_ev, rng,
                    )
                else:
                    noise = rng.standard_normal(size=r.shape)
                    new_position = compiled_step(
                        r, noise, kBT, gamma, H, dt, epsilon_ev, sigma_ev,
                    )
        except (FloatingPointError, ZeroDivisionError, ValueError) as error:
            raise RuntimeError(
                f"N={N}, 样本 {sample_start}-{sample_start + M - 1}, "
                f"{phase} 第 {step} 步出现非有限力或位置，请减小 dt。"
            ) from error
        if (not np.all(np.isfinite(new_position))
                or np.max(np.abs(new_position - r)) > sigma_ev):
            raise RuntimeError(
                f"N={N}, 样本 {sample_start}-{sample_start + M - 1}, "
                f"{phase} 第 {step} 步位置非有限或单坐标位移超过 sigma，"
                "WCA 显式推进未解析近距离排斥，请减小 dt。"
            )
        r = new_position

    # 预平衡过程不进入尺寸平均，也不作为相关函数的时间零点。
    for step in range(1, n_equilibration + 1):
        advance(step, "预平衡")

    time = []
    modes = []
    rg2_sum_per_chain = np.zeros(M)

    def record(step):
        nonlocal rg2_sum_per_chain
        time.append(step * dt)
        modes.append(ob.first_rouse_mode(r))
        rg2, _, _ = ob.obervables(r)
        rg2_sum_per_chain += rg2

    record(0)
    for step in range(1, n_steps + 1):
        advance(step, "正式采样")
        if step % save_every == 0 or step == n_steps:
            record(step)

    time = np.asarray(time)
    rg2_per_chain = rg2_sum_per_chain / len(time)
    return {
        "sample_start": sample_start,
        "chain_count": M,
        "time": time,
        "modes": np.asarray(modes),
        "rg2_per_chain": rg2_per_chain,
        "tau_theory": tau_theory,
        "equilibration_steps": n_equilibration,
        "n_steps": n_steps,
        "save_every": save_every,
    }


def combine_chain_batches(N, batches, config):
    """合并原始模态后统一归一化、拟合，避免直接平均各批次的 tau。"""
    batches = sorted(batches, key=lambda batch: batch["sample_start"])
    first = batches[0]
    M, dt = config["chains"], config["dt"]
    time = first["time"]
    expected_start = 0
    for batch in batches:
        if batch["sample_start"] != expected_start:
            raise ValueError("样本批次有遗漏或重复。")
        expected_start += batch["chain_count"]
        if not np.array_equal(batch["time"], time):
            raise ValueError("样本批次的采样时刻不一致。")
    if expected_start != M:
        raise ValueError("合并后的样本数与 chains 不一致。")
    rg2_per_chain = np.concatenate([batch["rg2_per_chain"] for batch in batches])
    modes = np.concatenate([batch["modes"] for batch in batches], axis=1)
    tau_theory = first["tau_theory"]
    n_equilibration = first["equilibration_steps"]
    n_steps = first["n_steps"]
    save_every = first["save_every"]
    rg2_mean = float(rg2_per_chain.mean())
    rg_rms = float(np.sqrt(rg2_mean))
    rg2_sem = float(rg2_per_chain.std(ddof=1) / np.sqrt(M))
    correlation = ob.mode_correlation(modes)
    fit_start_time = config["fit_start"] * tau_theory
    fit_end_time = config["fit_end"] * tau_theory
    selected = (time >= fit_start_time) & (time <= fit_end_time)
    tau = np.nan
    fit_status = "no_resolved_decay"
    if (np.count_nonzero(selected) >= 3
            and np.all(np.isfinite(correlation[selected]))
            and np.all(correlation[selected] > 0)):
        candidate = ob.fit_relaxation_time(
            time, correlation, fit_start_time, fit_end_time,
        )
        if np.isfinite(candidate) and candidate > 0:
            tau = float(candidate)
            fit_status = "positive_decay_fit"
    return {
        "N": int(N),
        "rg2_mean": rg2_mean,
        "rg2_sem": rg2_sem,
        "rg_rms": rg_rms,
        "rg_rms_sem": rg2_sem / (2 * rg_rms),
        "tau": tau,
        "tau_theory": tau_theory,
        "equilibration_time": float(n_equilibration * dt),
        "equilibration_steps": n_equilibration,
        "total_time": float(time[-1]),
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "time_over_tau_theory": float(time[-1] / tau_theory),
        "n_steps": n_steps,
        "save_every": save_every,
        "n_records": len(time),
        "correlation_at_end": float(correlation[-1]),
        "tau_fit_status": fit_status,
        "time": time,
        "correlation": correlation,
    }


def simulate_chain_length(N, config, pool=None):
    """同一 N 按独立样本分批并行；固定批次使结果不依赖进程完成顺序。"""
    started = perf_counter()
    batch_size = config.get("sample_batch_size", 8)
    tasks = [
        (sample_start, min(batch_size, config["chains"] - sample_start))
        for sample_start in range(0, config["chains"], batch_size)
    ]

    def collect_batches(executor):
        futures = [
            executor.submit(simulate_chain_batch, N, config, sample_start, count)
            for sample_start, count in tasks
        ]
        return [future.result() for future in as_completed(futures)]

    if pool is not None:
        batches = collect_batches(pool)
    elif config.get("workers", 1) == 1:
        batches = [
            simulate_chain_batch(N, config, sample_start, count)
            for sample_start, count in tasks
        ]
    else:
        workers = min(config["workers"], len(tasks), 61 if os.name == "nt" else len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = collect_batches(executor)
    result = combine_chain_batches(N, batches, config)
    result["elapsed_seconds"] = perf_counter() - started
    return result


def plot_results(results, fits, config, output_dir, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([row["N"] for row in results])
    x = np.log(N)
    x_line = np.linspace(x.min(), x.max(), 200)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    panels = [
        ("rg", "rg_rms",
         r"$\log R_g$", "Excluded-volume single-chain size"),
        ("tau", "tau",
         r"$\log\tau_R$", "Longest mode correlation time"),
    ]
    for ax, (key, measured, ylabel, title) in zip(axes, panels):
        values = np.array([row[measured] for row in results])
        valid = np.isfinite(values) & (values > 0)
        ax.plot(x[valid], np.log(values[valid]), "o", label="Simulation")
        fit = fits[key]
        if fit is not None:
            ax.plot(
                x_line, fit["slope"] * x_line + fit["intercept"],
                color="tab:orange", label="Linear fit",
            )
            note = (
                f'Slope = {fit["slope"]:.3f}\n'
                f'$\\Delta$ = {fit["slope_deviation"]:+.3f} '
                f'(theory: {fit["theory_slope"]:.3f})'
            )
        else:
            note = "Insufficient decaying correlations for a fit"
        ax.text(
            0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
        ax.set(xlabel=r"$\log N$", ylabel=ylabel, title=title)
        # 固定的是绘图区的物理长宽比；两幅图的数据单位无需等比例。
        ax.set_box_aspect(3 / 4)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f'Hookean + WCA: M={config["chains"]}, dt={config["dt"]:g}, '
        f'epsilon={config["epsilon_ev"]:g}, sigma={config["sigma_ev"]:g}, '
        f'T={config["duration"]:g} tau_Hooke per N',
        fontsize=12,
    )
    fig.canvas.draw()
    geometry = [
        {"width": float(ax.get_window_extent().width),
         "height": float(ax.get_window_extent().height)}
        for ax in axes
    ]
    fig.savefig(output_dir / "scaling_laws.png", dpi=200)

    fig_c, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for row in results:
        ax.plot(
            row["time"] / row["tau_theory"], row["correlation"],
            label=f'N={row["N"]}',
        )
    ax.axvspan(config["fit_start"], config["fit_end"], color="grey", alpha=0.1)
    ax.set(
        xlabel=r"$t/\tau_{\rm Hooke}$", ylabel=r"$C_1(t)$",
        title="First-mode correlations and fitting window",
        xlim=(0, config["duration"]),
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig_c.savefig(output_dir / "mode_correlations.png", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig_c)
    return geometry


def save_results(results, fits, config, output_dir, geometry):
    fields = [key for key in results[0] if key not in ("time", "correlation")]
    with (output_dir / "scaling_results.csv").open(
        "w", newline="", encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    curves = {}
    for row in results:
        curves[f'time_N{row["N"]}'] = row["time"]
        curves[f'correlation_N{row["N"]}'] = row["correlation"]
    np.savez_compressed(output_dir / "mode_correlations.npz", **curves)
    summary = {
        "config": config,
        "fits": fits,
        "axes_sizes_pixels_at_figure_dpi": geometry,
        "notes": [
            "Hookean springs plus the existing WCA force; independent chain replicas do not interact.",
            "For each N, fixed batches of independent samples use SeedSequence(seed, spawn_key=(N, sample_start)); CPU processes share no mutable chain state.",
            "Raw mode histories are concatenated across all samples before normalizing correlations and fitting tau; batch tau values are never averaged.",
            "Both backends implement the same explicit Euler-Maruyama step without force clipping, potential softening or overlap projection.",
            "The numba backend compiles the Hookean/WCA force and deterministic update; Gaussian noise still comes from the same NumPy Generator streams, with fastmath disabled.",
            "Initialization excludes close bead pairs; this growth distribution is not the WCA equilibrium distribution.",
            "Equilibration precedes all recorded data; verify that longer equilibration and smaller dt give consistent statistics.",
            "Left panel fits log(sqrt(mean(Rg^2))) against log(N); right panel fits log(measured tau) against log(N).",
            "Three-dimensional good-solvent long-chain targets: nu=0.588, tau exponent=1+2nu=2.176; Flory nu=0.6 gives 11/5.",
            "The --nu option changes the comparison exponent only, never the force or the measured data.",
            "slope_deviation = fitted slope - theory_slope; finite chain lengths can show crossover and scaling corrections.",
            "tau_theory is the Hookean reference used only for durations, sampling, fit windows and time normalization; no analytical reference curves are drawn.",
            "Both plot boxes have width:height=4:3 and the same dimensions.",
            "A non-decaying or non-positive correlation fit is omitted, never replaced by theory.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=60)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--duration", type=float, default=3.0, help="总时间 / 胡克参考相关时间")
    parser.add_argument("--samples-per-tau", type=int, default=100)
    parser.add_argument("--fit-start", type=float, default=0.2, help="拟合起点 / 胡克参考相关时间")
    parser.add_argument("--fit-end", type=float, default=1.5, help="拟合终点 / 胡克参考相关时间")
    parser.add_argument("--backend", choices=("numba", "numpy"), default="numba",
                        help="默认用 Numba 编译核，numpy 用于原实现对照")
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--epsilon-ev", type=float, default=1.0, help="WCA 能量尺度")
    parser.add_argument("--sigma-ev", type=float, default=1.0, help="WCA 长度尺度")
    parser.add_argument("--nu", type=float, default=0.588, help="尺寸标度的理论对照指数，不改变动力学")
    parser.add_argument("--equilibration", type=float, default=3.0, help="预平衡时间 / 胡克参考相关时间")
    parser.add_argument("--gamma", type=float, default=p.GAMMA)
    parser.add_argument("--kBT", type=float, default=p.KB_DIMLESS * p.TEMP_DIMLESS)
    parser.add_argument("--seed", type=int, default=p.RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="同一 N 内并行计算独立样本批次的进程数")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "noFlow_EV_results",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not (2 <= args.n_min <= args.n_max and args.n_step > 0):
        parser.error("需要 2 <= n-min <= n-max，且 n-step > 0。")
    bead_numbers = list(range(args.n_min, args.n_max + 1, args.n_step))
    if bead_numbers[-1] != args.n_max:
        bead_numbers.append(args.n_max)
    if len(bead_numbers) < 3:
        parser.error("拟合至少需要三个不同的珠子数。")
    if min(args.H, args.gamma, args.kBT, args.dt, args.duration,
           args.epsilon_ev, args.sigma_ev) <= 0:
        parser.error("物理参数、WCA 参数、dt 和 duration 必须为正。")
    if args.equilibration < 0 or not 0 < args.nu < 1:
        parser.error("需要 equilibration >= 0，且 0 < nu < 1。")
    if not all(np.isfinite(value) for value in (
            args.H, args.gamma, args.kBT, args.dt, args.duration,
            args.epsilon_ev, args.sigma_ev, args.equilibration, args.nu,
            args.fit_start, args.fit_end)):
        parser.error("所有浮点参数必须为有限值。")
    if args.dt >= args.gamma / (2 * args.H):
        parser.error("显式推进需满足弹簧的 dt < gamma/(2H)；WCA 稳定性还需进一步减小 dt。")
    if not (0 <= args.fit_start < args.fit_end <= args.duration):
        parser.error("需要 0 <= fit-start < fit-end <= duration。")
    if (args.fit_end - args.fit_start) * args.samples_per_tau < 2:
        parser.error("拟合窗口至少需要三个记录时刻。")
    if args.chains < 2 or args.workers < 1 or args.samples_per_tau < 10:
        parser.error("需要 chains >= 2、workers >= 1、samples-per-tau >= 10。")
    config = {
        key: value for key, value in vars(args).items()
        if key not in ("output", "show")
    }
    config["bead_numbers"] = bead_numbers
    config["sample_batch_size"] = 8
    n_batches = (args.chains + config["sample_batch_size"] - 1) // config["sample_batch_size"]
    config["workers"] = min(args.workers, n_batches, 61 if os.name == "nt" else n_batches)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f'Hookean + WCA: N={bead_numbers}; M={args.chains}; dt={args.dt}; '
        f'epsilon={args.epsilon_ev:g}; sigma={args.sigma_ev:g}; '
        f'equilibration={args.equilibration:g}*tau_Hooke; '
        f'T={args.duration:g}*tau_Hooke per N; '
        f'fit={args.fit_start:g}..{args.fit_end:g} tau_Hooke; '
        f'workers={config["workers"]}; samples_per_task={config["sample_batch_size"]}; '
        f'backend={config["backend"]}',
        flush=True,
    )
    if config["backend"] == "numba":
        from ev_acceleration import warm_up
        print("Preparing Numba kernels...", flush=True)
        warm_up()
    results = []

    def collect(row):
        results.append(row)
        print(
            f'N={row["N"]:3d}: Rg={row["rg_rms"]:.5g}, '
            f'tau_fit={row["tau"]:.5g}, '
            f'T={row["total_time"]:.5g}, wall={row["elapsed_seconds"]:.1f}s',
            flush=True,
        )

    if config["workers"] == 1:
        for N in bead_numbers:
            print(f'N={N:3d}: starting {args.chains} independent samples', flush=True)
            collect(simulate_chain_length(N, config))
    else:
        # 复用进程池；把 CPU 用于同一 N 的独立样本，完成后立即报告 tau。
        with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            for N in bead_numbers:
                print(f'N={N:3d}: starting {args.chains} independent samples', flush=True)
                collect(simulate_chain_length(N, config, pool))
    results.sort(key=lambda row: row["N"])
    measured_tau = np.array([row["tau"] for row in results])
    valid = np.isfinite(measured_tau) & (measured_tau > 0)
    fits = {
        "rg": fit_power_law(bead_numbers, [row["rg_rms"] for row in results], config["nu"]),
        "tau": (
            fit_power_law(np.array(bead_numbers)[valid], measured_tau[valid], 1 + 2 * config["nu"])
            if valid.sum() >= 3 else None
        ),
    }
    geometry = plot_results(results, fits, config, args.output, args.show)
    save_results(results, fits, config, args.output, geometry)
    print(f'Rg slope: {fits["rg"]["slope"]:.5f}', flush=True)
    if fits["tau"] is not None:
        print(f'tau slope: {fits["tau"]["slope"]:.5f}', flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
