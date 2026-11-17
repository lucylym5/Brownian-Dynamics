"""从已有 configuration.log 绘制 xy 构象：一条链一张图，五个应变时刻。

两端珠子均为红色圆点，中间珠子为蓝色圆点，灰线按珠子编号连接。
默认展示 Hencky 应变约为 1、2、3、4、5 的真实保存帧，不插值。
坐标由 Å 换算为 μm；每帧平移到自身质心，保留原有朝向和形状。
所有图片使用相同坐标范围，x、y 长度比例为 1:1。

在“布朗动力学”目录运行，例如：
    python postprocess_conformations_xy.py results/run_20261116_135426_715316 --wi 2 --lambda-h 3.9447279486012916

也可以直接指定应变率（单位 s^-1），不必提供 Wi：
    python postprocess_conformations_xy.py results/my_run --strain-rate 0.5070058128366377

可选参数：
    --strains 1 2 3 4 5       需要展示的应变
    --chains 1 25 100        只画指定的全局链编号；默认画全部链
    --output results/my_xy   指定输出目录；默认是运行目录/conformations_xy

lambda-h 的单位是秒，对应该次运行设置 Wi 时采用的参考时间。
程序不读取当前 parameters.py，也不会启动新的模拟。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


_HEADER = re.compile(r"# step=(\d+) time_ps=(\S+) chains=(\d+) beads=(\d+)\s*")
END_COLOR = "#D62728"
INTERIOR_COLOR = "#2474B5"


def _frame_times(batch: Path) -> tuple[np.ndarray, np.ndarray]:
    """先读很小的伸长量日志，确定所需坐标帧的位置。"""
    path = batch / "extension.tsv"
    with path.open("r", encoding="utf-8-sig") as stream:
        if stream.readline().strip() != "step\ttime_ps\tmean_span_x":
            raise ValueError(f"{path}: 列名不符合现有日志格式。")
        records = np.loadtxt(stream, delimiter="\t", ndmin=2)
    if records.ndim != 2 or records.shape[1] != 3 or len(records) < 2:
        raise ValueError(f"{path}: 至少需要两帧完整记录。")
    if not np.isfinite(records).all():
        raise ValueError(f"{path}: 存在非有限值。")
    steps, times = records[:, 0], records[:, 1]
    if (np.any(steps != np.floor(steps)) or steps[0] != 0 or times[0] != 0
            or np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError(f"{path}: 步数和时间须从 0 开始并严格递增。")
    return steps.astype(np.int64), times


def _nearest_frames(actual: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """取最近的真实帧；只允许端点外不超过半个保存间隔的目标。"""
    lower = actual[0] - (actual[1] - actual[0]) / 2
    upper = actual[-1] + (actual[-1] - actual[-2]) / 2
    tolerance = 1e-12 * max(1.0, abs(actual[-1]))
    if np.any(targets < lower - tolerance) or np.any(targets > upper + tolerance):
        raise ValueError(
            f"目标应变超出日志范围 [{actual[0]:.6g}, {actual[-1]:.6g}]；"
            "仅允许不超过端点半个保存间隔的偏差。"
        )
    # argmin 在等距时选较早的帧；不插值坐标。
    return np.asarray([np.argmin(np.abs(actual - target)) for target in targets])


def _read_selected(
    batch: Path, steps: np.ndarray, times_ps: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, dict]:
    """顺序扫描日志，只保留选中帧的坐标，避免把整段轨迹放入内存。"""
    path = batch / "configuration.log"
    before = path.stat()
    selected = {}
    wanted = set(indices.tolist())
    shape = None
    with path.open("r", encoding="utf-8-sig") as stream:
        for frame in range(int(indices.max()) + 1):
            line = stream.readline()
            while line and not line.strip():
                line = stream.readline()
            match = _HEADER.fullmatch(line.strip())
            if match is None:
                raise ValueError(f"{path}: 第 {frame} 帧缺失或帧头无法识别。")
            step, time_ps, chains, beads = match.groups()
            step, time_ps = int(step), float(time_ps)
            chains, beads = int(chains), int(beads)
            if (step != steps[frame] or not np.isfinite(time_ps)
                    or not np.isclose(time_ps, times_ps[frame], rtol=1e-12, atol=0)):
                raise ValueError(f"{path}: 第 {frame} 帧与 extension.tsv 时间不一致。")
            if chains < 1 or beads < 2:
                raise ValueError(f"{path}: 链数或珠子数无效。")
            if shape is None:
                shape = (chains, beads)
                expected_ids = np.column_stack((
                    np.repeat(np.arange(1, chains + 1), beads),
                    np.tile(np.arange(1, beads + 1), chains),
                ))
            elif shape != (chains, beads):
                raise ValueError(f"{path}: 各帧的链数或珠子数改变。")
            if stream.readline().strip() != "# chain bead x_A y_A z_A":
                raise ValueError(f"{path}: 第 {frame} 帧坐标列名不正确。")
            block = list(islice(stream, chains * beads))
            if len(block) != chains * beads:
                raise ValueError(f"{path}: 第 {frame} 帧坐标不完整。")
            if frame not in wanted:
                continue
            values = np.fromstring("".join(block), dtype=float, sep=" ")
            if values.size != chains * beads * 5:
                raise ValueError(f"{path}: 第 {frame} 帧坐标列数不正确。")
            values = values.reshape(chains * beads, 5)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: 第 {frame} 帧含 NaN 或无穷大。")
            if not np.array_equal(values[:, :2], expected_ids):
                raise ValueError(f"{path}: 第 {frame} 帧链/珠子编号不正确。")
            selected[frame] = values[:, 2:].reshape(chains, beads, 3).copy()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{path}: 读取期间日志发生变化，请等运行结束后再处理。")
    return np.stack([selected[int(index)] for index in indices]), {
        "batch": batch.name,
        "coordinate_log": str(path),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "chains": shape[0],
        "beads": shape[1],
    }


def _plot_chain(
    xy_um: np.ndarray, targets: np.ndarray, actual: np.ndarray,
    times_s: np.ndarray, chain_id: int, batch: str, local_id: int,
    half_ranges: np.ndarray, output: Path, wi: float | None, dpi: int,
) -> None:
    """五个时刻纵向排列；相同坐标范围和等比例坐标保证可比较。"""
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = Figure(figsize=(11.5, 1.9 * len(targets) + 1.05))
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(targets), 1, squeeze=False)[:, 0]
        for index, ax in enumerate(axes):
            chain = xy_um[index]
            ax.plot(chain[:, 0], chain[:, 1], color="#747B84", lw=1.0, zorder=1)
            ax.scatter(chain[1:-1, 0], chain[1:-1, 1], marker="o", s=25,
                       color=INTERIOR_COLOR, edgecolors="white", linewidths=0.3,
                       zorder=3)
            ax.scatter(chain[[0, -1], 0], chain[[0, -1], 1], marker="o", s=42,
                       color=END_COLOR, edgecolors="white", linewidths=0.5,
                       zorder=4)
            ax.set_xlim(-half_ranges[0], half_ranges[0])
            ax.set_ylim(-half_ranges[1], half_ranges[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x-x_{\mathrm{cm}}$ ($\mu$m)", labelpad=2)
            ax.set_ylabel(r"$y-y_{\mathrm{cm}}$ ($\mu$m)", labelpad=4)
            ax.set_title(
                rf"$\epsilon_H$ = {actual[index]:.4f} (target {targets[index]:g})"
                f"    |    t = {times_s[index]:.5g} s"
                f"    |    x span = {np.ptp(chain[:, 0]):.3f} "
                r"$\mu$m",
                fontsize=10, loc="left", pad=5,
            )
            ax.xaxis.set_major_locator(MaxNLocator(7))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.grid(color="#D9DEE4", linewidth=0.5, alpha=0.75)
            ax.set_axisbelow(True)
        prefix = "" if wi is None else f"Wi = {wi:g}  |  "
        fig.suptitle(f"{prefix}Chain {chain_id:03d}  |  {batch}, local chain {local_id}",
                     fontsize=14, y=0.981)
        handles = [
            Line2D([], [], marker="o", linestyle="", color=END_COLOR,
                   label="End beads (1 and N)", markersize=6),
            Line2D([], [], marker="o", linestyle="", color=INTERIOR_COLOR,
                   label="Interior beads", markersize=5),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.956),
                   ncol=2, frameon=False)
        fig.text(0.5, 0.016,
                 "xy projection; centered per frame; equal x/y scale; "
                 "identical axis limits across all figures.",
                 ha="center", fontsize=9, color="#555555")
        fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.065, hspace=0.65)
        fig.savefig(output, dpi=dpi, facecolor="white")
        fig.clear()


def plot_run_conformations(
    run_dir: str | Path, *, strain_rate: float, wi: float | None = None,
    strains=(1, 2, 3, 4, 5), output_dir: str | Path | None = None,
    chain_ids=None, dpi: int = 180,
) -> dict:
    """应变率单位为 s^-1；全局链号按批次名称顺序连接，从 1 开始。"""
    root = Path(run_dir).expanduser().resolve()
    targets = np.asarray(strains, dtype=float)
    if (not np.isfinite(strain_rate) or strain_rate <= 0 or targets.ndim != 1
            or len(targets) == 0 or not np.isfinite(targets).all()
            or np.any(targets < 0) or not isinstance(dpi, int) or dpi < 1):
        raise ValueError("需要正的有限应变率、非负有限应变列表及正整数 dpi。")
    if wi is not None and (not np.isfinite(wi) or wi <= 0):
        raise ValueError("Wi 标签须为正的有限值。")
    batches = sorted(path for path in root.glob("batch_*") if path.is_dir())
    if not batches and (root / "configuration.log").is_file():
        batches = [root]
    if not batches:
        raise FileNotFoundError(f"{root}: 找不到 batch_*/configuration.log。")

    all_positions, reports, batch_names, local_ids = [], [], [], []
    steps = times_ps = indices = None
    for batch in batches:
        current_steps, current_times = _frame_times(batch)
        if steps is None:
            steps, times_ps = current_steps, current_times
            indices = _nearest_frames(times_ps * 1e-12 * strain_rate, targets)
        elif (not np.array_equal(current_steps, steps)
              or not np.allclose(current_times, times_ps, rtol=1e-12, atol=0)):
            raise ValueError(f"{batch}: 各批次记录的步数和时间不一致。")
        print(f"读取 {batch.name} 的 {len(targets)} 个构象时刻 ...", flush=True)
        positions, report = _read_selected(batch, steps, times_ps, indices)
        if all_positions and positions.shape[2:] != all_positions[0].shape[2:]:
            raise ValueError(f"{batch}: 各批次珠子数不一致。")
        all_positions.append(positions)
        reports.append(report)
        batch_names.extend([batch.name] * positions.shape[1])
        local_ids.extend(range(1, positions.shape[1] + 1))

    positions = np.concatenate(all_positions, axis=1)
    count = positions.shape[1]
    ids = list(range(1, count + 1)) if chain_ids is None else list(chain_ids)
    if (not ids or any(not isinstance(i, (int, np.integer)) or i < 1 or i > count
                       for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError(f"链号须是 1 到 {count} 内的不重复整数。")
    centered_xy = (positions[..., :2]
                   - positions[..., :2].mean(axis=2, keepdims=True)) * 1e-4
    half_ranges = np.maximum(np.abs(centered_xy).max(axis=(0, 1, 2)) * 1.08, 0.01)
    selected_times = times_ps[indices] * 1e-12
    actual = selected_times * strain_rate
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "conformations_xy"
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    for number, chain_id in enumerate(ids, start=1):
        filename = f"chain_{chain_id:03d}_xy.png"
        _plot_chain(centered_xy[:, chain_id - 1], targets, actual, selected_times,
                    chain_id, batch_names[chain_id - 1], local_ids[chain_id - 1],
                    half_ranges, output / filename, wi, dpi)
        figures.append({"global_chain_id": int(chain_id),
                        "batch": batch_names[chain_id - 1],
                        "chain_id_within_batch": local_ids[chain_id - 1],
                        "figure": filename})
        if number % 10 == 0 or number == len(ids):
            print(f"已绘制 {number}/{len(ids)} 条链。", flush=True)

    # 保留原始选中帧与应变定位，便于复查，不改变生产日志。
    np.savez_compressed(
        output / "selected_snapshots.npz", positions_A=positions,
        target_strain=targets, actual_strain=actual, time_s=selected_times,
        steps=steps[indices], global_chain_id=np.arange(1, count + 1),
        batch_names=np.asarray(batch_names), chain_id_within_batch=np.asarray(local_ids),
    )
    summary = {
        "source_run": str(root), "created_utc": datetime.now(timezone.utc).isoformat(),
        "strain_definition": "Hencky strain = strain_rate_per_s * time_s",
        "strain_rate_per_s": float(strain_rate), "wi_label": wi,
        "target_strains": targets.tolist(), "actual_strains": actual.tolist(),
        "selected_steps": steps[indices].tolist(), "selected_times_s": selected_times.tolist(),
        "chains_in_source": count, "beads_per_chain": int(positions.shape[2]),
        "figures_created": len(figures), "coordinate_unit": "micrometre",
        "centering": "subtract each chain's center of mass at each selected frame",
        "equal_xy_scale": True, "shared_half_ranges_um": half_ranges.tolist(),
        "interpolation": False, "end_bead_color": END_COLOR,
        "interior_bead_color": INTERIOR_COLOR, "sources": reports, "figures": figures,
    }
    (output / "conformations_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"实际应变：{', '.join(f'{value:.6f}' for value in actual)}", flush=True)
    print(f"完成，图片目录：{output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="含 batch_* 日志的运行目录，也支持单批次目录")
    rate_group = parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument("--strain-rate", type=float, help="该次运行的应变率，单位 s^-1")
    rate_group.add_argument("--wi", type=float, help="该次运行设置的 Wi，需同时提供 --lambda-h")
    parser.add_argument("--lambda-h", type=float, help="该次运行定义 Wi 的参考松弛时间，单位秒")
    parser.add_argument("--strains", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--chains", type=int, nargs="+", help="只画这些全局链号；默认全部")
    parser.add_argument("--output", type=Path, help="图片和定位记录的输出目录")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.wi is not None:
        if (not np.isfinite(args.wi) or args.wi <= 0 or args.lambda_h is None
                or not np.isfinite(args.lambda_h) or args.lambda_h <= 0):
            parser.error("--wi 需要正的有限值，并须提供正的 --lambda-h（秒）。")
        rate = args.wi / args.lambda_h
    else:
        if args.lambda_h is not None:
            parser.error("--lambda-h 应与 --wi 同时使用。")
        rate = args.strain_rate
    plot_run_conformations(
        args.run_dir, strain_rate=rate, wi=args.wi, strains=args.strains,
        output_dir=args.output, chain_ids=args.chains, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
