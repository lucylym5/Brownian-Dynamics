"""用已有七帧坐标制作 64 条链的分类视图；不重新运行模拟。

输出四张 16 链速览图，以及每条链的详细图。详细图先画 x-应变曲线，
再逐帧画 x-y、x-z 投影和珠子编号与 x 坐标的关系。投影分别放大，
实际尺寸由坐标刻度和标注的 x 跨度给出。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


DEFAULT_INPUT = (Path(__file__).resolve().parent / "ElongationalFlow_result1"
                 / "trajectories.npz")
FIRST_COLOR = "#d62728"
LAST_COLOR = "#1769c2"


def load_results(path):
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}；请先运行 ElongationFlow.py。")
    with np.load(path, allow_pickle=False) as data:
        keys = ("snapshots", "snapshot_target_strain", "snapshot_strain",
                "strain", "extension")
        missing = set(keys).difference(data.files)
        if missing:
            raise ValueError(f"{path} 缺少字段：{', '.join(sorted(missing))}")
        snapshots, target, actual, strain, extension = (data[key] for key in keys)
    if snapshots.ndim != 4 or snapshots.shape[:2] != (7, 64) \
            or snapshots.shape[-1] != 3 or snapshots.shape[2] < 2:
        raise ValueError("snapshots 应为 (7, 64, N, 3)。")
    if not np.array_equal(target, np.arange(7, dtype=float)):
        raise ValueError("需要目标 Hencky 应变 0、1、2、3、4、5、6 的七帧。")
    if actual.shape != (7,) or strain.ndim != 1 \
            or extension.shape != (len(strain), 64):
        raise ValueError("应变与伸展量数组的形状不匹配。")
    if not all(np.isfinite(array).all() for array in
               (snapshots, actual, strain, extension)):
        raise ValueError("结果中存在非有限坐标或伸展量。")
    return snapshots, target, actual, strain, extension


def padded_limits(values, minimum_width=2.0):
    low, high = float(np.min(values)), float(np.max(values))
    half = max((high - low) * 0.56, minimum_width / 2)
    middle = (low + high) / 2
    return middle - half, middle + half


def projection(ax, chain, second_axis):
    points = chain[:, [0, second_axis]]
    segments = np.stack((points[:-1], points[1:]), axis=1)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(segments)))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.7))
    ax.scatter(*points[0], s=28, color=FIRST_COLOR, zorder=4)
    ax.scatter(*points[-1], s=28, color=LAST_COLOR, zorder=4)
    ax.set_xlim(*padded_limits(points[:, 0]))
    ax.set_ylim(*padded_limits(points[:, 1]))
    ax.grid(alpha=0.22)


def contour_plot(ax, chain):
    beads = np.arange(1, len(chain) + 1)
    ax.plot(beads, chain[:, 0], color="#38566b", linewidth=1.5)
    ax.scatter([beads[0]], [chain[0, 0]], s=28, color=FIRST_COLOR, zorder=4)
    ax.scatter([beads[-1]], [chain[-1, 0]], s=28, color=LAST_COLOR, zorder=4)
    ax.set_xlim(0.5, len(chain) + 0.5)
    ax.set_ylim(*padded_limits(chain[:, 0]))
    ax.grid(alpha=0.22)


def draw_chain_report(snapshots, target, actual, strain, extension, chain_id, output):
    index = chain_id - 1
    fig = plt.figure(figsize=(15, 27))
    grid = fig.add_gridspec(8, 3, height_ratios=[1.2] + [1] * 7,
                            left=0.06, right=0.97, bottom=0.035, top=0.955,
                            wspace=0.24, hspace=0.58)
    curve_ax = fig.add_subplot(grid[0, :])
    curve_ax.plot(strain, extension[:, index], color="#274c68",
                  linewidth=1.5, label=f"Chain {chain_id}")
    x_at_frames = np.ptp(snapshots[:, index, :, 0], axis=1)
    curve_ax.scatter(actual, x_at_frames, s=28, color="#d95b21",
                     zorder=3, label="Saved shapes")
    curve_ax.set(xlabel="Hencky strain", ylabel="x span", xlim=(0, 6.05),
                 ylim=(0, max(float(np.max(extension[:, index])) * 1.08, 1.0)))
    curve_ax.grid(alpha=0.22)
    curve_ax.legend(loc="upper left", ncol=2, frameon=False)

    for frame in range(7):
        chain = snapshots[frame, index]
        xy_ax = fig.add_subplot(grid[frame + 1, 0])
        xz_ax = fig.add_subplot(grid[frame + 1, 1])
        contour_ax = fig.add_subplot(grid[frame + 1, 2])
        projection(xy_ax, chain, 1)
        projection(xz_ax, chain, 2)
        contour_plot(contour_ax, chain)
        xy_ax.set_title(f"strain {target[frame]:g} | x={x_at_frames[frame]:.2f} | x-y",
                        fontsize=10)
        xz_ax.set_title("x-z projection", fontsize=10)
        contour_ax.set_title("x along bead index", fontsize=10)
        xy_ax.set(xlabel="x", ylabel="y")
        xz_ax.set(xlabel="x", ylabel="z")
        contour_ax.set(xlabel="Bead index", ylabel="x")

    fig.suptitle(f"Chain {chain_id:02d} | x span at strain 6 = {x_at_frames[-1]:.2f}",
                 fontsize=17, y=0.988)
    fig.text(0.5, 0.012,
             "Projections zoom independently. Red/blue mark beads 1/N; line color follows bead order.",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def draw_overview(snapshots, first_chain, output):
    """每页 16 条链，叠加应变 2、3、4 的归一化 x(珠子编号) 曲线。"""
    colors = ((2, "#6a51a3"), (3, "#238b45"), (4, "#e6550d"))
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    bead_index = np.arange(1, snapshots.shape[2] + 1)
    x6 = np.ptp(snapshots[-1, :, :, 0], axis=1)
    for offset, ax in enumerate(axes.flat):
        index = first_chain + offset
        for frame, color in colors:
            x = snapshots[frame, index, :, 0]
            normalized = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
            ax.plot(bead_index, normalized, color=color, linewidth=1.15,
                    label=f"strain {frame}")
        ax.set_title(f"Chain {index + 1:02d} | x(6)={x6[index]:.1f}", fontsize=10)
        ax.grid(alpha=0.18)
        ax.set(xlim=(1, len(bead_index)), ylim=(-0.05, 1.05))
    for ax in axes[-1]:
        ax.set_xlabel("Bead index")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.968), frameon=False)
    fig.suptitle(f"Contour overview | chains {first_chain + 1:02d}–"
                 f"{first_chain + 16:02d}", fontsize=17, y=0.996)
    fig.text(0.5, 0.015,
             "Each curve is scaled to its own x span. Large reversals may indicate folds; inspect the chain report.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.055, top=0.925,
                        wspace=0.18, hspace=0.30)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path,
                        help="默认输出到结果目录的 classification_views/")
    args = parser.parse_args()
    snapshots, target, actual, strain, extension = load_results(args.input)
    output = args.output or args.input.parent / "classification_views"
    chains_dir = output / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    for first in range(0, 64, 16):
        image = output / f"overview_chains_{first + 1:02d}-{first + 16:02d}.png"
        draw_overview(snapshots, first, image)
        print(f"已保存速览 {image.name}", flush=True)

    for chain_id in range(1, 65):
        image = chains_dir / f"chain_{chain_id:02d}.png"
        draw_chain_report(snapshots, target, actual, strain, extension, chain_id, image)
        if chain_id % 8 == 0:
            print(f"已完成详细图 {chain_id}/64", flush=True)
    print(f"分类视图已保存到 {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
