"""为 64 条链各画一张七行的三维构象图。

每张图纵向展示同一条链在 Hencky 应变 0、1、2、3、4、5、6 的七帧。
全部图片共用从 64 条链的七帧坐标计算出的 xlim、ylim、zlim 和视角。
红点是第 1 颗珠子，蓝点是第 N 颗珠子，其余珠子画为黑点。
视角偏向 x-y 平面，以显示 x 向拉伸、y 向压缩及少量 z 起伏。
默认读取 ElongationalFlow_result1/trajectories.npz，输出到其旁边的
all_chains_3d_beads/。可用 --selected-four 只画应变 6 的 x 伸展量四个代表值。

运行：
    python ElongationFlow_3D_snapshots.py
    python ElongationFlow_3D_snapshots.py --input my_run/trajectories.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_INPUT = (Path(__file__).resolve().parent / "ElongationalFlow_result1"
                 / "trajectories.npz")


def load_snapshots(path):
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}；请先运行 ElongationFlow.py。")
    with np.load(path, allow_pickle=False) as data:
        required = {"snapshots", "snapshot_target_strain", "snapshot_strain",
                    "snapshot_time"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} 缺少字段：{', '.join(sorted(missing))}")
        positions = data["snapshots"]
        targets = data["snapshot_target_strain"]
        actual = data["snapshot_strain"]
        times = data["snapshot_time"]

    if positions.ndim != 4 or positions.shape[0] != 7 or positions.shape[1] != 64 \
            or positions.shape[2] < 2 or positions.shape[3] != 3:
        raise ValueError("snapshots 应为 (7, 64, N, 3) 的坐标数组。")
    if not (np.array_equal(targets, np.arange(7, dtype=float))
            and actual.shape == (7,) and times.shape == (7,)):
        raise ValueError("需要目标 Hencky 应变为 0、1、2、3、4、5、6 的七帧。")
    if not (np.isfinite(positions).all() and np.isfinite(actual).all()
            and np.isfinite(times).all()):
        raise ValueError("坐标、应变或时间中存在非有限值。")
    return positions, targets, actual, times


def select_chains(final_positions):
    """按应变 6 的 x 跨度排序；1/3、2/3 为升序第 22、43 位。"""
    extension_x = np.ptp(final_positions[:, :, 0], axis=1)
    order = np.argsort(extension_x, kind="stable")
    count = len(order)
    return [
        ("Longest x", "longest", int(order[-1]), float(extension_x[order[-1]])),
        ("Shortest x", "shortest", int(order[0]), float(extension_x[order[0]])),
        ("1/3 quantile", "lower_third", int(order[(count - 1) // 3]),
         float(extension_x[order[(count - 1) // 3]])),
        ("2/3 quantile", "upper_third", int(order[2 * (count - 1) // 3]),
         float(extension_x[order[2 * (count - 1) // 3]])),
    ]


def common_limits(positions, selected):
    """从传入链的七帧坐标确定所有图共用的对称坐标范围。"""
    chosen = positions[:, [item[2] for item in selected], :, :]
    return np.maximum(np.max(np.abs(chosen), axis=(0, 1, 2)) * 1.08, 1.0)


def draw_chain_history(positions, targets, actual, times, selected, half_ranges,
                       output, elev, azim):
    label, _, chain_index, final_extension = selected
    fig = plt.figure(figsize=(8, 32))
    display_aspect = np.minimum(half_ranges / half_ranges.min(), 4.0)
    for frame, target in enumerate(targets):
        ax = fig.add_subplot(7, 1, frame + 1, projection="3d", proj_type="ortho")
        chain = positions[frame, chain_index]
        current_extension = float(np.ptp(chain[:, 0]))
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2],
                color="#38566b", linewidth=1.5)
        ax.scatter(chain[1:-1, 0], chain[1:-1, 1], chain[1:-1, 2],
                   s=20, color="#111111", edgecolors="white", linewidths=0.2,
                   depthshade=True)
        ax.scatter(*chain[0], s=55, color="#d62728", edgecolors="white",
                   linewidths=0.6, depthshade=False)
        ax.scatter(*chain[-1], s=55, color="#1769c2", edgecolors="white",
                   linewidths=0.6, depthshade=False)

        # 全部图片的七帧使用同一组坐标边界和三维显示比例。
        ax.set_xlim(-half_ranges[0], half_ranges[0])
        ax.set_ylim(-half_ranges[1], half_ranges[1])
        ax.set_zlim(-half_ranges[2], half_ranges[2])
        ax.set_box_aspect(display_aspect, zoom=1.12)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("x", labelpad=7)
        ax.set_ylabel("y", labelpad=4)
        ax.set_zlabel("z", labelpad=4)
        ax.set_title(f"Hencky strain {target:g} (actual {actual[frame]:.6f}, "
                     f"t={times[frame]:.4f}) | x={current_extension:.2f}",
                     fontsize=11, pad=5)

    prefix = f"{label} | " if label else ""
    fig.suptitle(f"{prefix}chain {chain_index + 1} | x(6)={final_extension:.2f}",
                 fontsize=17, y=0.995)
    handles = [
        Line2D([], [], color="#38566b", linewidth=1.5, label="Chain"),
        Line2D([], [], marker="o", linestyle="", color="#111111", label="Interior beads"),
        Line2D([], [], marker="o", linestyle="", color="#d62728", label="Bead 1"),
        Line2D([], [], marker="o", linestyle="", color="#1769c2", label="Bead N"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.982),
               ncol=4, frameon=False)
    fig.text(0.5, 0.012,
             "All figures use identical x/y/z limits and view. Camera favors the x-y plane; display aspect is capped at 4:1.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.965,
                        hspace=0.18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="包含 snapshots 的 trajectories.npz")
    parser.add_argument("--output", type=Path,
                        help="PNG 输出目录；默认是输入文件旁的 all_chains_3d_beads")
    parser.add_argument("--selected-four", action="store_true",
                        help="只画应变 6 的 x 最长、最短和 1/3、2/3 位链")
    parser.add_argument("--elev", type=float, default=65.0,
                        help="三维视角的俯仰角，默认 65 度")
    parser.add_argument("--azim", type=float, default=-90.0,
                        help="三维视角的方位角，默认 -90 度")
    args = parser.parse_args()

    positions, targets, actual, times = load_snapshots(args.input)
    if not np.isfinite(args.elev) or not np.isfinite(args.azim):
        parser.error("视角角度必须为有限数值。")
    final_extension = np.ptp(positions[-1, :, :, 0], axis=1)
    all_chains = [("", f"chain_{index + 1:02d}", index, float(final_extension[index]))
                  for index in range(positions.shape[1])]
    half_ranges = common_limits(positions, all_chains)
    selected = select_chains(positions[-1]) if args.selected_four else all_chains
    print(f"统一坐标范围：x=±{half_ranges[0]:.4g}, "
          f"y=±{half_ranges[1]:.4g}, z=±{half_ranges[2]:.4g}", flush=True)
    default_folder = ("selected_chains_3d_beads" if args.selected_four
                      else "all_chains_3d_beads")
    output = args.output or args.input.parent / default_folder
    output.mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(selected, start=1):
        name = (f"{item[1]}_x_chain_{item[2] + 1:02d}.png" if args.selected_four
                else f"chain_{item[2] + 1:02d}.png")
        image = output / name
        draw_chain_history(positions, targets, actual, times, item, half_ranges,
                           image, args.elev, args.azim)
        if number % 8 == 0 or number == len(selected):
            print(f"已完成 {number}/{len(selected)} 张：{output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
