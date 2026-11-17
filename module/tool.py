""" 可复用的工具 """

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import observables as ob


def fit_power_law(N, values, theory_slope = None):
    """ 拟合幂律 values = A * N^slope """
    x = np.log(np.asarray(N, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    coefficient, covariance = np.polyfit(x, y, 1, cov=True)
    fit = {
        "slope": float(coefficient[0]),
        "intercept": float(coefficient[1]),
        "slope_se": float(np.sqrt(covariance[0, 0])),
    }

    if theory_slope is not None:
        fit["theory_slope"] = float(theory_slope)
        fit["slope_deviation"] = float(coefficient[0] - theory_slope)
    return fit


def plot_xy(
    x,
    y,
    *,
    xlabel = None,
    ylabel = None,
    label  = None,
    linewidth = None,
    color = None,
    title  = None,
    ax = None,
    figsize = (7, 4.5),
    grid = True,
    save_path = None,
    dpi = 200,
    show = False,
    **plot_kwargs,
):
    """ 在 ax 中画曲线，y.shape = (xdim, 曲线数) """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim not in (1, 2):
        raise ValueError("x 必须是一维数组，y 必须是一维或二维数组。")
    if len(x) != len(y):
        raise ValueError("x 和 y 的长度必须相同。")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    if color is not None:
        plot_kwargs["color"] = color
    ax.plot(x, y, label=label, **plot_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(grid)

    # 只有存在有效的曲线名称时才显示图例，也兼容多次调用后的叠加。
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def output_log(
    step: int,
    time_ps: float,
    mean_end2: float,
    mean_rg2: float,
    output_dir: str | Path,
):
    """ 记录统计量 """
    path = Path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        if file.tell() == 0:
            file.write("step\ttime_ps\tmean_end2_A2\tmean_rg2_A2\n")
        file.write(
            f"{step}\t{time_ps:.17g}\t{mean_end2:.17g}\t{mean_rg2:.17g}\n"
        )


def output_configuration_log(
    step: int,
    time_ps: float,
    r: np.ndarray,
    config_dir: str | Path,
    extension_dir: str | Path,
):
    """ 记录某步的所有柱子位置，帧与帧之间用一个空行分隔 """
    r = np.asarray(r)
    path1 = Path(config_dir)
    path1.parent.mkdir(parents=True, exist_ok=True)
    if path1.exists():
        with path1.open("rb") as existing:
            if existing.read(6) == b"\x93NUMPY":
                raise ValueError("该路径已有 NPY 二进制数据，请为文本日志使用新的文件路径。")

    with path1.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            f"# step={step} time_ps={time_ps:.17g} "
            f"chains={r.shape[0]} beads={r.shape[1]}\n"
        )
        file.write("# chain bead x_A y_A z_A\n")
        for chain_id, chain in enumerate(r, start=1):
            for bead_id, position in enumerate(chain, start=1):
                x, y, z = position
                file.write(f"{chain_id} {bead_id} {x:.17g} {y:.17g} {z:.17g}\n")
        file.write("\n")

    """ 记录伸展量 """
    path2 = Path(extension_dir)
    _, _, span_x = ob.obervables(r)
    mean_span_x = float(span_x.mean())

    with path2.open("a", encoding="utf-8") as file:
        if file.tell() == 0:
            file.write("step\ttime_ps\tmean_span_x\n")
        file.write(
            f"{step}\t{time_ps:.17g}\t{mean_span_x:.17g}\n"
        )

