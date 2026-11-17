"""从已保存的坐标日志重建单链 x 跨度并绘图，不导入或运行仿真程序。

函数用法（路径也可传入字符串）：
    postprocess_parallel_run(run_dir, wi=20, zoom_end_s=2)

命令行用法，在“布朗动力学”目录执行：
    python postprocess_parallel.py results/run_20261112_211835_769691 --wi 20 --zoom-end 2

x 跨度定义为每条链 max(x)-min(x)，不是末端距。
坐标日志的单位为 Å、ps；图中的单位转换为 μm、s。
每条链绘制为蓝色细线，黑色粗线为全部单链等权的总体均值。
本文件独立于 parameters.py，避免用后来修改过的参数解释旧数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_FRAME_HEADER = re.compile(
    r"# step=(\d+) time_ps=(\S+) chains=(\d+) beads=(\d+)\s*"
)


def _source_record(path: Path) -> dict:
    """记录真实输入的哈希；不依赖当前仿真参数。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _read_batch(batch_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """逐帧读取坐标，内存中只保留跨度，不累积全部三维坐标。"""
    path = batch_dir / "configuration.log"
    before = path.stat()
    steps, times_ps, spans = [], [], []
    shape = None
    expected_ids = None

    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            match = _FRAME_HEADER.fullmatch(line.strip())
            if match is None:
                raise ValueError(f"{path}: 无法识别帧头：{line[:100]!r}")
            step, time_ps, chains, beads = match.groups()
            step, time_ps = int(step), float(time_ps)
            chains, beads = int(chains), int(beads)
            if chains < 1 or beads < 2 or not np.isfinite(time_ps):
                raise ValueError(f"{path}: 第 {step} 步的帧元数据无效。")
            if shape is None:
                shape = (chains, beads)
                expected_ids = np.column_stack(
                    (np.repeat(np.arange(1, chains + 1), beads),
                     np.tile(np.arange(1, beads + 1), chains))
                )
            elif shape != (chains, beads):
                raise ValueError(f"{path}: 不同帧的链数或珠子数不一致。")

            columns = next(stream, "").strip()
            if columns != "# chain bead x_A y_A z_A":
                raise ValueError(f"{path}: 第 {step} 步的列名不符合坐标日志格式。")
            count = chains * beads
            block = list(islice(stream, count))
            if len(block) != count:
                raise ValueError(f"{path}: 第 {step} 步坐标不完整，不绘制残缺帧。")
            try:
                values = np.fromstring("".join(block), dtype=np.float64, sep=" ")
            except ValueError as error:
                raise ValueError(f"{path}: 第 {step} 步含无法解析的坐标。") from error
            if values.size != count * 5:
                raise ValueError(f"{path}: 第 {step} 步坐标列数或行数不正确。")
            values = values.reshape(count, 5)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: 第 {step} 步含 NaN 或无穷大。")
            if not np.array_equal(values[:, :2], expected_ids):
                raise ValueError(f"{path}: 第 {step} 步链/珠子编号缺失、重复或顺序改变。")
            if steps and (step <= steps[-1] or time_ps <= times_ps[-1]):
                raise ValueError(f"{path}: 记录步数和时间必须严格递增。")

            x = values[:, 2].reshape(chains, beads)
            spans.append(np.ptp(x, axis=1))
            steps.append(step)
            times_ps.append(time_ps)

    if len(steps) < 2 or steps[0] != 0 or times_ps[0] != 0:
        raise ValueError(f"{path}: 需要从开流第 0 步开始的至少两帧完整记录。")
    step_array = np.asarray(steps, dtype=np.int64)
    time_array = np.asarray(times_ps, dtype=np.float64)
    span_array = np.asarray(spans, dtype=np.float64)
    report = {
        "batch": batch_dir.name,
        "chains": shape[0],
        "beads_per_chain": shape[1],
        "frames": len(steps),
        "last_step": int(step_array[-1]),
        "last_time_ps": float(time_array[-1]),
        "mean_crosscheck": "extension.tsv not present",
    }

    # 独立核对：坐标重算的批次平均值应与生产程序写下的均值一致。
    extension_path = batch_dir / "extension.tsv"
    if extension_path.exists():
        with extension_path.open("r", encoding="utf-8-sig") as stream:
            if stream.readline().strip() != "step\ttime_ps\tmean_span_x":
                raise ValueError(f"{extension_path}: 无法识别列名。")
        logged = np.loadtxt(extension_path, delimiter="\t", skiprows=1, ndmin=2)
        if logged.shape != (len(steps), 3) or not np.isfinite(logged).all():
            raise ValueError(f"{extension_path}: 均值记录不完整或含无效值。")
        if not np.array_equal(logged[:, 0], step_array):
            raise ValueError(f"{extension_path}: 均值记录与坐标记录的步数不一致。")
        if not np.allclose(logged[:, 1], time_array, rtol=1e-12, atol=0):
            raise ValueError(f"{extension_path}: 均值记录与坐标记录的实际时间不一致。")
        rebuilt_mean = span_array.mean(axis=1)
        max_error = float(np.max(np.abs(rebuilt_mean - logged[:, 2])))
        if not np.allclose(rebuilt_mean, logged[:, 2], rtol=1e-12, atol=1e-7):
            raise ValueError(f"{extension_path}: 坐标重算均值与日志不符，最大差异 {max_error:g} Å。")
        report["mean_crosscheck"] = "passed"
        report["max_mean_difference_A"] = max_error
        report["extension_source"] = _source_record(extension_path)

    report["coordinate_source"] = _source_record(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"读取期间 {path} 发生变化，请在文件稳定后重新处理。")
    return step_array, time_array, span_array, report


def _plot_extension(
    time_s: np.ndarray,
    span_um: np.ndarray,
    output_stem: Path,
    *,
    beads: int,
    wi: float | None,
    end_s: float | None = None,
) -> None:
    """用蓝色细线显示单链差异，叠加黑色总体均值，不做平滑。"""
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": True, "axes.spines.right": True,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }):
        fig = Figure(figsize=(8, 6), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        chains = span_um.shape[1]
        ax.plot(time_s, span_um, color="#4477AA", alpha=0.4,
                linewidth=0.8, zorder=2)
        ax.plot(time_s, span_um.mean(axis=1), color="black",
                linewidth=2.5, label="Ensemble mean", zorder=4)
        ax.set_xlabel(r"Time $t$ (s)")
        ax.set_ylabel(r"Extension $x(t)$ ($\mu$m)")
        title = f"N={beads}, {chains} independent chains"
        if wi is not None:
            title = f"Wi={wi:g}, " + title
        ax.set_title(title, pad=8)
        upper_time = float(time_s[-1]) if end_s is None else float(end_s)
        ax.set_xlim(float(time_s[0]), upper_time)
        ax.set_ylim(bottom=0)
        ax.grid(True, color="#DDDDDD", linewidth=0.65, alpha=0.65)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
        fig.savefig(output_stem.with_suffix(".png"), dpi=240)
        fig.savefig(output_stem.with_suffix(".svg"))
        fig.clear()


def postprocess_parallel_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    wi: float | None = None,
    zoom_end_s: float | None = None,
) -> dict[str, Path]:
    """重建并绘制并行运行结果，返回生成文件的绝对路径。

    run_dir: 含 batch_*/configuration.log 的运行目录，也支持单批次目录。
    output_dir: 默认为运行目录下的新 postprocess 子目录。
    wi: 可选的图题标签，必须由该次运行记录确认，不读取当前 parameters.py。
    zoom_end_s: 可选的开流早期局部图结束时间（秒），使用原始数据，不做平滑。

    四批数据按各自链数合并后再计算均值；不把批次均值简单等权平均。
    不运行模拟、不改输入、不插值补帧；遇到不完整或不一致的数据会明确报错。
    """
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if wi is not None and not np.isfinite(wi):
        raise ValueError("Wi 标签必须为有限数值。")
    batches = sorted(p for p in root.glob("batch_*") if p.is_dir())
    if not batches and (root / "configuration.log").is_file():
        batches = [root]
    if not batches:
        raise FileNotFoundError(f"{root} 下没有 batch_* 坐标输出。")

    reports, all_spans, batch_ids, local_ids = [], [], [], []
    steps = time_ps = beads = None
    for batch in batches:
        print(f"正在读取 {batch.name}/configuration.log ...", flush=True)
        b_steps, b_times, b_spans, report = _read_batch(batch)
        if steps is None:
            steps, time_ps = b_steps, b_times
            beads = report["beads_per_chain"]
        elif (not np.array_equal(steps, b_steps)
              or not np.allclose(time_ps, b_times, rtol=1e-12, atol=0)
              or beads != report["beads_per_chain"]):
            raise ValueError(f"{batch}: 各批步数、实际时间或珠子数不一致，不能直接合并。")
        all_spans.append(b_spans)
        reports.append(report)
        batch_ids.extend([batch.name] * b_spans.shape[1])
        local_ids.extend(range(1, b_spans.shape[1] + 1))
        print(f"  {report['frames']} 帧，{report['chains']} 条链；均值核对："
              f"{report['mean_crosscheck']}", flush=True)

    time_s = time_ps * 1e-12
    span_um = np.concatenate(all_spans, axis=1) * 1e-4
    if zoom_end_s is not None and not (time_s[0] < zoom_end_s < time_s[-1]):
        raise ValueError("局部图结束时间必须位于记录时间范围内部。")
    out = Path(output_dir).expanduser().resolve() if output_dir else root / "postprocess"
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "figure": out / "extension_all_chains.png",
        "vector_figure": out / "extension_all_chains.svg",
        "data": out / "extension_data.npz",
        "statistics": out / "extension_statistics.csv",
        "report": out / "postprocess_summary.json",
    }
    _plot_extension(time_s, span_um, files["figure"].with_suffix(""), beads=beads, wi=wi)
    if zoom_end_s is not None:
        stem = out / "extension_early_time"
        _plot_extension(time_s, span_um, stem, beads=beads, wi=wi, end_s=zoom_end_s)
        files["zoom_figure"] = stem.with_suffix(".png")
        files["zoom_vector_figure"] = stem.with_suffix(".svg")

    mean_um = span_um.mean(axis=1)
    np.savez_compressed(
        files["data"], steps=steps, time_ps=time_ps, time_s=time_s,
        span_x_um=span_um, mean_span_x_um=mean_um,
        batch_names=np.asarray(batch_ids), chain_id_within_batch=np.asarray(local_ids),
        global_chain_id=np.arange(1, span_um.shape[1] + 1),
    )
    statistics = np.column_stack((steps, time_s, mean_um,
        span_um.std(axis=1, ddof=1 if span_um.shape[1] > 1 else 0),
        span_um.min(axis=1), span_um.max(axis=1)))
    np.savetxt(
        files["statistics"], statistics, delimiter=",", comments="",
        header="step,time_s,mean_span_x_um,std_over_chains_um,min_span_x_um,max_span_x_um",
        fmt=["%d", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g"],
    )
    summary = {
        "source_run": str(root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "observable": "x-span = max_bead(x) - min_bead(x), per chain",
        "coordinate_units": {"length": "angstrom", "time": "ps"},
        "plot_units": {"length": "micrometre", "time": "second"},
        "frames": int(len(time_s)), "chains": int(span_um.shape[1]),
        "beads_per_chain": beads, "batches": len(batches),
        "time_range_s": [float(time_s[0]), float(time_s[-1])],
        "first_step": int(steps[0]), "last_step": int(steps[-1]),
        "initial_mean_span_x_um": float(mean_um[0]),
        "final_mean_span_x_um": float(mean_um[-1]),
        "wi_label_supplied_by_caller": wi, "zoom_end_s": zoom_end_s,
        "interpolation": False, "smoothing": False,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "batch_checks": reports,
    }
    files["report"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(f"完成：{span_um.shape[1]} 条链、{len(time_s)} 帧、"
          f"{time_s[-1]:g} 秒；结果目录：{out}", flush=True)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="已有运行结果目录")
    parser.add_argument("--output", type=Path, help="另存后处理结果的目录")
    parser.add_argument("--wi", type=float, help="由该次运行记录确认的 Wi，仅用于图题")
    parser.add_argument("--zoom-end", type=float, help="额外绘制 0 到此时刻（秒）的局部图")
    args = parser.parse_args()
    postprocess_parallel_run(args.run_dir, output_dir=args.output,
                             wi=args.wi, zoom_end_s=args.zoom_end)
