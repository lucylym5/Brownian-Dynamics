from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib

import parameters as p
import necklace_chain as nc
import time_stepping as ts
import observables as ob

import tool

def main():

    """ 计算前准备 """
    # 理想 Hookean 链参考松弛时间
    tau_H = p.GAMMA / (
        4.0 * p.H * np.sin(np.pi / (2.0 * p.N))**2
    )

    lambda_H = tau_H / 2.0         # 二阶矩的松弛时间
    strain_rate = p.WI / lambda_H

    # 确定开流时长和积分步数
    flow_steps = int(np.ceil(p.TOTAL_TIME / p.TIME_STEP))

    print(f"N = {p.N}, M = {p.M}")
    print(f"Wi = {p.WI}")
    print(f"target_flow_time = {p.TOTAL_TIME:.6e} ps")
    print(f"flow_steps = {flow_steps}")

    rng = np.random.default_rng(p.RANDOM_SEED)

    # 生成满足FENE最大键长约束、且初始珠子不过度靠近的链。
    r = nc.initialize_chain(
        name="FENE",
        M=p.M,
        N=p.N,
        kBT=p.KBT,
        H=p.H,
        epsilon=p.EPSILON,
        sigma=p.SIGMA,
        rng=rng,
        dimension=3,
        Q0_square=p.Q02,
    )
    q = nc.connection_vector(r)
    q2 = np.sum(q**2, axis=-1)

    # 日志文件路径
    root_dir = Path(__file__).resolve().parent
    log_equilibria = root_dir / "results" / "log_equilibria.log"
    log_elongation = root_dir / "results" / "log_elongation.log"
    root_dir = Path(__file__).resolve().parent
    run_dir = (
        root_dir / "results"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    log_equilibria = run_dir / "equilibria.csv"
    log_elongation = run_dir / "configuration.log"
    log_extension = run_dir / "extension.tsv"

    print(f"结果目录：{run_dir}")



    """ 预平衡（无流动）, 3tau_H """ 
    no_flow = p.velocity_gradient(0)
    equilibria_time = 3 * tau_H
    equilibria_steps = int(
        np.ceil(equilibria_time / p.TIME_STEP)
    )

    log_step = max(1, equilibria_steps // 200)

    for step in range(1, equilibria_steps + 1):
        r = ts.fene_newton_step(
            r = r,
            velocity_gradient = no_flow,
            kBT = p.KBT,
            gamma = p.GAMMA,
            H = p.H,
            Q0_square = p.Q02,
            dt = p.TIME_STEP,
            epsilon_ev = p.EPSILON,
            sigma_ev = p.SIGMA,
            rng = rng,
        )
        # 记录 200 次回转半径，观察是否达平衡
        if step % log_step == 0 or step == equilibria_steps:
            rg2, end2, _ = ob.obervables(r)
            time_ps = step * p.TIME_STEP

            tool.output_log(
                    step,
                    time_ps,
                    end2.mean(),
                    rg2.mean(),
                    log_equilibria,
            )
            print(f"预平衡，共{equilibria_steps}步，第{step}步计算完成")




    """ 开启流场 """
    flow_gradient = p.velocity_gradient(strain_rate)
    tool.output_configuration_log(
            0,
            0,
            r,
            log_elongation,
    )

    _, _, span_x = ob.obervables(r)
    time_history = [0]
    span_x_history = [float(span_x.mean())]
    with log_extension.open("w", encoding="utf-8") as file:
        file.write("step\ttime_ps\tmean_span_x_A\n")
        file.write(f"0\t0\t{span_x_history[0]:.17g}\n")

    for step in range(1, flow_steps + 1):
        r = ts.fene_newton_step(
            r = r,
            velocity_gradient = flow_gradient,
            kBT = p.KBT,
            gamma = p.GAMMA,
            H = p.H,
            Q0_square = p.Q02,
            dt = p.TIME_STEP,
            epsilon_ev = p.EPSILON,
            sigma_ev = p.SIGMA,
            rng = rng,
        )
        if step % p.LOG_STEP == 0 or step == flow_steps:
            time_ps = step * p.TIME_STEP

            _, _, span_x = ob.obervables(r)
            mean_span_x = float(span_x.mean())

            time_history.append(time_ps)
            span_x_history.append(mean_span_x)

            # 每个记录点立即写入磁盘。
            with log_extension.open("a", encoding="utf-8") as file:
                file.write(
                    f"{step}\t{time_ps:.17g}\t{mean_span_x:.17g}\n"
                )

            tool.output_configuration_log(
                step,
                time_ps,
                r,
                log_elongation,
            )

            print(f"开流，共{flow_steps}步，第{step}步计算完成")

    tool.plot_xy(
        np.asarray(time_history) * 1e-12,     # ps → s
        np.asarray(span_x_history) * 1e-4,   # Å → μm
        xlabel="Time after flow onset (s)",
        ylabel=r"Mean x-span ($\mu$m)",
        label="Ensemble mean",
        title=f"FENE + EV, Wi = {p.WI} (Hookean reference)",
        save_path=run_dir / "extension.png",
    )

if __name__ == "__main__":
    main()
