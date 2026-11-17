from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from module import parameters as p
from module import necklace_chain as nc
from module import time_stepping as ts
from module import observables as ob

from module import tool

def main():

    """ 计算前准备 """

    # 确定开流时长和积分步数
    flow_steps = int(np.ceil(p.TOTAL_TIME / p.DT))

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
    run_dir = (
        root_dir / "results"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    log_equilibria = run_dir / "equilibria.tsv"
    log_elongation = run_dir / "configuration.log"
    log_extension = run_dir / "extension.tsv"



    """ 预平衡（无流动）, 3tau_H """ 
    no_flow = p.velocity_gradient(0)
    equilibria_time = 3 * p.TAU_H
    equilibria_steps = int(
        np.ceil(equilibria_time / p.DT_PRE)
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
            dt = p.DT_PRE,
            epsilon_ev = p.EPSILON,
            sigma_ev = p.SIGMA,
            rng = rng,
        )
        # 记录 200 次回转半径，观察是否达平衡
        if step % log_step == 0 or step == equilibria_steps:
            rg2, end2, _ = ob.obervables(r)
            time_ps = step * p.DT_PRE

            tool.output_log(
                    step,
                    time_ps,
                    end2.mean(),
                    rg2.mean(),
                    log_equilibria,
            )
            print(f"预平衡，共{equilibria_steps}步，第{step}步计算完成")



    """ 开启流场 """
    flow_gradient = p.velocity_gradient(p.STRAIN_RATE)
    tool.output_configuration_log(
            0,
            0,
            r,
            log_elongation,
            log_extension,
    )

    for step in range(1, flow_steps + 1):
        r = ts.fene_newton_step(
            r = r,
            velocity_gradient = flow_gradient,
            kBT = p.KBT,
            gamma = p.GAMMA,
            H = p.H,
            Q0_square = p.Q02,
            dt = p.DT,
            epsilon_ev = p.EPSILON,
            sigma_ev = p.SIGMA,
            rng = rng,
        )
        if step % p.LOG_STEP == 0 or step == flow_steps:
            time_ps = step * p.DT

            tool.output_configuration_log(
                step,
                time_ps,
                r,
                log_elongation,
                log_extension,
            )

            print(f"开流，共{flow_steps}步，第{step}步计算完成")

    print(f"计算完成，结果目录：{run_dir}")

if __name__ == "__main__":
    main()
