from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

import parameters as p
import necklace_chain as nc
import time_stepping as ts
import observables as ob

import tool

def run_batch_chain(batch_id, batch_size, run_dir):   
    """ 计算一批链条：批次，链数，结果目录 """
    # 理想 Hookean 链参考松弛时间
    tau_H = p.GAMMA / (
        4.0 * p.H * np.sin(np.pi / (2.0 * p.N))**2
    )

    lambda_H = tau_H / 2.0         # 二阶矩的松弛时间
    strain_rate = p.WI / lambda_H

    # 确定开流时长和积分步数
    flow_steps = int(np.ceil(p.TOTAL_TIME / p.TIME_STEP))

    print(f"N = {p.N}, M = {batch_size}")
    print(f"Wi = {p.WI}")
    print(f"target_flow_time = {p.TOTAL_TIME:.6e} ps")
    print(f"flow_steps = {flow_steps}")

    rng = np.random.default_rng([batch_id, p.RANDOM_SEED])

    # 生成满足FENE最大键长约束、且初始珠子不过度靠近的链。
    r = nc.initialize_chain(
        name="FENE",
        M=batch_size,
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
    run_dir = Path(run_dir) / f"batch_{batch_id + 1:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)

    log_equilibria = run_dir / "equilibria.tsv"
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
            dt = p.TIME_STEP,
            epsilon_ev = p.EPSILON,
            sigma_ev = p.SIGMA,
            rng = rng,
        )
        if step % p.LOG_STEP == 0 or step == flow_steps:
            time_ps = step * p.TIME_STEP

            tool.output_configuration_log(
                step,
                time_ps,
                r,
                log_elongation,
                log_extension,
            )

            print(
                f"[批次 {batch_id + 1}] "
                f"开流，共{flow_steps}步，第{step}步计算完成"
            )


def main():
    
    # 将链分配给 workers 个进程 
    workers = min(4, p.M)   # 进程数

    batch_sizes = [p.M // workers] * workers
    for i in range(p.M % workers):
        batch_sizes[i] += 1

    # 创建计算目录
    run_dir = (
        Path(__file__).resolve().parent 
        / "results"/ datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"进程数：{workers}，每批链数：{batch_sizes}\n"
        f"结果目录：{run_dir}\n"
    )

    # 提交各批任务，收集结果
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = []

        for batch_id in range(workers):
            job = pool.submit(
                run_batch_chain,    # 要执行的函数
                batch_id,
                batch_sizes[batch_id],
                run_dir,
            )
            jobs.append(job)
        
        for job in jobs:
            job.result()

    print(f"计算完成，结果目录：{run_dir}")


if __name__ == "__main__":
    main()
