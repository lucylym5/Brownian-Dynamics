import numpy as np
import necklace_chain as nc
from scipy.linalg import solve_banded


""" Newton法更新r """
def fene_newton_step(
        r: np.ndarray,
        velocity_gradient: np.ndarray,
        kBT: float,
        gamma: float,
        H: float,
        Q0_square: float,
        dt: float,
        epsilon_ev,
        sigma_ev,
        rng,
        max_iterations=500,
        tol=1e-9,
):
    M, N, dimension = r.shape
    
    Q0 = np.sqrt(Q0_square)
    D = kBT / gamma

    flow_term = r @ velocity_gradient.T * dt
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(size = r.shape)
    ev_force = nc.WCA_force(r, epsilon_ev, sigma_ev)
    r_mid = r + flow_term + ev_force * dt / gamma + noise  # 不考虑弹簧力时的位置

    # 正确新位置与 r_mid 残差
    def equation_residual(r_new):
        force = nc.spring_force(
            "FENE", r_new, H, Q0_square
        )

        return (
            r_new
            - r_mid
            - force * dt / gamma
        )

    # 求position处切线方向
    def fene_jacobian(position, H, Q0_square, dt, gamma):
        M, N, dimension = position.shape

        q = nc.connection_vector(position)
        s = 1 - np.sum(q**2, axis=-1) / Q0_square   # 1 - |q|^2/Q0^2

        identity = np.eye(dimension)                # 单位矩阵
        stiffness = H * (                           # 每根弹簧的张力导数，∂T/∂q，shape = (M, N-1, 3, 3)
            identity / s [..., None, None]
            + 2 * q[..., :, None] * q[..., None, :]
            / (Q0_square * s[..., None, None]**2)
        )

        size = N * dimension

        J = np.broadcast_to(
            np.eye(size), (M, size, size)
        ).copy()

        for i in range(N-1):
            left = slice(i * dimension, (i+1) * dimension)
            right = slice((i+1) * dimension, (i+2)*dimension)

            block = (dt / gamma) * stiffness[:, i]

            J[:, left, left] += block
            J[:, right, right] += block
            J[:, left, right] -= block
            J[:, right, left] -= block
        
        return J


    # Newton法（沿切线方向修正）迭代更新位置
    candidate_position = r.copy()

    for _ in range(max_iterations):
        residual = equation_residual(candidate_position)
        residual_norm = np.linalg.norm(residual)    # 残差的模长
        
        # 残差符合要求时，输出
        if np.max(np.abs(residual)) < tol:
            candidate_position -= candidate_position.mean(axis=1, keepdims=True)    # 去除质心漂移
            return candidate_position

        rhs = -residual.reshape(M, N * dimension, 1)                    # N*dimension 个未知数和方程
        J = fene_jacobian(candidate_position, H, Q0_square, dt, gamma)
        direction = np.linalg.solve(J,rhs).reshape(M, N, dimension)
        correction=1

        for _ in range(50):
            trial_position = candidate_position + correction * direction
            trial_q = nc.connection_vector(trial_position)
            trial_q_square = np.sum(trial_q**2, axis=-1)

            if np.all(trial_q_square < Q0_square):    # 判断弹簧伸长是否越界
                trial_residual = equation_residual(trial_position)
                if np.linalg.norm(trial_residual) < residual_norm:    # 修正后残差减小，接受这次修正
                    candidate_position = trial_position
                    break
            correction *= 0.5    # 越界或残差未减小，减小修正幅度   
        else:
            raise RuntimeError("位置修正失败，建议减小时间步 dt。")  
    else:   
        residual = equation_residual(candidate_position)
        if np.max(np.abs(residual)) < tol:
            candidate_position -= candidate_position.mean(axis=1, keepdims=True)    # 去除质心漂移
            return candidate_position
        else: 
            raise RuntimeError(
            "FENE 隐式求解未收敛，建议减小 dt 或增加最大循环次数。"
            )

    

""" Euler-Maruyama方法更新r """
def euler_maruyama_step(
        name: str,
        r: np.ndarray,
        velocity_gradient: np.ndarray,
        kBT: float,
        gamma: float,
        H: float,
        Q0_square: float,
        dt: float,
        epsilon_ev: float,
        sigma_ev: float,
        rng,
):
    spring_force = nc.spring_force(
        name, r, H, Q0_square
    )
    ev_force = nc.WCA_force(
        r, epsilon_ev, sigma_ev
    )

    flow_term = r @ velocity_gradient.T * dt
    force_term = (spring_force + ev_force) / gamma * dt
    D = kBT / gamma
    noise = np.sqrt(2 * D * dt) * rng.standard_normal(size = r.shape)

    return r + flow_term + force_term + noise