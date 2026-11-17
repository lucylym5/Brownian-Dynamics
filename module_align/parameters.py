import numpy as np

""" in metal unit: Å, ps, eV """
""" 链参数 """
N = 30                     # 每条链珠子数
M = 100                    # 链数
H = 7.89e-9                # eV/Å^2
Q02 = 5.7555e7    
Q0 = np.sqrt(Q02)          # 单根弹簧的最大长度
INITIAL_POSITION = (0, 0, 0)

""" 物理常数 """
KBT = 2.55e-2              # 296 K
GAMMA = 6.82e2             # eV·ps/Å²
EPSILON = 2.55e-2          # WCA能量尺度
SIGMA = 9.79e2             # 有效排斥长度

""" 模拟参数 """
WI = 2

# 理想 Hookean 链参考松弛时间
TAU_H = GAMMA / (          
    4.0 * H * np.sin(np.pi / (2.0 * N))**2
)
# 二阶矩的松弛时间    
LAMBDA_H = TAU_H / 2.0     

STRAIN_RATE = WI / LAMBDA_H
TOTAL_STRAIN = 5
TOTAL_TIME = TOTAL_STRAIN / STRAIN_RATE
DT = 5e6
DT_PRE = 5e6

LOG_STRAIN = 0.01
LOG_STEP = max(
    1,
    int(round(LOG_STRAIN / (STRAIN_RATE * DT)))
)


RANDOM_SEED = 2026


def velocity_gradient(strain_rate):
    return np.array([
        [strain_rate, 0.0, 0.0],
        [0.0, -strain_rate, 0.0],
        [0.0, 0.0, 0.0]
    ])