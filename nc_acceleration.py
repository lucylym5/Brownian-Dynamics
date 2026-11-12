""" 对 WCA的计算 和 时间步更新 进行numba加速 """

import numpy as np
from numba import njit