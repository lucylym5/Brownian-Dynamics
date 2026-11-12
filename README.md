# Zimm/Rouse chain Elongation Simulation
高分子流场操控仿真环境搭建

## Windows 与 Linux / WSL 环境

两种系统共用源代码和 `requirements.txt`，各自创建虚拟环境：Windows 使用
`.venv-win`，Linux / WSL 使用 `.venv-lin`。依赖包括 NumPy、Matplotlib 和
Numba，覆盖无流动、排斥体积、伸长流及后处理绘图程序。两边均可使用 Python 3.12。

不要直接把旧虚拟环境改名或跨系统共用。激活脚本和已安装命令含绝对路径，
NumPy / Numba 等包也有平台相关的二进制文件，应在目标目录重新创建环境。
参见 [Python venv 官方说明](https://docs.python.org/3.12/library/venv.html#how-venvs-work)。

### Windows：首次创建

在 PowerShell 中进入项目目录：

```powershell
cd "D:\博士\实习参考材料-刘一鸣-20260921\布朗动力学"
```

在新电脑上先安装 Python 3.12，再创建环境：

```powershell
py -3.12 -m venv .venv-win
```

若没有 `py` 启动器，可改用已安装的 Python 3.12 解释器运行 `-m venv .venv-win`。
本项目当前的 `.venv-win` 已创建完成，无需重复创建。安装依赖：

```powershell
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-win\Scripts\python.exe -m pip check
```

### Linux / WSL：首次创建

在 WSL 的 Bash 终端中执行，不能在 Windows PowerShell 中直接执行这些命令：

```bash
cd "/mnt/d/博士/实习参考材料-刘一鸣-20260921/布朗动力学"
python3 -m venv .venv-lin
.venv-lin/bin/python -m pip install -r requirements.txt
.venv-lin/bin/python -m pip check
```

若 Ubuntu 提示 `ensurepip` 不可用，先执行 `sudo apt install python3-venv`，
再重新运行创建命令。原生 Linux 使用自己的项目路径替换 `/mnt/d/...`。

### 每次运行

Windows PowerShell：

```powershell
cd "D:\博士\实习参考材料-刘一鸣-20260921\布朗动力学"
.\.venv-win\Scripts\Activate.ps1
python main_NoFlow_FENE.py
```

Linux / WSL Bash：

```bash
cd "/mnt/d/博士/实习参考材料-刘一鸣-20260921/布朗动力学"
source .venv-lin/bin/activate
python main_NoFlow_FENE.py
```

激活后两边的运行命令相同，将文件名替换为下面的入口即可：

| 入口 | 功能 |
| --- | --- |
| `main_NoFlow.py` | 无流动胡克链 |
| `main_NoFlow_FENE.py` | 无流动 FENE 链 |
| `main_NoFlow_EV.py` | 无流动胡克链与 WCA 排斥体积，默认使用 Numba |
| `ElongationFlow.py` | 伸长流模拟 |
| `ElongationFlow_3D_snapshots.py` | 轨迹三维快照 |
| `ElongationFlow_classification_views.py` | 轨迹分类视图 |

两个后处理入口目前要求输入包含 64 条链、目标应变 0 到 6 的七帧坐标，
与伸长流程序的默认链数和应变范围一致。

例如 `python main_NoFlow_EV.py --help` 可查看参数。`necklace_chain.py`、
`time_stepping.py` 等文件是供入口调用的模块。模块文件名和导入名的大小写必须一致。
仿真默认参数可能运行较久；环境检查可先用 `--help`，不会启动模拟。

检查当前使用的解释器和依赖：

```bash
python -c "import sys, numpy, matplotlib, numba; print(sys.executable); print('NumPy', numpy.__version__, 'Matplotlib', matplotlib.__version__, 'Numba', numba.__version__)"
```

解释器应位于当前系统对应的 `.venv-win` 或 `.venv-lin` 内。
安装包统一使用 `python -m pip`，让安装和运行使用同一个解释器；退出环境用 `deactivate`。
也可以直接调用 `.venv-win\Scripts\python.exe` 或 `.venv-lin/bin/python`，无需激活。
PowerShell 若限制运行 `Activate.ps1`，直接调用解释器即可。

VS Code 的 `.vscode/settings.json` 已配置 `python-envs.workspaceSearchPaths`，
将 `.venv-lin`、`.venv-win` 加入环境搜索范围。更新配置后：

1. 按 `Ctrl+Shift+P`，运行 `Python Environments: Refresh All Environment Managers`。
2. 再运行 `Python: Select Interpreter`，选择当前系统对应的环境。
3. 若列表仍未出现，选择 `Enter Interpreter Path...`，直接输入解释器的完整路径：
   - WSL：`/mnt/d/博士/实习参考材料-刘一鸣-20260921/布朗动力学/.venv-lin/bin/python`
   - Windows：`D:\博士\实习参考材料-刘一鸣-20260921\布朗动力学\.venv-win\Scripts\python.exe`
4. 关闭旧终端，新建终端，再用下面的命令确认解释器和 NumPy：

```bash
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
```

WSL 窗口使用 Linux 解释器，普通 Windows 窗口使用 Windows 解释器。
如果刷新后列表仍显示已删除的环境，可运行 `Developer: Reload Window` 后再选择。
只激活终端环境，不一定会改变编辑器运行按钮使用的解释器。
参见 [VS Code 环境发现与配置说明](https://code.visualstudio.com/docs/python/environments#configure-search-paths)。
Linux 无图形界面时先不加 `--show`，程序仍会保存图片。

两个系统可共用结果目录，但同时运行时应通过 `--output` 指定不同目录。
`.gitignore` 已忽略 `.venv-win/`、`.venv-lin/`；
提交代码和依赖清单即可，不提交虚拟环境。

2026-11-04 验证：Windows Python 3.12.14、WSL Ubuntu 24.04 Python 3.12.3；
两边均安装 NumPy 2.5.4、Matplotlib 3.11.2、Numba 0.68.1。
已通过 `pip check`、11 个模块导入、6 个入口的 `--help`、四类小规模模拟
（覆盖多进程与 Numba）以及两种后处理的代表图输出。这些检查用于确认环境可运行。

## noFlow：无流动胡克链与标度图

在包含 NumPy 和 Matplotlib 的 Python 环境中运行：

~~~powershell
python main_NoFlow.py
~~~

默认 M=500、dt=0.05，扫描 N=20,30,...,90，H=gamma=kBT=1，随机种子为2026。
每个 N 从胡克平衡构象出发，复用 euler_maruyama_step()，推进约3个理论最长相关时间。
总时间随N增大；默认最多同时计算4个N。
本脚本使用自己的配置，不使用 parameters.py 中的 TIME_STEP、MOLECULE_NUMBER 和 TOTAL_TIME。

~~~powershell
python main_NoFlow.py --chains 500 --dt 0.05 --duration 3
python main_NoFlow.py --n-step 1
python main_NoFlow.py --fit-start 0.3 --fit-end 1.2 --show
~~~

--duration 表示总时间与理论相关时间之比；--fit-start、--fit-end 也以理论相关时间为单位。
--n-step 1 逐个扫描所有整数N；--show 在保存后打开图片窗口。
当前结果和默认输出目录均为 noFlow_results。

### 绘图与拟合

同一个figure包含两个子图，采用等宽布局和 set_box_aspect(3/4)，
保证两块绘图区等大，宽高比均为4:3。左上角标注拟合斜率与R²，不标注Expected slope。

- 左图：log(sqrt(<Rg^2>))-log(N)。每条链先相对于自己的质心计算Rg²，
  再对链及记录时刻平均、最后开平方。
- 右图：log(tau_R)-log(N)。tau_R从第一Rouse模态的相关函数拟合得到。
- 两图均先取自然对数，再用np.polyfit拟合直线，并在原图绘制拟合线。

默认拟合0.2*tau_theory到1.5*tau_theory的相关函数。
理论值只用于安排运行长度、采样、拟合窗口及作为对照，不替代模拟测量值。
若某条曲线在整个选定窗口中无法给出有效正衰减时间，该点会被省略，不会替换为理论时间。
这里tau_R是模态向量的相关时间，不是用于二阶矩或Wi的tau_R/2。

每个理论相关时间记录约100帧，以避免在长链上保存每一步的所有模态数据。
输出中的时间仍为真实记录时刻，没有用帧编号代替物理时间。

### 输出文件

- scaling_laws.png：两个等大长方形绘图区的标度图。
- mode_correlations.png：以t/tau_theory为横轴的相关函数，灰色区域为拟合窗口。
- scaling_results.csv：各N的尺寸、相关时间、实际总时间、拟合窗口和采样信息。
- mode_correlations.npz：实际记录时间与相关曲线。
- summary.json：运行参数、拟合结果及两块绘图区的实际尺寸。

## 当前长时间运行结果

N=20,30,...,100，M=500，dt=0.05，每个N推进约3个理论相关时间：

| 量 | 拟合斜率 | 回归标准误差 | R² |
|---|---|---|---|
| Rg = sqrt(<Rg²>) | 0.50774 | 0.00256 | 0.99982 |
| 第一模态相关时间tau_R | 2.06301 | 0.06183 | 0.99375 |

数据和图片均保存在noFlow_results。回归标准误差描述不同N的拟合散布；
每个N的相关曲线、拟合区间和实际运行时间均可在保存的数据中查阅。
