#!/bin/bash

# 用途：在一个计算节点上启动一个 Python 主程序，由它创建多个工作进程。
# 每次提交前检查下面标有“按需修改”的位置；配置相同时不必重复修改。

# 【按需修改 1：任务名】用于 squeue 显示和日志文件命名。
#SBATCH --job-name=parallel
# 【按需修改 2：分区】换队列时修改。
#SBATCH --partition=amd_a8_384
# 当前使用单机多进程，通常保持 1 个节点、1 个 Slurm 任务。
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 【按需修改 3：核数和内存】当前为 4 核，整个节点上的本作业共用 8 GB。
# main_1_parallel.py 的 main() 中目前写的是 workers = min(4, p.M)。
# 若更改核数，也要同步修改 Python 中的 workers；只增加申请核数不会增加进程数。
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# 【按需修改 4：远程程序目录】换账号或目录时修改为真实的绝对路径。
# #SBATCH 指令不展开 $HOME 等 Shell 变量，不要将下一行写成 $HOME/elongation_ev。
#SBATCH --chdir=/publicfs01/fs1-a8/home/a8s002767/elongation_ev
# %x 为任务名，%j 为作业号；每次提交会生成不同的日志文件名。
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 【按需修改 5：运行时限】本文件不固定时限，提交时通过 --time 指定。
# 在远程登录节点执行以下命令；脚本和程序须事先上传到该目录：
#   cd "$HOME/elongation_ev"
#   sbatch --time=12:00:00 submit_elongation_ev.sh
# 12:00:00 只是示例，请按本次任务修改，且不要超过分区允许的时限。
# 若希望直接使用分区允许的最长时限，可复制执行下面命令：
#
# ev_time_limit=$(sinfo -h -p amd_a8_384 -o '%l' | head -n 1)
# case "$ev_time_limit" in
#   infinite|INFINITE|unlimited|UNLIMITED) ev_time_limit=0 ;;
# esac
# test -n "$ev_time_limit" && sbatch --time="$ev_time_limit" submit_elongation_ev.sh
# 若修改了上面的分区，这里的 sinfo -p 也要同步修改。

# 以下为运行部分。所有 #SBATCH 资源指令必须放在第一条 Shell 命令之前。
set -euo pipefail

# 【按需修改 6：并行程序文件名】本地当前入口名为 main_1_parallel.py。
# 若远程文件仍叫 main_parallel.py，或要运行 main_2_align1.py，只改下一行。
PYTHON_SCRIPT="main_1_parallel.py"

# 【按需修改 7：Python 环境】当前直接使用 Anaconda base 的解释器。
# 换环境时改为对应的 .../envs/环境名/bin/python 路径。
PYTHON_BIN="$HOME/anaconda3/bin/python"

# 【物理参数】链数、珠子数、Wi、时间步等在 module/parameters.py 中修改。
# 每个工作进程使用 1 个数学库线程；4 核由 Python 的 4 个进程使用。
# 即使更改申请核数，下面三个线程数通常仍保持为 1。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

printf 'Job: %s\nHost: %s\nStarted: %s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
printf 'Directory: %s\nProgram: %s\nPython: %s\nCPUs: %s\nTime limit: %s\n' \
  "$PWD" "$PYTHON_SCRIPT" "$PYTHON_BIN" "$SLURM_CPUS_PER_TASK" \
  "${SLURM_TIMELIMIT:-scheduler-controlled}"

# 自动沿用本次实际申请的核数，这里无需重复手动修改。
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$PYTHON_BIN" -u "$PYTHON_SCRIPT"

printf 'Finished: %s\n' "$(date -Is)"
