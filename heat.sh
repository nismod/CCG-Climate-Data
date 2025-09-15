#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --partition=Short
#SBATCH --array=0-999

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE='/lustre/soge1/users/mert2014/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/lustre/soge1/users/mert2014/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
export XDG_DATA_HOME=/ouce-home/staff/mert2014/.cache/

date
hostname
pwd

micromamba activate heat-analysis
python 2_heat.py /ouce-home/projects/mistral/ccg-2025-hazards/ $SLURM_ARRAY_TASK_ID 1000

date
echo "Done."
