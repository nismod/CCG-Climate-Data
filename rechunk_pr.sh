#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=01:20:00
#SBATCH --partition=Short
#SBATCH --array=0-163

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

date
hostname
pwd

micromamba activate heat-analysis
python 1_rechunk_var.py /ouce-home/projects/mistral/ccg-2025-hazards/ /ouce-home/data/incoming/NEX-GDDP-CMIP6/ pr $SLURM_ARRAY_TASK_ID

date
echo "Done."
