import submitit
import shlex

# singularity overlay file
# :ro is for read-only, so that multiple jobs can use the same overlay concurrently
OVERLAY = ["/scratch/qt2094/HW/DL/overlay-25GB-500K.ext3:ro"]  
# singularity image
SIF = "/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
# any environment setup commands after entering singularity, e.g. initializing conda
ENV_SETUP = """
source /scratch/qt2094/HW/DL/env.sh
conda activate dl_final
"""

def _submitit_command_str(self) -> str:
    overlay_str = " --overlay ".join(OVERLAY)
    return f"""singularity exec --nv --overlay {overlay_str} {SIF} /bin/bash -c "
    {ENV_SETUP}
    python3 -u -m submitit.core._submit {shlex.quote(str(self.folder))}
    "
    """

submitit.SlurmExecutor._submitit_command_str = property(_submitit_command_str)