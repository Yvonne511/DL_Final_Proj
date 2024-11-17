import submitit
import shlex

# singularity overlay file
# :ro is for read-only, so that multiple jobs can use the same overlay concurrently
OVERLAY = ["/vast/yw4142/overlays/overlay-50G-10M.ext3:ro"]  
# singularity image
SIF = "/scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif"
# any environment setup commands after entering singularity, e.g. initializing conda
ENV_SETUP = """
source /ext3/env.sh
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

