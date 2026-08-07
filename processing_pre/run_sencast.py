"""
run_sencast.py
────────────────
Shared helper that drives the eawag/sencast Docker image (satellite image
download + atmospheric correction) once per .ini parameter file found in a
given folder.

Equivalent to the PowerShell loop:

    $paramFiles = Get-ChildItem "<params_dir>/*.ini"
    foreach ($file in $paramFiles) {
        docker run -it --rm `
            -v "<sencast_dir>:/sencast" `
            -v "<dias_dir>:/DIAS" `
            eawag/sencast:latest `
            -e /sencast/docker.ini `
            -p "/sencast/parameters/$($file.Name)"
    }

Differences from the raw PowerShell command:
  - The parameters folder is mounted explicitly as its own volume
    (host params_dir -> container /sencast/parameters), instead of always
    being the "parameters" subfolder of sencast_dir. This is what lets
    each main_*.py point at a different parameters folder on disk while
    keeping the exact same docker invocation.
  - "-it" (interactive + pseudo-TTY) is dropped since this runs
    non-interactively from a Python subprocess, which has no TTY to
    attach. Re-add it (pass extra_docker_args=["-it"]) if you need to
    watch/interrupt a run interactively instead.

Requires Docker Desktop (or equivalent) running locally and the
eawag/sencast:latest image pulled.
"""

import subprocess
from pathlib import Path


def run_sencast(sencast_dir, dias_dir, params_dir, extra_docker_args=None):
    """Run sencast once per *.ini file in params_dir.

    sencast_dir : host path to the sencast install (contains docker.ini)
                  -> mounted to /sencast
    dias_dir    : host path used as scratch/download space
                  -> mounted to /DIAS
    params_dir  : host path containing the *.ini parameter files
                  -> mounted to /sencast/parameters
    """
    params_dir = Path(params_dir)
    ini_files = sorted(params_dir.glob("*.ini"))

    if not ini_files:
        print(f"No .ini parameter files found in {params_dir}, skipping sencast.")
        return

    for ini_file in ini_files:
        print(f"\n--- sencast: {ini_file.name} ---")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{sencast_dir}:/sencast",
            "-v", f"{params_dir}:/sencast/parameters",
            "-v", f"{dias_dir}:/DIAS",
            *(extra_docker_args or []),
            "eawag/sencast:latest",
            "-e", "/sencast/docker.ini",
            "-p", f"/sencast/parameters/{ini_file.name}",
        ]
        subprocess.run(cmd, check=True)
