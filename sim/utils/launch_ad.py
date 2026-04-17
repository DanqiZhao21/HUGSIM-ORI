import subprocess
import time
import os
import shutil


def _resolve_shell():
    shell_from_env = os.environ.get("SHELL")
    if shell_from_env and os.path.exists(shell_from_env):
        return shell_from_env

    for shell_name in ("zsh", "bash", "sh"):
        shell_path = shutil.which(shell_name)
        if shell_path is not None:
            return shell_path

    raise FileNotFoundError("No usable shell found. Install bash/sh or set SHELL.")


def launch(shell_path, cuda_id, output, extra_env=None):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    shell_bin = _resolve_shell()
    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        process = subprocess.Popen(
            [shell_bin, shell_path, cuda_id, output],
            stdout=f,
            stderr=f,
            env=child_env,
        )
    return process


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            process.kill()
            return
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    process.kill()
    print("The AD algorithm process is killed.")
