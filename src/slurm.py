import os
import hashlib
import json


SBATCH_STR = '#SBATCH'
PKG_HOME = 'CIPHER'
SCRIPT_LOC = 'scripts'
OPTION_STR = '--'
TMP_DIR = os.path.join(SCRIPT_LOC, '.tmp')
STARTUP = [
    "",
    "module purge",
    "source ~/.bashrc",
    "module load mamba",
    'echo "Startup done..."',
    ""
]
DEACTIVATE_STR = 'mamba deactivate'
MODULES = ['r2', 'cov']


def str_to_bool(s: str):
    s = s.strip().lower()
    if s in {"true", "yes"}:
        return True
    elif s in {"false", "no"}:
        return False
    else:
        return None

def dict_to_hash8(d):
    # Serialize the dict in a consistent way
    dict_str = json.dumps(d, sort_keys=True)
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(dict_str.encode())
    # Convert to integer and keep 8 digits
    return str(int(hash_obj.hexdigest(), 16))[:8]

def slurm_script(
    config: dict,
    conda_env: str = 'cipher',
    module: str = 'r2',
    cache: bool = True,
    **kw_args,
    ) -> str | None:
    stp = ['#!/bin/bash']
    # Add all sbatch parameters
    for k, v in config.items():
        # Handle flags
        bv = str_to_bool(str(v))
        if bv is not None:
            if bv:
                stp.append(f'{SBATCH_STR} {OPTION_STR}{str(k)}')
        else:
            stp.append(f'{SBATCH_STR} {OPTION_STR}{str(k)}={str(v)}')
    # Add startup
    stp.extend(STARTUP)
    stp.append(f"mamba activate {str(conda_env)}")
    # Select module to run
    if module in MODULES:
        I = kw_args.get('input')
        O = kw_args.get('output', 'output')
        if I is None or O is None:
            raise ValueError(f"Both 'input' and 'output' arguments must be provided for script '{module}'.")
        if module == 'r2':
            call = f'python {SCRIPT_LOC}/r2.py -i "{I}" -o "{O}"'
        elif module == 'cov':
            call = f'python {SCRIPT_LOC}/cov.py -i "{I}" -o "{O}"'
    else:
        return [f"Module {module} not recognized. scuzi"]
    stp.append(call)
    stp.append(DEACTIVATE_STR)
    # Create tmp script path
    f = os.path.join(TMP_DIR, f"{module}_{dict_to_hash8(kw_args)}.sh")
    # Check if script already exists
    if os.path.exists(f) and cache:
        return None
    with open(f, "w") as tmp:
        for cl in stp:
            tmp.write(cl + "\n")
    return f

def clean_tmp():
    os.removedirs(TMP_DIR)
