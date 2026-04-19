from __future__ import annotations

import builtins
import torch
import gc
import os
import pdb
import sys
import random
import numpy as np
import torch.distributed as dist

_builtin_print = builtins.print


def disable_interactive_breakpoints():
    """Disable breakpoint()/pdb.set_trace() in non-interactive training runs."""
    os.environ.setdefault("PYTHONBREAKPOINT", "0")

    def _noop_breakpoint(*args, **kwargs):
        return None

    def _noop_exit(*args, **kwargs):
        return None

    builtins.breakpoint = _noop_breakpoint
    pdb.set_trace = _noop_breakpoint
    builtins.exit = _noop_exit
    builtins.quit = _noop_exit
    sys.exit = _noop_exit


def cuda_gc():
    """Run Python GC and return cached CUDA blocks to the driver after large object drops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed: int = 42):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_seed_ddp(seed: int = 42):
    """Call this AFTER dist.init_process_group()"""
    rank = dist.get_rank()
    worker_seed = seed + rank # Each rank gets a unique offset seed

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)