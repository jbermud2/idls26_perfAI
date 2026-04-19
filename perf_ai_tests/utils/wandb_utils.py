from __future__ import annotations

import os
from typing import Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(args):
    if not args.use_wandb:
        print("W&B disabled. Pass --use-wandb to enable experiment logging.")
        return None

    if wandb is None:
        print("W&B logging requested but wandb is not installed. Install with: pip install wandb")
        return None

    if args.wandb_mode == "disabled":
        print("W&B disabled via --wandb-mode disabled.")
        return None

    api_key = os.environ.get("WANDB_API_KEY", "") or args.wandb_api_key

    if args.wandb_mode == "offline":
        print("W&B running in offline mode. Use wandb sync later to upload the run.")

    if args.wandb_mode == "online" and not api_key and args.wandb_anonymous == "never":
        print("No WANDB_API_KEY found in env/args. Will try existing local wandb login credentials.")

    if api_key:
        wandb.login(key=api_key)
    elif args.wandb_mode == "online":
        print(
            "W&B has no API key from --wandb-api-key or WANDB_API_KEY. "
            "Online runs may fail unless this machine is already logged in."
        )

    run_config = vars(args).copy()
    run_config.pop("wandb_api_key", None)

    entity = args.wandb_entity if args.wandb_entity else None
    init_kwargs = dict(
        project=args.wandb_project,
        entity=entity,
        name=args.wandb_run_name if args.wandb_run_name else None,
        mode=args.wandb_mode,
        config=run_config,
        anonymous=args.wandb_anonymous,
        id=args.wandb_run_id if args.wandb_run_id else None,
        resume=args.wandb_resume if args.wandb_run_id else None,
    )

    run = wandb.init(**init_kwargs)

    print(
        "W&B initialized: project={}, entity={}, mode={}, run_name={}".format(
            args.wandb_project,
            entity if entity else "<default>",
            args.wandb_mode,
            args.wandb_run_name if args.wandb_run_name else "<auto>",
        )
    )
    print(f"W&B run id: {run.id}")
    return run


def log_to_wandb(run, metrics: Dict, step: Optional[int] = None):
    if run is None:
        return
    run.log(metrics, step=step)


def finish_wandb(run):
    if run is None:
        return
    run.finish()
