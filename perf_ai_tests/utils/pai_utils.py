from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

_FORWARD_FN_MAP = {
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    GPA = None
    UPA = None


def selected_convert_module_id(args):
    return ".classifier_fc" if args.pai_convert_target == "classifier_fc" else ".pre_fc"


def configure_pai(args, model: torch.nn.Module):
    GPA.pc.set_verbose(True)
    GPA.pc.set_testing_dendrite_capacity(False)

    convert_module_id = selected_convert_module_id(args)

    # TODO: Need to verify this with Rorry - In the meet he mentioned it will be an empty list anyway. But globals_perforatedai.py (688 - 694) has few items.
    # GPA.pc.set_module_names_to_perforate([])
    GPA.pc.set_module_ids_to_perforate([convert_module_id])

    ids_to_track = [f".{name}" for name, module in model.named_modules() if len(list(module.children())) == 0 and not name == convert_module_id.lstrip(".")]
    GPA.pc.set_module_ids_to_track(ids_to_track)
    print(f"Set the following modules for perforation: {convert_module_id}")

    if args.strict_unwrapped_check or args.strict_weight_decay_check:
        print("PAI selectors:")
        if hasattr(GPA.pc, "get_module_ids_to_track"):
            print("  module_ids_to_track:", GPA.pc.get_module_ids_to_track())
        if hasattr(GPA.pc, "get_module_ids_to_perforate"):
            print("  module_ids_to_perforate:", GPA.pc.get_module_ids_to_perforate())
        if hasattr(GPA.pc, "get_module_names_to_perforate"):
            print("  module_names_to_perforate:", GPA.pc.get_module_names_to_perforate())

    if hasattr(GPA.pc, "set_weight_decay_accepted") and not args.strict_weight_decay_check:
        GPA.pc.set_weight_decay_accepted(True)
    
    #GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    #GPA.pc.set_n_epochs_to_switch(args.epochs//10)

    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    switch_interval = args.epochs // (args.max_dendrites + 1)
    GPA.pc.set_fixed_switch_num(switch_interval)
    GPA.pc.set_first_fixed_switch_num(switch_interval)


    if float(args.improvement_threshold).is_integer():
        preset = int(args.improvement_threshold)
        if preset == 0:
            threshold = [0.01, 0.001, 0.0001, 0]
        elif preset == 1:
            threshold = [0.001, 0.0001, 0]
        elif preset == 2:
            threshold = [0]
        else:
            print(f"WARNING: You are trying to set an improvement threshold that does not match any of the preset.")
            threshold = args.improvement_threshold
    else:
        threshold = args.improvement_threshold
    
    GPA.pc.set_improvement_threshold(threshold)
    print(f"Improvement Threshold Used: {threshold}")

    GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_mult)
    print(f"Weight Initialization Multiplier Used: {args.candidate_weight_init_mult}")

    forward_fn = _FORWARD_FN_MAP.get(args.pai_forward_function)
    if forward_fn is None:
        raise ValueError(f"Unknown pai_forward_function '{args.pai_forward_function}'. Choose from: {list(_FORWARD_FN_MAP)}")
    GPA.pc.set_pai_forward_function(forward_fn)
    print(f"Forward Activation Function Used: {args.pai_forward_function}")

    if args.dendrite_mode == 0: # No dendrites
        GPA.pc.set_max_dendrites(0)
        if hasattr(GPA.pc, "set_perforated_backpropagation"):
            print("Adding no Dendrites [set to 0 - set_max_dendrites(0)]")
            GPA.pc.set_perforated_backpropagation(False)
    elif args.dendrite_mode in (1, 2):
        GPA.pc.set_max_dendrites(args.max_dendrites)
        if hasattr(GPA.pc, "set_perforated_backpropagation"):
            if args.dendrite_mode == 2:
                print("Enabled Perforated Backpropagation")
            elif args.dendrite_mode == 1:
                print("Enabled Normal Gradient Descent Upgrades (No Perforated Backpropagation)")
            GPA.pc.set_perforated_backpropagation(args.dendrite_mode == 2)
            
    
    print("Finished configuring PAI")


def setup_pai_optimizer_scheduler(args, model: torch.nn.Module):
    GPA.pai_tracker.set_optimizer(optim.AdamW)
    GPA.pai_tracker.set_scheduler(CosineAnnealingLR)
    # GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    for name in ["pre_fc", "classifier_fc"]:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found (all layers are frozen). "
            "Enable trainable layers or run with --finetune-backbone."
        )

    optim_args = {
        "params": trainable_params,  # model.parameters() ?
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if args.epochs > 0:
        t_max = args.epochs
    else:
        raise RuntimeError("Epochs is set to 0. Cannot train. This is used to mainly set T_max for CosineAnnealingLR Scheduler")
    sched_args = {
        "T_max": t_max
    }
    # sched_args = {'mode':'max', 'patience': 1} # Make sure this is lower than epochs to switch

    optim_args_for_print = {k: v for k, v in optim_args.items() if k != "params"}
    print(f"Optimizer hyperparameters: {optim_args_for_print}")
    print(f"Scheduler hyperparameters: {sched_args}")

    return GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
    print("Finished configuring Optimizer and Scheduler")



def call_pai_add_validation_score(args, validation_accuracy: float, model: torch.nn.Module):
    return GPA.pai_tracker.add_validation_score(validation_accuracy, model)
