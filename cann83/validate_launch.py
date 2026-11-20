"""Validate the real Accelerate CLI and DeepSpeed plugin without starting ranks.

Run in a separate process with the exact arguments passed to accelerate launch.
Only the final worker launcher is replaced; parsing, environment construction,
DeepSpeedPlugin validation and precision handling all use installed Accelerate.
"""
import importlib
import json
import os
import sys


def validate_cli(argv):
    from accelerate import DeepSpeedPlugin
    from accelerate.utils.launch import prepare_deepspeed_cmd_env

    cli = importlib.import_module('accelerate.commands.launch')
    result = {}

    def inspect_workers(args):
        command, child = prepare_deepspeed_cmd_env(args)
        if command is not None or args.deepspeed_multinode_launcher != 'standard':
            raise ValueError('Expected the standard node-local torchrun launcher')
        original = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(child)
            plugin = DeepSpeedPlugin()
            plugin.set_mixed_precision(child['ACCELERATE_MIXED_PRECISION'])
            config = plugin.deepspeed_config
            if config.get('bf16', {}).get('enabled') is not True or config.get('fp16', {}).get('enabled') is not False:
                raise ValueError('DeepSpeed must retain explicit BF16 enabled and FP16 disabled')
            if plugin.zero_stage != 2 or plugin.zero3_init_flag:
                raise ValueError('Expected ZeRO2 without ZeRO3 initialization')
            batch = config['train_batch_size']
            micro = config['train_micro_batch_size_per_gpu']
            accum = config['gradient_accumulation_steps']
            if batch != micro * accum * args.num_processes:
                raise ValueError('DeepSpeed batch size does not match the launch topology')
            if int(os.environ['STARVLA_GRAD_ACCUM']) != accum:
                raise ValueError('Trainer and DeepSpeed gradient accumulation differ')
            result.update(bf16=True, zero_stage=plugin.zero_stage, world_size=args.num_processes,
                          node_rank=args.machine_rank, micro_batch=micro,
                          gradient_accumulation_steps=accum, global_batch=batch,
                          accelerate_precision=child['ACCELERATE_MIXED_PRECISION'])
        finally:
            os.environ.clear()
            os.environ.update(original)

    def reject_other_launcher(*args, **kwargs):
        raise RuntimeError('The supplied configuration did not select DeepSpeed')

    names = ('deepspeed_launcher', 'multi_gpu_launcher', 'tpu_launcher',
             'tpu_pod_launcher', 'sagemaker_launcher', 'simple_launcher')
    original_launchers = {name: getattr(cli, name) for name in names}
    try:
        for name in names:
            setattr(cli, name, inspect_workers if name == 'deepspeed_launcher' else reject_other_launcher)
        cli.launch_command(cli.launch_command_parser().parse_args(argv))
    finally:
        for name, launcher in original_launchers.items():
            setattr(cli, name, launcher)
    if not result:
        raise RuntimeError('The supplied configuration did not select DeepSpeed')
    print('ACCELERATE_DEEPSPEED_CONFIG_PASSED', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    validate_cli(sys.argv[1:])
