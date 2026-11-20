"""Parse the real Accelerate launch configuration without starting workers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import launch_spec


class AccelerateLaunchTests(unittest.TestCase):
    def test_real_parser_and_standard_launcher_generate_eight_workers_per_node(self):
        from accelerate.commands.launch import launch_command_parser, _validate_launch_command
        from accelerate.utils.launch import prepare_deepspeed_cmd_env, _filter_args
        import torch.distributed.run as torchrun
        for node in (0, 5):
            with self.subTest(node=node), tempfile.TemporaryDirectory() as folder:
                runtime = Path(folder)
                env = {'MA_NUM_HOSTS': '8', 'VC_TASK_INDEX': str(node),
                       'VC_WORKER_HOSTS': ','.join(f'worker-{i}' for i in range(8)),
                       'RUN_NAME': 'config-test', 'MODEL_DIR': '/cache/models/Qwen2.5-VL-3B-Instruct',
                       'DATA_ROOT': '/cache/data', 'OUTPUT_ROOT': '/cache/output',
                       'STARVLA_RUNTIME': str(runtime)}
                spec = launch_spec(env, Path('/code/A'))
                for name in ('accelerate', 'deepspeed'):
                    (runtime / f'{name}.json').write_text(json.dumps(spec[name]))
                args = launch_command_parser().parse_args(spec['command'][3:])
                args, defaults, precision_from_config = _validate_launch_command(args)
                fields = list(defaults.deepspeed_config)
                if precision_from_config:
                    fields.append('mixed_precision')
                args.deepspeed_fields_from_accelerate_config = ','.join(fields)
                with patch('accelerate.utils.launch.is_port_in_use', return_value=False):
                    command, child = prepare_deepspeed_cmd_env(args)
                self.assertIsNone(command)  # standard uses torchrun, not ssh/pdsh
                self.assertTrue(args.use_deepspeed)
                self.assertEqual(args.nproc_per_node, '8')
                self.assertEqual(args.nnodes, '8')
                self.assertEqual(args.node_rank, node)
                # Accelerate 1.11's argparse False default may shadow the
                # JSON same_network value. Both branches must target exactly
                # the same static rendezvous endpoint.
                if args.same_network:
                    self.assertEqual(args.master_addr, 'worker-0')
                    self.assertEqual(str(args.master_port), '28436')
                else:
                    self.assertEqual(args.rdzv_endpoint, 'worker-0:28436')
                self.assertEqual(args.rdzv_backend, 'static')
                self.assertEqual(child['ACCELERATE_USE_DEEPSPEED'], 'true')
                self.assertEqual(child['ACCELERATE_MIXED_PRECISION'], 'bf16')
                self.assertEqual(child['ACCELERATE_DEEPSPEED_CONFIG_FILE'], str(runtime / 'deepspeed.json'))
                self.assertEqual(child['ACCELERATE_DEEPSPEED_ZERO3_INIT'], 'false')
                self.assertEqual(args.training_script, str(Path('/code/A/starVLA/training/train_starvla.py')))
                self.assertEqual(args.training_script_args, ['--config_yaml', str(Path('/code/A/cann83/train.yaml'))])
                worker_args = _filter_args(args, torchrun.get_args_parser(),
                    ['--training_script', args.training_script, '--training_script_args', args.training_script_args])
                self.assertEqual(torchrun.get_rdzv_endpoint(worker_args), 'worker-0:28436')
                self.assertEqual(worker_args.node_rank, node)
                self.assertEqual(str(worker_args.nproc_per_node), '8')


if __name__ == '__main__':
    unittest.main(verbosity=2)
