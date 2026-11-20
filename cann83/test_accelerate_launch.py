"""Exercise actual Accelerate dispatch and worker-side plugin configuration.

The DeepSpeed engine, distributed process group and NPU are not initialized.
Only the final launcher dispatch and port probe are substituted. The parser,
worker environment builder, DeepSpeedPlugin and precision handling are real.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import launch_spec


class AccelerateLaunchTests(unittest.TestCase):
    def spec(self, runtime, node=0):
        env = {'MA_NUM_HOSTS': '8', 'VC_TASK_INDEX': str(node),
               'VC_WORKER_HOSTS': ','.join(f'worker-{i}' for i in range(8)),
               'RUN_NAME': 'config-test', 'MODEL_DIR': '/cache/models/Qwen2.5-VL-3B-Instruct',
               'DATA_ROOT': '/cache/data', 'OUTPUT_ROOT': '/cache/output',
               'STARVLA_RUNTIME': str(runtime)}
        return launch_spec(env, Path('/code/A'))

    def dispatch(self, spec, runtime):
        from accelerate.commands.launch import launch_command_parser, launch_command
        from accelerate.utils.launch import prepare_deepspeed_cmd_env
        captured = {}
        for name in ('accelerate', 'deepspeed'):
            (runtime / f'{name}.json').write_text(json.dumps(spec[name]))
        # Let the real launch_command compute ACCELERATE_CONFIG_DS_FIELDS.
        # Recreating those fields in test code previously missed the crash.
        def capture(args):
            with patch('accelerate.utils.launch.is_port_in_use', return_value=False):
                command, child = prepare_deepspeed_cmd_env(args)
            captured.update(args=args, child=child, command=command)
        clean = {k: v for k, v in os.environ.items() if not k.startswith('ACCELERATE_')}
        with patch.dict(os.environ, clean, clear=True), \
             patch('accelerate.commands.launch.deepspeed_launcher', side_effect=capture) as launcher:
            launch_command(launch_command_parser().parse_args(spec['command'][3:]))
            launcher.assert_called_once()
        return captured

    def worker_plugin(self, child):
        from accelerate import DeepSpeedPlugin
        with patch.dict(os.environ, child, clear=True):
            plugin = DeepSpeedPlugin()  # Same call that failed at train_starvla.py:46.
            # AcceleratorState calls this after creating the plugin. "no" means
            # DeepSpeed owns precision; explicit bf16.enabled must remain True.
            plugin.set_mixed_precision(child['ACCELERATE_MIXED_PRECISION'])
        return plugin

    def assert_training_config(self, plugin):
        cfg = plugin.deepspeed_config
        self.assertIs(cfg['bf16']['enabled'], True)
        self.assertIs(cfg['fp16']['enabled'], False)
        self.assertEqual(plugin.zero_stage, 2)
        self.assertEqual(cfg['train_batch_size'], 64)
        self.assertEqual(cfg['train_micro_batch_size_per_gpu'], 1)
        self.assertEqual(cfg['gradient_accumulation_steps'], 1)
        self.assertEqual(plugin.gradient_accumulation_steps, 1)
        self.assertEqual(cfg['gradient_clipping'], 1.0)
        self.assertFalse(plugin.zero3_init_flag)
        self.assertFalse(cfg['zero_optimization']['elastic_checkpoint'])
        self.assertTrue(cfg['zero_optimization']['load_from_fp32_weights'])

    def test_real_dispatch_and_plugin_preserve_bf16_zero2_on_all_eight_nodes(self):
        from accelerate.utils.launch import _filter_args
        import torch.distributed.run as torchrun
        for node in range(8):
            with self.subTest(node=node), tempfile.TemporaryDirectory() as folder:
                runtime = Path(folder)
                spec = self.spec(runtime, node)
                result = self.dispatch(spec, runtime)
                args, child = result['args'], result['child']
                self.assertIsNone(result['command'])  # standard, no ssh/pdsh
                self.assertTrue(args.use_deepspeed)
                self.assertEqual(args.nproc_per_node, '8')
                self.assertEqual(args.nnodes, '8')
                self.assertEqual(args.node_rank, node)
                self.assertEqual(args.rdzv_backend, 'static')
                self.assertEqual(child['ACCELERATE_USE_DEEPSPEED'], 'true')
                self.assertEqual(child['ACCELERATE_MIXED_PRECISION'], 'no')
                self.assertNotIn('mixed_precision', child['ACCELERATE_CONFIG_DS_FIELDS'].split(','))
                self.assertEqual(child['ACCELERATE_DEEPSPEED_CONFIG_FILE'], str(runtime / 'deepspeed.json'))
                self.assertEqual(child['ACCELERATE_DEEPSPEED_ZERO3_INIT'], 'false')
                self.assertEqual(args.training_script, str(Path('/code/A/starVLA/training/train_starvla.py')))
                self.assertEqual(args.training_script_args, ['--config_yaml', str(Path('/code/A/cann83/train.yaml'))])
                worker_args = _filter_args(args, torchrun.get_args_parser(),
                    ['--training_script', args.training_script, '--training_script_args', args.training_script_args])
                self.assertEqual(torchrun.get_rdzv_endpoint(worker_args), 'worker-0:28436')
                self.assertEqual(worker_args.node_rank, node)
                self.assertEqual(str(worker_args.nproc_per_node), '8')
                self.assert_training_config(self.worker_plugin(child))

    def test_previous_bf16_accelerate_config_reproduces_reported_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            spec = self.spec(runtime)
            spec['accelerate']['mixed_precision'] = 'bf16'
            child = self.dispatch(spec, runtime)['child']
            self.assertIn('mixed_precision', child['ACCELERATE_CONFIG_DS_FIELDS'].split(','))
            with self.assertRaisesRegex(ValueError, 'When using `deepspeed_config_file`'):
                self.worker_plugin(child)

    def test_setting_accelerate_precision_to_no_is_not_a_valid_fix(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            spec = self.spec(runtime)
            spec['accelerate']['mixed_precision'] = 'no'
            child = self.dispatch(spec, runtime)['child']
            with self.assertRaisesRegex(ValueError, 'When using `deepspeed_config_file`'):
                self.worker_plugin(child)

    def test_emitted_environment_initializes_real_plugin_in_fresh_python(self):
        import accelerate
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            child = self.dispatch(self.spec(runtime, 7), runtime)['child']
            # Use the same Accelerate source being tested, regardless of the
            # installed distribution in this Python's site-packages.
            child['PYTHONPATH'] = str(Path(accelerate.__file__).resolve().parents[1]) + os.pathsep + child.get('PYTHONPATH', '')
            code = (
                "import json,os,accelerate; from accelerate import DeepSpeedPlugin; "
                "p=DeepSpeedPlugin(); p.set_mixed_precision(os.environ['ACCELERATE_MIXED_PRECISION']); "
                "c=p.deepspeed_config; "
                "print('PLUGIN_RESULT '+json.dumps({'accelerate':accelerate.__version__,"
                "'bf16':c['bf16']['enabled'],'fp16':c['fp16']['enabled'],'zero_stage':p.zero_stage,"
                "'batch':c['train_batch_size'],'micro':c['train_micro_batch_size_per_gpu'],"
                "'accumulation':p.gradient_accumulation_steps}))"
            )
            result = subprocess.run([sys.executable, '-c', code], env=child,
                                    text=True, encoding='utf-8', errors='replace',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line.removeprefix('PLUGIN_RESULT ')) for line in result.stdout.splitlines()
                       if line.startswith('PLUGIN_RESULT ')]
            self.assertEqual(records, [{'accelerate': accelerate.__version__, 'bf16': True,
                                       'fp16': False, 'zero_stage': 2, 'batch': 64,
                                       'micro': 1, 'accumulation': 1}])

    def test_production_probe_runs_in_fresh_python_before_workers(self):
        import accelerate
        from runtime_contract import clean_child_environment
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            spec = self.spec(runtime, 7)
            for name in ('accelerate', 'deepspeed'):
                (runtime / f'{name}.json').write_text(json.dumps(spec[name]))
            env = clean_child_environment(os.environ)
            env['STARVLA_GRAD_ACCUM'] = '1'
            env['PYTHONPATH'] = str(Path(accelerate.__file__).resolve().parents[1])
            probe = Path(__file__).with_name('validate_launch.py')
            result = subprocess.run([sys.executable, str(probe), *spec['command'][3:]],
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ACCELERATE_DEEPSPEED_CONFIG_PASSED', result.stdout)
            self.assertIn('"bf16": true', result.stdout)
            # A mismatch must fail this probe, not be discovered by 64 workers.
            env['STARVLA_GRAD_ACCUM'] = '2'
            bad = subprocess.run([sys.executable, str(probe), *spec['command'][3:]],
                                 env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('gradient accumulation differ', bad.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
