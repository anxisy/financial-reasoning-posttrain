"""Offline unit tests; no GPU, training, vLLM or network required."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_runner', ROOT/'tools'/'portable_runner.py')
mod=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)

class PortableTests(unittest.TestCase):
    def test_all_tasks_supported_and_stay_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf={k: Path(v) for k,v in mod.default_config(Path(tmp)).items()}
            for task in mod.TASKS:
                plan=mod.build_plan(task,conf, limit=5)
                self.assertTrue(plan.command[0]==sys.executable)
                self.assertTrue(plan.cwd.is_dir(),plan.cwd)
                self.assertFalse(plan.output.is_relative_to(mod.REPO))
                self.assertIn('5',plan.command) if (task.endswith(('dev','test','oracle','self'))) else None

    def test_checks_only_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf={k: Path(v) for k,v in mod.default_config(Path(tmp)).items()}
            p=mod.build_plan('conv-transfer-smoke',conf)
            missing=mod.validate(p)
            self.assertTrue(any('original_sft_adapter' in s for s in missing))
            self.assertTrue(any('conv_sft_train' in s for s in missing))

    def test_new_data_output_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf={k: Path(v) for k,v in mod.default_config(Path(tmp)).items()}
            p=mod.build_plan('replay-build',conf)
            p.output.mkdir(parents=True)
            self.assertTrue(any('OUTPUT EXISTS' in s for s in mod.validate(p,reject_existing=True)))
            pv=mod.build_plan('replay-verify',conf)
            self.assertFalse(any('OUTPUT EXISTS' in s for s in mod.validate(pv,reject_existing=True)))

    def test_config_invalid_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'paths.json'
            obj=mod.default_config(Path(tmp)); obj['base_model']='relative/model'
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):mod.load_config(p)

    def test_self_history_tasks_use_separate_adapter_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf={k: Path(v) for k,v in mod.default_config(Path(tmp)).items()}
            for v in ('transfer','direct','replay'):
                p=mod.build_plan('conv-'+v+'-self',conf)
                self.assertIn('--variant',p.command)
                self.assertIn(v,p.command)
                self.assertIn(str(conf[v+'_adapter']),p.command)

if __name__=='__main__': unittest.main()
