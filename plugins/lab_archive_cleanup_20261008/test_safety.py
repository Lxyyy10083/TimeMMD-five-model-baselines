"""Meaningful deletion guards, using only isolated temporary fixtures."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('archive_remote', HERE/'remote.py')
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class Safety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.base = Path(self.temp.name).resolve()
        assert self.base.parent == HERE
        self.root = self.base/'legacy'
        self.root.mkdir()
        self.control = self.base/'control'
        self.control.mkdir()
        self.patches = [patch.object(remote,'ALLOWED',self.base), patch.object(remote,'TARGET',self.root),
            patch.object(remote,'CONTROL',self.control), patch.object(remote,'active_legacy_processes',return_value=[])]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        assert self.base.parent == HERE
        self.temp.cleanup()

    def receipt(self):
        p=self.root/'plugins/old/resume.pt'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'verified old optimizer state')
        asset=self.root/'models/aurora/checkpoint.pt'
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b'active pretrained asset')
        best=self.root/'plugins/best/checkpoint.pt'
        best.parent.mkdir(parents=True)
        best.write_bytes(b'best converged checkpoint')
        (self.control/'retention_plan.json').write_text(json.dumps({'keep_paths':['plugins/best/checkpoint.pt']}))
        data=remote.inventory()
        for item in data['entries']:
            if item['kind']=='file': item['sha256']=remote.digest(self.root/item['path'])
        kept=next(i for i in data['entries'] if i['path']=='plugins/best/checkpoint.pt')
        kept['cleanup']=False
        data.update(mode='best_version_only',metadata_verified=True,retained=[dict(kept)],git_commit='a'*40,local_archive='isolated test archive')
        return data,p,asset

    def test_only_verified_old_state_removed_assets_kept(self):
        receipt,p,asset=self.receipt()
        result=remote.verify_and_cleanup(receipt,True)
        self.assertEqual(result['deleted'],1)
        self.assertFalse(p.exists())
        self.assertEqual(asset.read_bytes(),b'active pretrained asset')
        self.assertEqual((self.root/'plugins/best/checkpoint.pt').read_bytes(),b'best converged checkpoint')

    def test_changed_source_refuses_deletion(self):
        receipt,p,asset=self.receipt();p.write_bytes(b'new state')
        with self.assertRaises(ValueError): remote.verify_and_cleanup(receipt,True)
        self.assertTrue(p.exists()); self.assertTrue(asset.exists())

    def test_github_and_local_verification_required(self):
        receipt,p,asset=self.receipt();receipt['git_commit']=''
        with self.assertRaises(ValueError): remote.verify_and_cleanup(receipt,True)
        self.assertTrue(p.exists())
        receipt['git_commit']='a'*40;receipt['metadata_verified']=False
        with self.assertRaises(ValueError): remote.verify_and_cleanup(receipt,True)
        self.assertTrue(p.exists())

    def test_traversal_and_models_are_rejected(self):
        for p in ['../resume.pt','/resume.pt','plugins/../../resume.pt']:
            with self.assertRaises(ValueError):remote.candidate(p)
        self.assertFalse(remote.candidate('models/aurora/checkpoint.pt'))
        self.assertFalse(remote.candidate('benchmark/readgpt_data/test.npz'))

    def test_protected_checkpoint_cannot_be_added_to_cleanup(self):
        receipt,p,asset=self.receipt()
        next(i for i in receipt['entries'] if i['path']=='plugins/best/checkpoint.pt')['cleanup']=True
        with self.assertRaises(ValueError):remote.verify_and_cleanup(receipt,True)
        self.assertTrue(p.exists())

    def test_missing_best_checkpoint_refuses_all_deletions(self):
        receipt,p,asset=self.receipt()
        (self.root/'plugins/best/checkpoint.pt').unlink()
        with self.assertRaises(FileNotFoundError):remote.verify_and_cleanup(receipt,True)
        self.assertTrue(p.exists())

    def test_symlink_candidate_cannot_escape_target(self):
        receipt,p,asset=self.receipt();item=next(i for i in receipt['entries'] if i.get('cleanup'))
        outside=self.base/'outside.pt';outside.write_bytes(b'must be preserved')
        p.unlink()
        try:p.symlink_to(outside)
        except OSError:self.skipTest('Local OS does not allow symlink creation')
        with self.assertRaises(ValueError):remote.check_entry(self.root,item)
        self.assertEqual(outside.read_bytes(),b'must be preserved')


if __name__=='__main__':unittest.main()
