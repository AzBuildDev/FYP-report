"""Small synthetic archives test restoration, corruption and traversal rejection without model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from 恢复语音模型附件 import restore, digest


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'models.zip'
        self.files = {'Qwen2.5-1.5B-Instruct/model.safetensors': b'test-qwen',
                      'whisper-tiny/model.safetensors': b'test-whisper'}
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in self.files.items():
                z.writestr(name, data)
        self.part = self.root / 'models.part01'
        self.part.write_bytes(self.archive.read_bytes())
        self.spec = {'schema_version': 1,
                     'files': [{'path': n, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                               for n, b in self.files.items()],
                     'archive': {'bytes': self.archive.stat().st_size, 'sha256': digest(self.archive),
                                 'parts': [{'name': self.part.name, 'bytes': self.part.stat().st_size,
                                            'sha256': digest(self.part)}]}}
        self.manifest = self.root / 'speech-models-manifest.json'
        self.destination = self.root / 'output/models'

    def run_restore(self, check=False):
        self.manifest.write_text(json.dumps(self.spec), encoding='utf8')
        return restore(self.manifest, self.destination, check)

    def test_roundtrip_and_no_overwrite(self):
        self.assertEqual(self.run_restore()['status'], 'restored')
        for name, data in self.files.items():
            self.assertEqual((self.destination/name).read_bytes(), data)
        with self.assertRaises(FileExistsError): self.run_restore()

    def test_check_only(self):
        self.assertEqual(self.run_restore(True)['status'], 'verified')
        self.assertFalse(self.destination.exists())

    def test_corrupt_part(self):
        self.part.write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.run_restore()

    def test_file_checksum(self):
        self.spec['files'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.run_restore()
        self.assertFalse(self.destination.exists())

    def test_archive_checksum(self):
        self.spec['archive']['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.run_restore()

    def test_traversal(self):
        self.spec['files'][0]['path'] = '../escape'
        with self.assertRaises(ValueError): self.run_restore()
        self.assertFalse((self.root/'escape').exists())

    def test_duplicate(self):
        self.spec['files'].append(self.spec['files'][0])
        with self.assertRaises(ValueError): self.run_restore()

    def test_part_traversal(self):
        self.spec['archive']['parts'][0]['name'] = '../models.part01'
        with self.assertRaises(ValueError): self.run_restore()


if __name__ == '__main__': unittest.main(verbosity=2)
