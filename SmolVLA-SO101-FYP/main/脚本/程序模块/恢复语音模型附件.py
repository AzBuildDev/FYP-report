"""Verify and restore the two licensed speech models from GitHub Release parts. No network or hardware."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

MODEL_DIRS = {'Qwen2.5-1.5B-Instruct', 'whisper-tiny'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def restore(manifest_path, destination, check_only=False):
    manifest_path, destination = Path(manifest_path), Path(destination)
    spec = json.loads(manifest_path.read_text(encoding='utf-8'))
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported_manifest')
    files = spec['files']
    expected = {}
    for entry in files:
        name = entry['path']
        p = PurePosixPath(name)
        if ('\\' in name or p.is_absolute() or '..' in p.parts or len(p.parts) != 2
                or p.parts[0] not in MODEL_DIRS or name in expected):
            raise ValueError('invalid_or_duplicate_model_path')
        expected[name] = entry
    if {PurePosixPath(name).parts[0] for name in expected} != MODEL_DIRS:
        raise ValueError('both_models_required')
    if not check_only and any((destination / name).exists() for name in MODEL_DIRS):
        raise FileExistsError('model_directory_exists_refusing_to_overwrite')
    # Validate all source parts before writing a temporary archive.
    parts = []
    names = set()
    for entry in spec['archive']['parts']:
        name = entry['name']
        if Path(name).name != name or '/' in name or '\\' in name or name in names:
            raise ValueError('invalid_or_duplicate_part_name')
        names.add(name)
        part = manifest_path.parent / name
        if part.stat().st_size != entry['bytes'] or digest(part) != entry['sha256']:
            raise ValueError('part_integrity_mismatch: ' + name)
        parts.append(part)
    if not parts:
        raise ValueError('parts_required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='speech-restore-', dir=destination.parent) as scratch:
        scratch = Path(scratch)
        archive = scratch / 'verified.zip'
        with archive.open('xb') as target:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, target, 8 * 1024**2)
        if (archive.stat().st_size != spec['archive']['bytes']
                or digest(archive) != spec['archive']['sha256']):
            raise ValueError('archive_integrity_mismatch')
        staged = scratch / 'models'
        with zipfile.ZipFile(archive) as package:
            if len(package.namelist()) != len(expected) or set(package.namelist()) != set(expected):
                raise ValueError('archive_file_list_mismatch')
            for name, entry in expected.items():
                info = package.getinfo(name)
                if info.file_size != entry['bytes'] or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('invalid_archive_entry')
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
                if digest(target) != entry['sha256']:
                    raise ValueError('model_file_integrity_mismatch: ' + name)
        if not check_only:
            destination.mkdir(parents=True, exist_ok=True)
            for name in sorted(MODEL_DIRS):
                if (destination / name).exists():
                    raise FileExistsError('model_directory_appeared_refusing_to_overwrite')
                (staged / name).rename(destination / name)
    return {'status': 'verified' if check_only else 'restored', 'files': len(expected),
            'models': sorted(MODEL_DIRS), 'hardware_connected': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing manifest and all Release parts')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(restore(args.assets / 'speech-models-manifest.json',
                             root / '模型/语音指令', args.check_only), ensure_ascii=False))


if __name__ == '__main__':
    main()
