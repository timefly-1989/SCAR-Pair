#!/usr/bin/env python3
"""Package source and measured outputs with a verified SHA-256 manifest."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent
    project = root.parent
    output = root / 'output'
    validation = json.loads((output / 'validation_report.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Run validate_outputs.py successfully before packaging')
    publication_validation = json.loads((output / 'publication_validation_report.json').read_text())
    if publication_validation['status'] != 'passed':
        raise ValueError('Run publication_integrity.py successfully before packaging')
    manifest_path = output / 'reproduction_manifest.json'
    legacy_outputs = {
        'reader_statistics_report.json',
        'statistics_report.json',
    }
    files = sorted(set(
        [p for p in root.iterdir() if p.suffix in ('.py', '.md', '.txt')]
        + [p for p in output.iterdir()
           if p.suffix in ('.json', '.jsonl') and p != manifest_path
           and p.name not in legacy_outputs]
        + list((root / 'figure_data').glob('*.csv'))
        + [project / name for name in ('main.tex', 'supplementary.tex', 'results_macros.tex')]
        + [project / 'references' / 'references.bib', project / 'figures' / 'scar_styles.tex']
        + [project / 'scripts' / name for name in (
            'build_publication_tables.py', 'rebuild_manuscript_figures.py',
            'build_metrics_by_k.py', 'audit_manuscript_consistency.py')]
        + list((project / 'tables' / 'generated').glob('*.tex'))
    ))
    dependencies = sorted(
        [{'name': dist.metadata['Name'], 'version': dist.version}
         for dist in importlib.metadata.distributions()], key=lambda d: d['name'].lower())
    commits = {
        name: subprocess.check_output(
            ['git', '-C', str(root / 'vendor' / name), 'rev-parse', 'HEAD'], text=True).strip()
        for name in ('Mem2ActBench', 'locomo')
    }
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'installed_packages': dependencies, 'source_commits': commits,
        'source_data_audit': json.loads((output / 'data_audit.json').read_text()),
        'status': 'Current manuscript reconstruction suite completed and validated',
        'limitations': [
            'This executable public-data reconstruction is the source of the current manuscript results; it is not historical code from an earlier unreleased run.',
            'The reconstructed 19,661-conversation full-corpus setting is not directly interchangeable with the official 2,029-session benchmark setting.',
            'Capacity and graph controls use documented surrogate definitions.',
            'Inference scripts target Apple MPS; cross-device equivalence is untested.',
            'The archive excludes model weights, third-party datasets, intermediate caches and logs; source revisions and hashes are recorded.',
            'Source download revisions are pinned; the remote fallback path was not used in this run.',
            'Current manuscript tables use publication_* analyses; superseded task-level '
            'statistics reports are excluded from this archive.',
            'The conflict-of-interest declaration, persistent archive identifier, and final author confirmation of the AI-use disclosure remain pending.',
        ],
        'files': [{
            'path': p.relative_to(root.parent).as_posix(),
            'bytes': p.stat().st_size, 'sha256': digest(p),
        } for p in files],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    bundle = output / 'SCAR_Pair_reproduction_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.relative_to(root.parent).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed')
        for entry in manifest['files']:
            actual = hashlib.sha256(archive.read(entry['path'])).hexdigest()
            if actual != entry['sha256']:
                raise ValueError(f"Archive hash mismatch: {entry['path']}")
    checksum = digest(bundle)
    bundle.with_suffix('.zip.sha256').write_text(f'{checksum}  {bundle.name}\n')
    print(json.dumps({'bundle': str(bundle), 'files': len(files) + 1,
                      'bytes': bundle.stat().st_size, 'sha256': checksum,
                      'archive_and_manifest_verified': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
