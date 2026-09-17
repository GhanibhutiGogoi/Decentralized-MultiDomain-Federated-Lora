import hashlib
import json
import tarfile
from pathlib import Path

root = Path.home() / 'ahlora-runs/methodology-exploration-20260915'
source = Path.home() / 'ahlora-exploration-20260915'
directories = ['masked-screen-a', 'masked-screen-b', 'masked-replication-43', 'masked-replication-44', 'pooled-validation-replication', 'residual-screen-v1']
expected = {}
for name in directories:
    manifest = json.loads((root / name / 'manifest.json').read_text())
    for path, digest in manifest['source']['files_sha256'].items():
        fullpath = 'project-3-hierarchical-gossip/' + path
        if fullpath in expected:
            assert expected[fullpath] == digest, fullpath
        expected[fullpath] = digest
record = json.loads((root / 'masked-screen-b/adaptive_domain/seed42_adaptive_domain.json').read_text())
for key in ['rank_policy', 'domain_policy']:
    policy = record[key]
    for field in ['source', 'config_source', 'feature_source']:
        if field in policy:
            expected[policy[field]] = policy[field + '_sha256']
files = []
for name, digest in sorted(expected.items()):
    path = source / name
    assert path.is_relative_to(source) and path.is_file(), name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == digest, (name, actual, digest)
    files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest})
output = root / 'masked-summary'
with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
    for record in files:
        archive.add(source / record['path'], arcname=record['path'], recursive=False)
result = {'status': 'complete', 'exact_launch_sources_matched': True, 'files': files,
          'source_snapshot_sha256': hashlib.sha256((output / 'source_snapshot.tar.gz').read_bytes()).hexdigest(),
          'note': 'All six launch manifests matched, plus canonical imported P1/P2 source hashes. One AppleDouble ._ source file is retained solely because the launch provenance included it; it is not Python code.'}
(output / 'source_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': 'complete', 'source_files_verified': len(files)}))
