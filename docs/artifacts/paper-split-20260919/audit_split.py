"""Verify the split manuscripts and preserve their build identities on gpu003.

Run after scripts/build_research_papers.sh from a complete repository checkout.
An optional previously-reviewed/multidomain/rendered folder binds earlier review
images to a later rebuild. Visual inspection is recorded separately.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

if socket.gethostname().split('.')[0] != 'gpu003':
    raise SystemExit('Run paper validation on gpu003.')
root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
out = root / 'docs/artifacts/paper-split-20260919'
out.mkdir(parents=True, exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def local_dependency(folder, name, suffix=''):
    target = (folder / name).resolve()
    if not target.suffix and suffix:
        target = target.with_suffix(suffix)
    assert target.is_relative_to(folder.resolve()), f'Outside dependency: {name}'
    assert target.is_file(), f'Missing dependency: {name}'
    return str(target.relative_to(folder))

baseline = json.loads((root / 'docs/artifacts/quantity-skew/paper-build/build-record.json').read_text())
archive = root / 'paper/archive/combined-20260918'
archive_checks = {}
for name, expected in baseline['source_and_figure_sha256'].items():
    actual = digest(archive / name)
    assert actual == expected, f'Archive differs: {name}'
    archive_checks[name] = actual

record = {
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host': socket.gethostname(),
    'scope': 'Document build, source isolation, figure identity and archive checks only; no experiment rerun or statistical recomputation.',
    'render_dpi': 100,
    'archive_sha256': archive_checks,
    'papers': {},
    'visual_review_record': 'docs/artifacts/paper-split-20260919/VISUAL_REVIEW.md',
}
for study in ('multidomain', 'quantity-skew'):
    folder = root / 'paper' / study
    tex_files = sorted(p for p in folder.glob('*.tex') if not p.name.startswith('.'))
    source = '\n'.join(p.read_text() for p in tex_files)
    forbidden = r'SST-2|RoBERTa|Dec-LoRA' if study == 'multidomain' else r'CIFAR|ResNet'
    assert not re.search(forbidden, source, re.I), f'Cross-study content found in {study}'
    dependencies = []
    for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', source):
        dependencies.append(local_dependency(folder, name, '.tex'))
    graphics = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source)
    dependencies += [local_dependency(folder, name) for name in graphics]
    for names in re.findall(r'\\bibliography\{([^}]+)\}', source):
        dependencies += [local_dependency(folder, name, '.bib') for name in names.split(',')]
    log = (folder / 'main.log').read_text()
    warnings = [line for line in log.splitlines() if re.search(r'Overfull|Underfull|undefined|Warning', line)]
    assert not warnings, (study, warnings)
    body = (folder / 'build/main.txt').read_text()
    assert '??' not in body, f'Unresolved text references in {study}'
    info = subprocess.check_output(['pdfinfo', str(folder / 'main.pdf')], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    renders = sorted((folder / 'rendered').glob('main-*.png'))
    assert len(renders) == pages
    bib_files = sorted(p for p in folder.glob('*.bib') if not p.name.startswith('.'))
    hashes = {str(p.relative_to(folder)): digest(p) for p in tex_files + bib_files + [folder / 'main.pdf']}
    figures = {}
    for asset in sorted((folder / 'figures').iterdir()):
        if asset.name.startswith('.') or asset.suffix not in ('.pdf', '.png', '.json'):
            continue
        actual = digest(asset)
        canonical = root / 'paper/figures' / asset.name
        assert actual == digest(canonical), f'Figure differs: {asset}'
        hashes[str(asset.relative_to(folder))] = actual
        figures[str(asset.relative_to(folder))] = str(canonical.relative_to(root))
    rendered_hashes = {p.name: digest(p) for p in renders}
    earlier = root / 'previously-reviewed' / study / 'rendered'
    matches = {}
    if earlier.is_dir():
        matches = {p.name: digest(earlier / p.name) == rendered_hashes[p.name] for p in renders}
        assert all(matches.values()), f'Render changed since review: {study}'
    log_dir = out / study
    log_dir.mkdir(exist_ok=True)
    for filename in ('pass1.log', 'bibtex.log', 'pass2.log', 'pass3.log', 'pdfinfo.txt', 'main.pdf.sha256'):
        shutil.copy2(folder / 'build' / filename, log_dir / filename)
    shutil.copy2(folder / 'main.log', log_dir / 'latex.log')
    record['papers'][study] = {
        'source': str((folder / 'main.tex').relative_to(root)),
        'pdf': str((folder / 'main.pdf').relative_to(root)),
        'pages': pages,
        'source_and_asset_sha256': hashes,
        'local_dependencies': sorted(set(dependencies)),
        'figure_copies_match_canonical': figures,
        'included_figure_count': len(graphics),
        'latex_layout_reference_warnings': warnings,
        'rendered_page_sha256': rendered_hashes,
        'previous_review_render_matches': matches,
        'self_contained_sources': True,
        'cross_study_content_check': 'passed',
    }
(out / 'build-record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'host':record['host'], 'papers':{k:{'pages':v['pages'], 'figures':v['included_figure_count'], 'pdf_sha256':v['source_and_asset_sha256']['main.pdf']} for k,v in record['papers'].items()}, 'archive_verified_files':len(archive_checks)}, indent=2))
