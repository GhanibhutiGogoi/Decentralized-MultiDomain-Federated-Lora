#!/usr/bin/env python3
"""Replot exact historical P1 final endpoints on gpu003; no inferred round curves."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'gpu003', 'Execute on the designated experiment host'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with args.input.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads(args.manifest.read_text())
    names = ['CIFAR-CNN', 'Fashion-MLP', 'AGNews-LSTM', 'Tabular-MLP', 'Audio-1DCNN']
    assert [r['Experiment'] for r in rows] == names
    assert manifest['seeds'] == [42] and manifest['rounds'] == 5 and manifest['fixed_rank'] == 32
    fixed = [float(r['Fixed Final Acc (%)']) for r in rows]
    adaptive = [float(r['Adaptive Final Acc (%)']) for r in rows]
    for row, f, a in zip(rows, fixed, adaptive):
        assert 0 <= f <= 100 and 0 <= a <= 100
        assert abs(a - f - float(row['Accuracy Delta (%)'])) < 1e-9
        assert int(row['Fixed Rank']) == 32
        assert float(row['Average Adaptive Rank']) == 2
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':12,
                         'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':11,
                         'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    figure, axis = plt.subplots(figsize=(7.6,4.6))
    positions = np.arange(len(names))
    for offset, scores, color, label in [(-.17, fixed, '#D55E00', 'Fixed rank 32 (exceeds caps)'),
                                         (.17, adaptive, '#0072B2', 'Original adaptive policy (rank 2)')]:
        axis.barh(positions+offset, scores, height=.30, color=color, label=label)
        for pos, score in zip(positions+offset, scores):
            axis.text(score+1.1, pos, f'{score:.2f}', va='center', fontsize=10)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(0,108)
    axis.set_xticks([0,20,40,60,80,100])
    axis.set_xlabel('Final accuracy (%)')
    axis.set_title('Historical Project 1: five tasks, five rounds, seed 42', pad=44)
    axis.grid(axis='x', alpha=.15)
    axis.set_axisbelow(True)
    axis.legend(loc='lower left', bbox_to_anchor=(0,1.01), frameon=False, fontsize=10)
    figure.text(.5,.005,'Single-seed endpoints; no uncertainty intervals. The revised controller is reported separately.',
                ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.05,1,1))
    stem='p1_adaptive_final_accuracy_readable'
    for suffix in ['pdf','png']:
        figure.savefig(args.output/f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(figure)
    data={'tasks':names,'fixed_rank32_final_accuracy_percent':fixed,'original_adaptive_final_accuracy_percent':adaptive,
          'seed':42,'rounds':5,'scope':'Original controller, not revised controller; fixed rank32 exceeds all declared caps; one seed, no uncertainty intervals',
          'source_csv_sha256':sha(args.input),'source_manifest_sha256':sha(args.manifest),
          'note':'Only final endpoints are plotted from exact CSV fields; original round-curve figures remain in the historical archive.'}
    (args.output/f'{stem}.data.json').write_text(json.dumps(data,indent=2)+'\n')
    provenance={'host':socket.gethostname(),'generated_utc':datetime.now(timezone.utc).isoformat(),
                'script_sha256':sha(Path(__file__)),'source_csv':str(args.input),'source_csv_sha256':sha(args.input),
                'source_manifest_sha256':sha(args.manifest),'matplotlib':matplotlib.__version__,
                'output_sha256':{f.name:sha(f) for f in args.output.glob(stem+'.*')},'validation':'Five exact endpoints; recorded differences and rank assignments checked; no new training or inferred round data'}
    (args.output/'p1-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':
    main()
