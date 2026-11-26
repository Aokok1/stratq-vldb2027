"""Render the evaluation figures from the saved per-trial records."""
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
BLUE, GRAY, ORANGE = '#136497', '#535b66', '#a45500'
SOURCES = {}


def apply_figure_style():
    mpl.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
                         'axes.titlelocation': 'left', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.65,
                         'legend.fontsize': 8, 'legend.frameon': False,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'xtick.direction': 'out', 'ytick.direction': 'out',
                         'pdf.fonttype': 42, 'figure.dpi': 180, 'savefig.dpi': 300})


def load(name):
    path = ROOT / 'results' / f'{name}.json'
    data = path.read_bytes()
    SOURCES[name] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def paired(runs, proxy_key, seeds):
    values = {}
    for state in (False, True):
        subset = [row for row in runs if proxy_key(row) == state]
        if sorted(row['seed'] for row in subset) != list(seeds):
            raise ValueError('Unexpected pairing or duplicate trials')
        values[state] = {row['seed']: row['calls'] for row in subset}
    a, b = ([values[state][seed] for seed in seeds] for state in (False, True))
    if min(a) <= 0:
        raise ValueError('Paired ratios require positive no-proxy calls')
    return {'seeds': list(seeds), 'no_proxy_calls': a, 'proxy_calls': b,
            'mean_calls_change_pct': 100 * (mean(b) / mean(a) - 1),
            'per_seed_change_pct': [100 * (y / x - 1) for x, y in zip(a, b)]}


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
             if t.get_visible() and t.get_text().strip()]
    for i, (text, box) in enumerate(boxes):
        if not (fig.bbox.contains(*box.min) and fig.bbox.contains(*box.max)):
            raise ValueError(f'{name}: cropped text {text!r}')
        for other, rect in boxes[i + 1:]:
            if box.overlaps(rect):
                raise ValueError(f'{name}: overlap {text!r}, {other!r}')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)


def proxy_figure():
    adult = load('adult_count_replay')['runs']
    sms = load('sms_live')['runs']
    fashion = load('fashion_live')['runs']
    scale = load('scale_10000000')['runs']
    rows = {
        'Adult': paired([r for r in adult if r['mode'] in ('no_proxy', 'proxy_only')],
                        lambda r: r['mode'] == 'proxy_only', range(1000, 1030)),
        'SMS': paired([r for r in sms if r['query'] == 'count'], lambda r: r['proxy'], range(2601, 2611)),
        'Fashion-MNIST': paired(fashion, lambda r: r['proxy'], range(2701, 2706)),
        'Synthetic 10M': paired(scale, lambda r: r['use_proxy'], range(2401, 2404)),
    }
    fig, ax = plt.subplots(figsize=(3.33, 2.12), layout='constrained')
    ax.axvline(0, color='#949aa1', linewidth=.7, linestyle='--', zorder=0)
    for y, (name, row) in enumerate(rows.items()):
        points = row['per_seed_change_pct']
        spread = np.linspace(-.13, .13, len(points))
        ax.scatter(sorted(points), y + spread, color=BLUE, alpha=.48, s=11,
                   linewidths=0, label='Seed pair' if y == 0 else None)
        ax.scatter(row['mean_calls_change_pct'], y, marker='D', facecolor='white', edgecolor=BLUE,
                   linewidth=1.25, s=28, label='Ratio of means' if y == 0 else None)
    ax.set_yticks(range(len(rows)), list(rows))
    ax.set_ylim(3.45, -.5)
    ax.set_xticks([-50, -40, -30, -20, -10, 0, 10])
    ax.set(xlabel='Reference-call change with proxy (%)', xlim=(-52, 13))
    if any(not -51 <= point <= 12 for row in rows.values() for point in row['per_seed_change_pct']):
        raise ValueError('Paired observations exceed the plotted range')
    ax.legend(loc='upper left', handletextpad=.25, labelspacing=.25)
    save(fig, 'proxy_calls')
    return rows


def fixed_figure():
    data = load('adult_fixed_budget')
    methods = [('proxy_stratified_fixed', 'Two-stage proxy strata', ORANGE, 's'),
               ('stratq', 'StratQ', BLUE, 'o')]
    truth = data['truth']
    budgets = (2048, 4096, 8192)
    fig, ax = plt.subplots(figsize=(3.33, 2.12), layout='constrained')
    values = {}
    for index, (method, label, color, marker) in enumerate(methods):
        means = []
        by_budget = {}
        for j, budget in enumerate(budgets):
            rs = [r for r in data['runs'] if (r['budget'], r['method']) == (budget, method)]
            if sorted(r['seed'] for r in rs) != list(range(1000, 1030)):
                raise ValueError('Unexpected fixed-budget trial set')
            ys = [100 * max((v['upper'] - v['lower']) / (2 * truth[g]['size'])
                            for g, v in r['groups'].items()) for r in rs]
            means.append(mean(ys))
            by_budget[budget] = ys
            ax.scatter(j + (index - .5) * .14 + np.linspace(-.035, .035, len(ys)), sorted(ys),
                       color=color, alpha=.4, s=8, linewidths=0)
        ax.plot(np.arange(len(budgets)) + (index - .5) * .14, means, color=color, marker=marker,
                markersize=4, label=label)
        values[method] = {'mean_max_rate_half_width_pct': means, 'per_seed_pct': by_budget}
    ax.set_xticks(range(len(budgets)), ['2,048', '4,096', '8,192'])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set(xlabel='Reference-call budget', ylabel='Largest group half-width\n(% of group size)', ylim=(0, 26))
    if any(not 0 <= point <= 25 for method in values.values()
           for points in method['per_seed_pct'].values() for point in points):
        raise ValueError('Fixed-budget observations exceed the plotted range')
    ax.legend(loc='upper right', handlelength=1.4)
    save(fig, 'adult_fixed_budget')
    return values


def scale_figure():
    reports = [load(f'scale_{n}') for n in (100_000, 1_000_000, 10_000_000)]
    stages = [(r['rows'], r['profile_ms'] / 1000, r['peak_working_set_bytes'] / 2**30) for r in reports]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.08), layout='constrained')
    for ax, i, label, ticks in [(axes[0], 1, 'Profile construction (s)', [.1, 1, 10]),
                                 (axes[1], 2, 'Peak working set (GiB)', [.1, .3, 1])]:
        ax.plot(range(3), [x[i] for x in stages], color=GRAY, marker='o', markersize=4)
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticks(ticks, [f'{tick:g}' for tick in ticks])
        ax.set_xticks(range(3), ['100k', '1M', '10M'])
        ax.set(xlabel='Parquet rows', ylabel=label)
        ax.margins(x=.17, y=.14)
    save(fig, 'parquet_scale')
    return [{'rows': n, 'profile_seconds': time, 'peak_gib': mem} for n, time, mem in stages]


def main():
    OUT.mkdir(exist_ok=True)
    apply_figure_style()
    metrics = {'proxy_calls': proxy_figure(), 'adult_fixed_budget': fixed_figure(), 'parquet_scale': scale_figure()}
    (OUT / 'figure_metrics.json').write_text(json.dumps({'source_sha256': SOURCES, 'metrics': metrics},
                                                         indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
