"""Build the deck's animated slide builds as mp4.

Renders one mp4 per animated slide into this folder. Nothing here writes to
`figures/` - the static PNGs remain the source of truth for the printed deck,
and these videos are drop-in replacements for those images on the five slides
that carry a build.

Slides 3, 9 and 10 reuse the plotting code and data from notebooks 02, 05 and
04, so their final frame is the static figure. Slides 12 and 15 are hand-built
graphics with no notebook source, so they are reconstructed here - which also
makes them reproducible for the first time.

Usage
-----
    python animations/make_animations.py            # all five
    python animations/make_animations.py 10 12      # just those slides

In PowerPoint: Insert > Video > This Device, then set Start: On Click and leave
Loop and "Rewind after playing" unchecked so the video holds its final frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import StrMethodFormatter
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import imageio_ffmpeg

mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------- config ----
# Durations in seconds. Sized to the number of reveals, not to slide length:
# a build that finishes early leaves a complete figure to talk over, one that
# lags makes the presenter wait. Tune here and re-render.
DURATION = {
    'slide03': 7.0,    # 2 beats: measured series -> dashed projection
    'slide08': 10.0,   # 42 beats: links merge in Ward distance order
    'slide09': 8.0,    # 4 beats: bars bottom-up, 315 -> 764
    'slide10': 8.0,    # 3 beats: left panel -> right panel -> callouts
    'slide12': 12.0,   # 6 beats: the SB 1383 reroute, start to finish
    'slide15': 9.0,    # 4 beats: beef/peas + 66x -> three action cards
}

# Longer cuts of the same build, for slides that get more airtime. The whole
# schedule is stretched uniformly to fill the target - beats, motion and pauses
# alike - so the pacing above is preserved in proportion and the build finishes
# just before the video ends. These are written alongside the base version with
# a _NNs suffix; the base file is never overwritten.
VARIANTS = {
    'slide03': (20, 30, 40),
    'slide08': (10, 20, 30, 40),
    'slide09': (20, 30, 40),
    'slide10': (20, 30, 40),
    'slide12': (20, 30, 40),
    'slide15': (20, 30, 40),
}

FPS = 30
FIGSIZE = (12.8, 7.2)   # 16:9, matches the slide
FIGSIZE_TALL = (8.5, 10.0)   # slide 8 keeps the static figure's portrait shape
DPI = 150               # -> 1920x1080

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'animations'
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------- house palette ------
# Identical to the notebooks so the videos sit in the same visual system.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
SUB = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'
BLUE_DEEP = '#104281'
BLUE_PALE = '#e8f0fb'
GREEN = '#008300'
MAGENTA = '#e87ba4'
RED = '#b03a2e'

mpl.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    # Must not be 'tight': every frame of a video has to be the same size.
    'savefig.bbox': None,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
    'text.color': INK,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


# ------------------------------------------------------------- helpers ------
def ease(t: float) -> float:
    """Smoothstep. Motion that starts and stops gently reads as deliberate;
    linear motion reads as a progress bar."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def beat(t: float, start: float, dur: float) -> float:
    """Eased 0->1 progress of a beat starting at `start` and lasting `dur`."""
    if dur <= 0:
        return 1.0 if t >= start else 0.0
    return ease((t - start) / dur)


def partial_line(xs, ys, p: float):
    """The first `p` of a polyline, interpolating within the final segment so
    the line grows smoothly rather than snapping point to point."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if p <= 0:
        return np.array([]), np.array([])
    total = len(xs) - 1
    f = p * total
    k = int(np.floor(f))
    if k >= total:
        return xs, ys
    frac = f - k
    return (np.append(xs[:k + 1], xs[k] + (xs[k + 1] - xs[k]) * frac),
            np.append(ys[:k + 1], ys[k] + (ys[k + 1] - ys[k]) * frac))


def style_barh(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(AXIS)
    ax.tick_params(length=0)


def fmt1(v: float) -> str:
    return f'{v:,.0f}' if abs(v - round(v)) < 0.05 else f'{v:,.1f}'


def render(name: str, fig, update, base: float, seconds: float = None):
    """Drive `update` over `seconds` of wall clock and write an mp4.

    `base` is the duration the builder's schedule was written against. When
    `seconds` differs, time is scaled so the same schedule fills the longer
    run: every beat, motion and pause stretches by the same factor, which
    keeps the tuned proportions and lands the final frame just before the end.
    """
    seconds = base if seconds is None else seconds
    path = OUT / name
    n = int(round(seconds * FPS))
    warp = base / seconds

    # Height is pinned to 1080 and the width follows the figure's aspect, so a
    # 16:9 build lands at 1920x1080 and the portrait dendrogram at 918x1080.
    dpi = 1080 / fig.get_figheight()
    w, h = int(round(fig.get_figwidth() * dpi)), 1080
    if w % 2 or h % 2:
        raise ValueError(f'{name}: {w}x{h} has an odd dimension; H.264 needs even')

    writer = FFMpegWriter(
        fps=FPS, codec='libx264',
        # yuv420p is what PowerPoint and QuickTime will actually decode.
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow'],
    )
    with writer.saving(fig, str(path), dpi=dpi):
        for i in range(n + 1):
            update(i / FPS * warp)
            writer.grab_frame()
    plt.close(fig)
    print(f'  wrote {path.name:46s} {seconds:5.1f}s  {w}x{h}  '
          f'{n + 1:5d} frames  {path.stat().st_size / 1e6:5.2f} MB')


# =========================================================== slide 3 ========
def build_slide03():
    """Measured 1990-2023 draws in, then the dashed projection extends to 2043.

    Splitting the reveal is the point: it puts a beat between what was measured
    and what is extrapolated, which is the same distinction the dashed line is
    already making.
    """
    raw = pd.read_csv(DATA / 'ghg-emissions.csv')
    unit = str(pd.unique(raw['unit'])[0])
    emissions = raw.drop(columns=['unit']).set_index('Sector')
    emissions.columns = emissions.columns.astype(int)

    obs = emissions.loc['Agriculture']
    X = obs.index.to_numpy().reshape(-1, 1)
    y = obs.to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    last = int(obs.index.max())
    future_years = np.arange(last, last + 21)
    future = model.predict(future_years.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.135)

    line_obs, = ax.plot([], [], color=BLUE, linewidth=2.2,
                        solid_capstyle='round', zorder=3)
    line_proj, = ax.plot([], [], color=GREEN, linewidth=2.2,
                         linestyle=(0, (6, 4)), dash_capstyle='round', zorder=3)

    lab_obs = ax.annotate('Observed', xy=(2004, obs.loc[2004]), xytext=(0, 16),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=11, fontweight=600, color=BLUE, alpha=0)
    lab_proj = ax.annotate('Projected (linear fit)',
                           xy=(2035, float(model.predict([[2035]])[0])),
                           xytext=(0, -42), textcoords='offset points',
                           ha='center', va='top', fontsize=11, fontweight=600,
                           color=GREEN, alpha=0)

    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    ax.spines['left'].set_color(AXIS)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(length=0)
    lo, hi = min(y.min(), future.min()), max(y.max(), future.max())
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(min(obs.index) - 1.5, future_years[-1] + 1.5)
    ax.set_xticks(range(1990, future_years[-1] + 1, 10))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='both', labelsize=11)

    fig.text(0.012, 0.965,
             f'Linear Regression Predicting Agriculture GHG Emissions 2023-2043',
             va='top', ha='left', fontsize=15, fontweight=600, color=INK)
    fig.text(0.012, 0.905,
             f'GHG emissions in million tonnes CO2e ({unit})',
             va='top', ha='left', fontsize=10.5, color=SUB)
    fig.text(0.012, 0.045,
             'Source: Climate Watch, global historical GHG emissions by sector. Projection is an '
             f'ordinary-least-squares fit on {min(obs.index)}-{last} (R² = {r2:.2f}), extended to '
             f'{future_years[-1]};\nit assumes the past trend continues and models no policy or '
             'technology change.',
             va='top', ha='left', fontsize=8.5, color=MUTED)

    D = DURATION['slide03']

    def update(t):
        p1 = beat(t, 0.4, 2.4)                 # observed draws
        line_obs.set_data(*partial_line(obs.index.to_numpy(), y, p1))
        lab_obs.set_alpha(beat(t, 2.0, 0.7))
        p2 = beat(t, 3.4, 2.0)                 # projection extends
        line_proj.set_data(*partial_line(future_years, future, p2))
        lab_proj.set_alpha(beat(t, 4.6, 0.7))

    return fig, update, D


# =========================================================== slide 8 ========
def build_slide08():
    """The tree assembles itself, nearest pair first.

    Links appear in ascending Ward distance, which is the order the algorithm
    actually merges in: tight pairs snap together early, the three families
    resolve in the middle, and the grey links that join different families
    arrive last - which is the slide's whole point. Leaf labels and axes are
    present from frame 0 so nothing reflows.

    Ward linkage is deterministic, so this recomputes the published figure
    exactly; it does not touch the k-means step whose numbers are validated.
    """
    df = pd.read_csv(DATA / 'Food_Product_Emissions.csv')
    STAGES = ['Land Use Change', 'Feed', 'Farm', 'Processing',
              'Transport', 'Packaging', 'Retail']
    X = StandardScaler().fit_transform(df[STAGES].to_numpy())
    Z = linkage(X, method='ward')
    cut3 = fcluster(Z, t=3, criterion='maxclust')

    # Colour each link by the 3-cluster cut, exactly as notebook 03 does:
    # a link inherits a family colour only if both children sit in the same
    # family, otherwise it is grey.
    pos = {f: i for i, f in enumerate(df['Food product'])}
    ward_color = {int(cut3[pos['Beef (beef herd)']]): BLUE,
                  int(cut3[pos['Palm Oil']]): MAGENTA}
    for c in set(cut3):
        ward_color.setdefault(int(c), GREEN)

    n = len(df)
    cluster_of = {i: int(cut3[i]) for i in range(n)}
    link_cols = {}
    for i, (a, b_, *_rest) in enumerate(Z):
        ca, cb = cluster_of.get(int(a)), cluster_of.get(int(b_))
        same = ca if (ca == cb and ca is not None) else None
        cluster_of[n + i] = same
        link_cols[n + i] = ward_color[same] if same is not None else AXIS

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    fig.subplots_adjust(left=0.225, right=0.975, top=0.873, bottom=0.093)

    dendrogram(Z, labels=df['Food product'].tolist(), orientation='right',
               link_color_func=lambda nid: link_cols.get(nid, AXIS),
               ax=ax, leaf_font_size=9)

    for lbl in ax.get_ymajorticklabels():
        lbl.set_color(SUB)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Ward linkage distance (standardized absolute stage space)')
    ax.set_axisbelow(True)
    ax.grid(True, axis='x', color=GRID, linewidth=0.8)
    ax.grid(False, axis='y')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)

    ax.set_title('A different algorithm finds the same three families',
                 loc='left', fontsize=13, fontweight=600, color=INK, pad=44)
    ax.text(0, 1.02,
            'Ward hierarchical clustering on the 7 standardized absolute stage-level '
            'features (kg CO2e per kg,\nglobal averages, 43 foods), colored by the '
            '3-cluster cut; gray links join different families',
            transform=ax.transAxes, fontsize=9.5, color=SUB, va='bottom')
    fig.text(0.01, 0.03, 'Source: Poore & Nemecek (2018), via Our World in Data',
             fontsize=8.5, color=MUTED, ha='left')

    # scipy draws one LineCollection per colour; animate per-segment alpha by
    # rewriting each collection's RGBA array every frame.
    layers = []
    heights_all = []
    for coll in ax.collections:
        segs = coll.get_segments()
        h = np.array([s[:, 0].max() for s in segs])     # x is distance here
        layers.append([coll, h, np.array(to_rgba(coll.get_colors()[0]))])
        heights_all.append(h)
    order = np.sort(np.concatenate(heights_all))
    n_links = len(order)
    if n_links != len(Z):
        raise RuntimeError(f'expected {len(Z)} links, collected {n_links}')

    T0, SPAN, FADE = 0.35, 8.0, 0.35
    step = SPAN / max(n_links - 1, 1)
    for layer in layers:
        layer.append(T0 + np.searchsorted(order, layer[1]) * step)   # start times

    D = DURATION['slide08']

    def update(t):
        for coll, _h, rgba0, starts in layers:
            a = np.clip((t - starts) / FADE, 0.0, 1.0)
            a = a * a * (3.0 - 2.0 * a)
            rgba = np.tile(rgba0, (len(starts), 1))
            rgba[:, 3] = a
            coll.set_color(rgba)

    return fig, update, D


# =========================================================== slide 9 ========
def _playbook_scenarios():
    per_kg = pd.read_csv(DATA / 'Food_Product_Emissions.csv')
    ltr = per_kg.set_index('Food product')['Total from Land to Retail']
    FOUR_OZ_KG = 4 * 28.349523125 / 1000
    BEEF_KG_PER_YR = 26.35
    LEGUME = 'Other Pulses'
    diff_chicken = ltr['Beef (beef herd)'] - ltr['Poultry Meat']
    diff_legumes = ltr['Beef (beef herd)'] - ltr[LEGUME]
    weekly = FOUR_OZ_KG * 52
    half = BEEF_KG_PER_YR / 2
    s = pd.DataFrame({
        'label': ['One 4-oz beef meal per week\n-> chicken',
                  'One 4-oz beef meal per week\n-> legumes',
                  'Half of annual beef (13.2 kg)\n-> chicken',
                  'Half of annual beef (13.2 kg)\n-> legumes'],
        'kg': [weekly, weekly, half, half],
        'per_kg': [diff_chicken, diff_legumes, diff_chicken, diff_legumes],
    })
    s['value'] = s['kg'] * s['per_kg']
    return s.sort_values('value').reset_index(drop=True)


def build_slide09():
    """Four bars bottom-up, smallest commitment first, so each number lands as
    it is spoken instead of the audience jumping straight to 764."""
    d = _playbook_scenarios()
    vals = d['value'].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.24, right=0.955, top=0.79, bottom=0.15)

    ypos = np.arange(len(d))
    bars = ax.barh(ypos, np.zeros_like(vals), height=0.55, color=BLUE)
    ax.set_yticks(ypos, d['label'])

    style_barh(ax)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlim(0, 850)
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel('Estimated footprint difference (kg CO2e per person per year)',
                  fontsize=10.5)

    labels = [ax.text(v + 12, i, f'{v:,.0f} kg', va='center', fontsize=11,
                      color=SUB, alpha=0) for i, v in enumerate(vals)]

    fig.text(0.012, 0.975,
             'Recurring substitutions add up: one beef meal a week substituted\n'
             'is an estimated 300+ kg CO2e per year',
             va='top', fontsize=16, fontweight=600, color=INK)
    fig.text(0.012, 0.845,
             'Estimated production-footprint difference under a 1:1 substitution, kg CO2e per '
             'person per year - global-average land-to-retail intensities',
             fontsize=11, color=SUB)
    fig.text(0.012, 0.012,
             'Source: Poore & Nemecek (2018) via Our World in Data (global averages, '
             'land-to-retail totals). US average beef consumption 26.35 kg retail weight/person/yr\n'
             '(USDA ERS, 2023); 4-oz serving = 0.113 kg; 52 weekly meals/yr.',
             fontsize=9, color=MUTED)

    D = DURATION['slide09']
    step = 1.55   # one bar per beat, then a hold on the finished chart

    def update(t):
        for i, (bar, lab) in enumerate(zip(bars, labels)):
            start = 0.35 + i * step
            bar.set_width(vals[i] * beat(t, start, 0.85))
            lab.set_alpha(beat(t, start + 0.7, 0.45))

    return fig, update, D


# ========================================================== slide 10 ========
def build_slide10():
    """Left panel, then right panel, then the callouts.

    The argument is the rank flip between the two measures - Farm 2nd by
    tonnage but last by GHG, Foodservice 4th by tonnage but 2nd by GHG - and it
    is invisible when both panels arrive at once. Axes and panel headers are
    present from frame 0 so nothing shifts as the bars fill in.
    """
    try:
        refed = pd.read_csv(DATA / 'ReFED_surplus_table(Surplus Data).csv', encoding='utf-8')
    except UnicodeDecodeError:
        refed = pd.read_csv(DATA / 'ReFED_surplus_table(Surplus Data).csv', encoding='cp1252')
    refed = refed.rename(columns={c: 'GHG (Mt CO2e)' for c in refed.columns if 'GHG' in c})

    order = refed.sort_values('Surplus (M tons)', ascending=False).reset_index(drop=True)
    ypos = np.arange(len(order))
    fs_i = int(order.index[order['Sector'] == 'Foodservice'][0])
    farm_i = int(order.index[order['Sector'] == 'Farm'][0])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.11, wspace=0.12)

    panels = [
        (axes[0], 'Surplus (M tons)', 'Surplus food (million tons)', BLUE),
        (axes[1], 'GHG (Mt CO2e)', 'Associated GHG (Mt CO2e)', GREEN),
    ]
    bars_by_panel, labels_by_panel = [], []
    for ax, col, header, hue in panels:
        vals = order[col].to_numpy()
        bars = ax.barh(ypos, np.zeros_like(vals), height=0.55, color=hue)
        style_barh(ax)
        ax.tick_params(labelsize=9.5)
        ax.set_title(header, loc='left', fontsize=11.5, color=SUB, pad=10)
        ax.set_xlim(0, vals.max() * 1.18)
        labs = [ax.text(v + vals.max() * 0.015, yi, fmt1(v), va='center', ha='left',
                        fontsize=10, color=SUB, alpha=0) for yi, v in zip(ypos, vals)]
        bars_by_panel.append((bars, vals))
        labels_by_panel.append(labs)

    axes[0].set_yticks(ypos, order['Sector'])
    axes[0].invert_yaxis()

    ann1 = axes[1].annotate('4th by surplus tons,\n2nd by associated GHG',
                            xy=(73, fs_i), xytext=(80, fs_i), va='center', ha='left',
                            fontsize=9.5, color=INK, alpha=0,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.9, alpha=0))
    ann2 = axes[1].annotate('2nd by surplus tons,\nlast by associated GHG',
                            xy=(13, farm_i), xytext=(22, farm_i), va='center', ha='left',
                            fontsize=9.5, color=INK, alpha=0,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.9, alpha=0))

    fig.text(0.01, 0.955,
             'Residential generates the most surplus, while foodservice carries '
             'disproportionate associated GHG',
             ha='left', fontsize=15, fontweight=600, color=INK)
    fig.text(0.01, 0.90,
             'US surplus food by sector: million tons versus associated GHG (Mt CO2e)',
             ha='left', fontsize=10.5, color=SUB)
    fig.text(0.01, 0.015,
             'Source: ReFED 2024 data, published in 2025 sector fact sheets. Geography: United '
             'States. ReFED publishes associated GHG as MMt CO2e; 1 MMt = 1 Mt = one million tonnes.',
             ha='left', va='bottom', fontsize=8.5, color=MUTED)

    D = DURATION['slide10']
    starts = [0.35, 2.95]   # left panel, then right

    def update(t):
        for (bars, vals), labs, start in zip(bars_by_panel, labels_by_panel, starts):
            p = beat(t, start, 1.5)
            a = beat(t, start + 1.25, 0.5)
            for bar, lab, v in zip(bars, labs, vals):
                bar.set_width(v * p)
                lab.set_alpha(a)
        a = beat(t, 5.6, 0.9)
        for ann in (ann1, ann2):
            ann.set_alpha(a)
            ann.arrow_patch.set_alpha(a)

    return fig, update, D


# ------------------------------------------- reconstruction scaffolding -----
# Slides 12 and 15 are hand-built graphics with no notebook source, so they are
# rebuilt here from the published PNGs. Layout is expressed on a 16x9 grid with
# equal aspect, so a "round" corner is actually round and coordinates read like
# positions on the slide.

class Group:
    """A set of artists revealed together as one beat."""

    def __init__(self, *artists):
        self.artists = list(artists)

    def add(self, *artists):
        self.artists.extend(artists)
        return artists[0] if len(artists) == 1 else artists

    def set_alpha(self, a):
        for art in self.artists:
            art.set_alpha(a)


def slide_canvas():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def card(ax, x, y, w, h, face, edge=None, lw=1.2, r=0.10, z=2):
    """Rounded panel. `y` is the bottom edge."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={r}',
                       facecolor=face, edgecolor=edge or 'none',
                       linewidth=lw if edge else 0, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy_from, xy_to, color, lw=2.2, z=3, style='-|>', ms=14):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=ms,
                        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)
    return a


def text_right(fig, ax, txt) -> float:
    """Right edge of a rendered text in data coords.

    Used to set what follows a run of text instead of hardcoding a gap, which
    would drift the moment a font substitutes or a label is reworded."""
    fig.canvas.draw()
    bb = txt.get_window_extent(fig.canvas.get_renderer())
    return float(ax.transData.inverted().transform((bb.x1, bb.y0))[0])


# ========================================================== slide 12 ========
def build_slide12():
    """The SB 1383 reroute, told in the order it happened.

    Six beats: surplus exists -> it used to go to landfill -> the law strikes
    that path out -> donation is mandated -> people get fed -> what the statute
    actually requires. Landing all of that at once is why the static version is
    hard to follow.
    """
    fig, ax = slide_canvas()

    # --- static chrome -------------------------------------------------
    ax.text(0.59, 8.28, 'SB 1383: rerouting surplus food from landfill to plate',
            fontsize=23, fontweight=700, color=INK, va='center')
    ax.text(0.59, 7.86, 'California Short-Lived Climate Pollutants law (Lara, 2016)   ·   '
                        'regulations in force since January 2022',
            fontsize=12.5, color=MUTED, va='center')
    ax.text(15.55, 0.26, 'Sources: CalRecycle (SB 1383); pantry operations data',
            fontsize=8.5, color=MUTED, ha='right', va='center')

    # Flow-row geometry, measured off the published graphic.
    BOX_Y, BOX_H = 4.33, 1.74
    MID = BOX_Y + BOX_H / 2

    # --- beat 1: the surplus exists ------------------------------------
    g_surplus = Group()
    g_surplus.add(card(ax, 0.46, BOX_Y, 3.60, BOX_H, SURFACE, BLUE, lw=1.4))
    g_surplus.add(ax.text(2.26, MID + 0.38, 'SURPLUS EDIBLE FOOD', fontsize=13,
                          fontweight=700, color=INK, ha='center', va='center', zorder=4))
    g_surplus.add(ax.text(2.26, MID - 0.28, 'supermarkets · grocers\nlarge foodservice',
                          fontsize=10.5, color=MUTED, ha='center', va='center',
                          linespacing=1.45, zorder=4))

    # --- beat 2: the old default ---------------------------------------
    g_landfill = Group()
    g_landfill.add(arrow(ax, (1.82, BOX_Y - 0.04), (2.20, 3.66), MUTED, lw=1.8, ms=12))
    g_landfill.add(card(ax, 0.72, 2.16, 3.24, 1.42, SURFACE, MUTED, lw=1.2))
    g_landfill.artists[-1].set_linestyle((0, (5, 4)))
    g_landfill.add(ax.text(2.34, 3.08, 'LANDFILL', fontsize=13, fontweight=700,
                           color=MUTED, ha='center', va='center', zorder=4))
    g_landfill.add(ax.text(2.34, 2.60, 'rotting organics → methane', fontsize=10,
                           color=MUTED, ha='center', va='center', zorder=4))
    g_landfill.add(ax.text(4.22, 3.02,
                           "The old default.\nLandfilled organic waste makes\n"
                           "20% of California's methane.",
                           fontsize=9.5, color=RED, ha='left', va='center',
                           linespacing=1.5, zorder=4))

    # --- beat 3: the law strikes it out --------------------------------
    # Two strokes crossing over the landfill arrow. Drawn explicitly rather
    # than mirrored in a loop: a sign error there silently yields one line.
    g_x = Group()
    for (xa, ya), (xb, yb) in ((((1.76, 3.84)), ((2.24, 4.30))),
                               (((1.76, 4.30)), ((2.24, 3.84)))):
        ln, = ax.plot([xa, xb], [ya, yb], color=RED, linewidth=3.2,
                      solid_capstyle='round', zorder=6)
        g_x.add(ln)

    # --- beat 4: donation is mandated ----------------------------------
    g_mandate = Group()
    g_mandate.add(ax.text(5.03, 6.38, 'SB 1383 mandates donation', fontsize=11,
                          fontweight=600, color=BLUE, ha='center', va='center'))
    g_mandate.add(arrow(ax, (4.14, MID), (5.98, MID), BLUE))
    g_mandate.add(card(ax, 6.05, BOX_Y, 3.96, BOX_H, BLUE, None))
    g_mandate.add(ax.text(8.03, MID + 0.38, 'FOOD RECOVERY ORGS', fontsize=13,
                          fontweight=700, color='white', ha='center', va='center', zorder=4))
    g_mandate.add(ax.text(8.03, MID - 0.28, 'food banks & pantries', fontsize=10.5,
                          color='#dce9f9', ha='center', va='center', zorder=4))

    # --- beat 5: people get fed ----------------------------------------
    g_fed = Group()
    g_fed.add(arrow(ax, (10.09, MID), (11.98, MID), BLUE))
    g_fed.add(card(ax, 12.05, BOX_Y, 3.42, BOX_H, SURFACE, BLUE, lw=1.4))
    g_fed.add(ax.text(13.76, MID + 0.38, 'PEOPLE FED', fontsize=13, fontweight=700,
                      color=INK, ha='center', va='center', zorder=4))
    g_fed.add(ax.text(13.76, MID - 0.28, 'meals, not methane', fontsize=10.5,
                      color=MUTED, ha='center', va='center', zorder=4))

    # --- beat 6: what the statute requires -----------------------------
    chips = [
        (6.86, '2016 → 2022', 'signed → rules\nin force'),
        (9.74, '−75%', 'organic waste to landfill\nby 2025 (vs 2014)'),
        (12.63, '≥20%', 'of disposed edible food\nrecovered for people'),
    ]
    g_chips = []
    for x0, big, small in chips:
        g = Group()
        g.add(card(ax, x0, 2.16, 2.84, 1.42, BLUE_PALE, None, r=0.09))
        g.add(ax.text(x0 + 1.42, 3.34, big, fontsize=19, fontweight=700,
                      color=BLUE, ha='center', va='center', zorder=4))
        g.add(ax.text(x0 + 1.42, 2.60, small, fontsize=9.5, color=SUB,
                      ha='center', va='center', linespacing=1.45, zorder=4))
        g_chips.append(g)

    # --- beat 7: it is not hypothetical --------------------------------
    g_banner = Group()
    g_banner.add(card(ax, 0.53, 0.52, 14.94, 0.90, BLUE_PALE, None, r=0.09))
    lead = ax.text(0.92, 0.97, 'This law in action at Laguna Food Pantry:',
                   fontsize=11.5, fontweight=700, color=BLUE, va='center', zorder=4)
    g_banner.add(lead)
    g_banner.add(ax.text(text_right(fig, ax, lead) + 0.22, 0.97,
                         '1.2M+ lbs recovered last year   ·   ~270 families '
                         '(>1,000 individuals) fed on average per day',
                         fontsize=11.5, color=INK, va='center', zorder=4))

    for g in [g_surplus, g_landfill, g_x, g_mandate, g_fed, g_banner] + g_chips:
        g.set_alpha(0)

    D = DURATION['slide12']
    schedule = [
        (g_surplus, 0.30, 0.60),
        (g_landfill, 1.50, 0.70),
        (g_x, 3.20, 0.45),
        (g_mandate, 4.30, 0.80),
        (g_fed, 6.10, 0.70),
        (g_chips[0], 7.50, 0.55),
        (g_chips[1], 7.90, 0.55),
        (g_chips[2], 8.30, 0.55),
        (g_banner, 9.50, 0.70),
    ]

    def update(t):
        for g, start, dur in schedule:
            g.set_alpha(beat(t, start, dur))

    return fig, update, D


# ========================================================== slide 15 ========
def build_slide15():
    """The close: the beef/peas gap that opened the deck, then what to do."""
    fig, ax = slide_canvas()

    BEEF, PEAS = 59.6, 0.90
    BAR_X, BAR_FULL = 0.555, 4.16      # width of the beef bar at 59.6

    # --- static chrome -------------------------------------------------
    ax.text(0.555, 8.66, 'T H E   T A K E A W A Y', fontsize=9.5, fontweight=700,
            color=MUTED, va='center')
    ax.text(0.555, 8.16, 'Food is the highest-leverage choice you make three times a day,',
            fontsize=22, fontweight=700, color=INK, va='center')
    ax.text(0.555, 7.52, 'and SB 1383 shows how that same choice scales into policy.',
            fontsize=22, fontweight=700, color=BLUE, va='center')
    ax.text(0.555, 6.73, 'WHERE WE STARTED', fontsize=9.5, fontweight=700,
            color=MUTED, va='center')
    ax.text(6.16, 6.73, 'WHAT TO DO WITH IT', fontsize=9.5, fontweight=700,
            color=MUTED, va='center')
    ax.plot([5.62, 5.62], [1.95, 6.90], color=GRID, linewidth=1.2, zorder=1)
    ax.text(15.58, 0.28, 'Poore & Nemecek (2018) · ReFED (2024) · Climate Watch · '
                         'CalRecycle (SB 1383)',
            fontsize=8.5, color=MUTED, ha='right', va='center')

    # --- beat 1: beef --------------------------------------------------
    g_beef = Group()
    g_beef.add(ax.text(0.555, 6.22, '1 kg beef', fontsize=11.5, color=INK, va='center'))
    beef_bar = card(ax, BAR_X, 5.32, BAR_FULL, 0.52, BLUE, None, r=0.0, z=3)
    g_beef.add(beef_bar)
    lab_beef = ax.text(BAR_X + BAR_FULL + 0.16, 5.58, '59.6', fontsize=12.5,
                       fontweight=700, color=INK, va='center', zorder=4)
    g_beef.add(lab_beef)

    # --- beat 2: peas --------------------------------------------------
    g_peas = Group()
    g_peas.add(ax.text(0.555, 4.93, '1 kg peas', fontsize=11.5, color=INK, va='center'))
    peas_w = BAR_FULL * PEAS / BEEF
    peas_bar = card(ax, BAR_X, 4.09, peas_w, 0.49, GREEN, None, r=0.0, z=3)
    g_peas.add(peas_bar)
    g_peas.add(ax.text(0.78, 4.33, '0.90', fontsize=12.5, fontweight=700,
                       color=INK, va='center', zorder=4))
    g_peas.add(ax.text(1.86, 4.33, 'kg CO₂e per kg, farm to shelf', fontsize=10,
                       color=MUTED, va='center', zorder=4))

    # --- beat 3: the ratio ---------------------------------------------
    # "66x" reads as one unit, so the multiplication sign is set against the
    # measured right edge of the numeral rather than at a guessed offset.
    g_ratio = Group()
    ratio_num = ax.text(0.52, 3.32, '66', fontsize=44, fontweight=700,
                        color=BLUE, va='center')
    g_ratio.add(ratio_num)
    times = ax.text(text_right(fig, ax, ratio_num) - 0.02, 3.28, '×', fontsize=40,
                    fontweight=400, color=BLUE, va='center')
    g_ratio.add(times)
    g_ratio.add(ax.text(text_right(fig, ax, times) + 0.16, 3.32,
                        'the gap between\ntwo dinners',
                        fontsize=11.5, color=SUB, va='center', linespacing=1.5))
    g_ratio.add(ax.text(0.555, 2.28,
                        "For high-impact foods, what you eat often beats\n"
                        "where it's from: transport is 0.6% of beef's footprint.",
                        fontsize=10, color=SUB, va='center', linespacing=1.6))

    # --- beats 4-6: the three actions ----------------------------------
    actions = [
        (6.375, '1', 'SHIFT YOUR PLATE',
         'Trade beef and lamb for chicken, fish, or legumes',
         'the single biggest change one person can make'),
        (4.800, '2', 'WASTE LESS',
         'Buy, store, and finish what you buy',
         'meaningful, and entirely within your control'),
        (3.225, '3', 'BACK THE POLICY',
         'Support SB 1383-style food recovery laws',
         'the same choice, multiplied across millions'),
    ]
    g_actions = []
    for top, num, title, body, tag in actions:
        y0 = top - 1.29
        g = Group()
        g.add(card(ax, 6.06, y0, 9.49, 1.29, BLUE_PALE, '#cfe0f5', lw=1.0, r=0.10))
        circ = plt.Circle((6.63, y0 + 0.645), 0.20, facecolor=BLUE,
                          edgecolor='none', zorder=4)
        ax.add_patch(circ)
        g.add(circ)
        g.add(ax.text(6.63, y0 + 0.645, num, fontsize=10.5, fontweight=700,
                      color='white', ha='center', va='center', zorder=5))
        g.add(ax.text(7.07, y0 + 0.96, title, fontsize=13, fontweight=700,
                      color=INK, va='center', zorder=5))
        g.add(ax.text(7.07, y0 + 0.62, body, fontsize=10.5, color=SUB,
                      va='center', zorder=5))
        g.add(ax.text(7.07, y0 + 0.28, tag, fontsize=10.5, color=BLUE,
                      style='italic', va='center', zorder=5))
        g_actions.append(g)

    # --- beat 7: the banner --------------------------------------------
    g_banner = Group()
    g_banner.add(card(ax, 0.43, 0.50, 15.17, 0.94, BLUE_PALE, None, r=0.09))
    g_banner.add(ax.text(0.91, 0.97, 'The lever is on your plate. The scale is in the policy.',
                         fontsize=11.5, fontweight=700, color=BLUE, va='center', zorder=4))
    g_banner.add(ax.text(7.61, 0.97, 'Laguna Food Pantry alone: 1.2M+ lbs recovered last year.',
                         fontsize=11.5, color=INK, va='center', zorder=4))

    for g in [g_beef, g_peas, g_ratio, g_banner] + g_actions:
        g.set_alpha(0)

    D = DURATION['slide15']

    def update(t):
        # The two bars grow rather than fade: the length is the message.
        p = beat(t, 0.30, 0.85)
        g_beef.set_alpha(1.0 if p > 0 else 0.0)
        beef_bar.set_width(BAR_FULL * p)
        lab_beef.set_x(BAR_X + BAR_FULL * p + 0.16)
        lab_beef.set_alpha(beat(t, 0.85, 0.40))

        p2 = beat(t, 1.50, 0.55)
        g_peas.set_alpha(p2)
        peas_bar.set_width(peas_w * max(p2, 1e-6))

        g_ratio.set_alpha(beat(t, 2.50, 0.70))
        for i, g in enumerate(g_actions):
            g.set_alpha(beat(t, 3.70 + i * 1.20, 0.60))
        g_banner.set_alpha(beat(t, 7.20, 0.65))

    return fig, update, D


BUILDERS = {
    '03': ('anim_slide03_regression', build_slide03),
    '08': ('anim_slide08_dendrogram', build_slide08),
    '09': ('anim_slide09_practical_scenarios', build_slide09),
    '10': ('anim_slide10_surplus_tons_vs_ghg', build_slide10),
    '12': ('anim_slide12_sb1383', build_slide12),
    '15': ('anim_slide15_close', build_slide15),
}


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    base_only = '--base-only' in argv
    variants_only = '--variants-only' in argv

    want = [a.zfill(2) for a in args] or list(BUILDERS)
    unknown = [w for w in want if w not in BUILDERS]
    if unknown:
        raise SystemExit(f'unknown slide(s): {unknown}; have {sorted(BUILDERS)}')

    for key in want:
        stem, fn = BUILDERS[key]
        slide = f'slide{key}'
        base = DURATION[slide]
        print(f'slide {key}  (schedule written for {base:.0f}s):')

        targets = []
        if not variants_only:
            targets.append((f'{stem}.mp4', None))
        if not base_only:
            targets += [(f'{stem}_{s}s.mp4', float(s)) for s in VARIANTS.get(slide, ())]

        for name, seconds in targets:
            # Rebuilt per render: each run mutates its artists to the final frame.
            fig, update, _ = fn()
            render(name, fig, update, base, seconds)


if __name__ == '__main__':
    main(sys.argv[1:])
