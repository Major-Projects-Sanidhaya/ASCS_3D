"""
matplotlib_backend.py
---------------------
Real-time 2D bird's-eye visualisation using matplotlib.

Layout
------
  ┌──────────────────────────────────────┐
  │   bird's-eye arena  (agents + text)  │
  ├──────────────────────────────────────┤
  │  w_sep   ──────●──────────────────   │  ← Slider row 4
  │  w_align ──────────●──────────────   │  ← Slider row 3
  │  w_coh   ──────────────●──────────   │  ← Slider row 2
  │  waypoint ─────────────────●──────   │  ← Slider row 1
  └──────────────────────────────────────┘

Keyboard controls
-----------------
  q / Esc    — quit
  Arrow keys — engage manual steering (hold direction)
  m          — toggle manual mode off
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# ── colour / style table ──────────────────────────────────────────────────────

_TIER_STYLES: Dict[str, dict] = {
    'general': dict(c='#7B3FC4', s=160, marker='*', label='General', zorder=5),
    'nodes':   dict(c='#1A9980', s=90,  marker='s', label='Node',    zorder=4),
    'scouts':  dict(c='#E07B00', s=28,  marker='o', label='Scout',   zorder=3),
    'workers': dict(c='#2255BB', s=55,  marker='D', label='Worker',  zorder=3),
}

_DEFAULT_WEIGHTS = {
    'w_sep':   0.6,
    'w_align': 0.4,
    'w_coh':   0.5,
    'w_wp':    0.35,
}

_SLIDER_SPECS = [
    # (weight_key, label,       v_min, v_max, v_init)
    ('w_wp',    'Waypoint',    0.0,   1.0,   0.35),
    ('w_coh',   'Cohesion',    0.0,   1.5,   0.5),
    ('w_align', 'Alignment',   0.0,   1.5,   0.4),
    ('w_sep',   'Separation',  0.0,   1.5,   0.6),
]


# ── backend ───────────────────────────────────────────────────────────────────

class MatplotlibBackend:
    """
    Bird's-eye 2D visualisation with live Reynolds sliders.

    All matplotlib imports are deferred to setup() so that the module can be
    imported without matplotlib installed (headless environments).
    """

    def __init__(self) -> None:
        self._running    = False
        self._quit       = False
        self._manual     = False
        self._direction  = [0.0, 0.0]
        self._altitude   = 3.0
        self._weights    = dict(_DEFAULT_WEIGHTS)

        self._fig        = None
        self._ax         = None
        self._scatters: Dict[str, object] = {}
        self._sliders:  Dict[str, object] = {}
        self._node_texts: List[object]    = []
        self._status_text = None
        self._plt        = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, config: dict) -> None:
        import matplotlib
        matplotlib.use('TkAgg') if _tk_available() else matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.widgets as mwidgets

        self._plt = plt

        arena_w  = float(config.get('arena_w', 20.0))
        arena_h  = float(config.get('arena_h', 20.0))
        half_w   = arena_w / 2.0
        half_h   = arena_h / 2.0

        # ── figure & axes ──
        self._fig = plt.figure(figsize=(11, 9))
        self._fig.patch.set_facecolor('#1C1C1C')

        # Main arena axes (leave room at bottom for sliders)
        self._ax = self._fig.add_axes([0.07, 0.34, 0.88, 0.60])
        self._ax.set_facecolor('#111111')
        for spine in self._ax.spines.values():
            spine.set_edgecolor('#444444')

        self._ax.set_xlim(-half_w - 1.5, half_w + 1.5)
        self._ax.set_ylim(-half_h - 1.5, half_h + 1.5)
        self._ax.set_aspect('equal')
        self._ax.set_xlabel('X (m)', color='#AAAAAA', fontsize=8)
        self._ax.set_ylabel('Y (m)', color='#AAAAAA', fontsize=8)
        self._ax.tick_params(colors='#666666', labelsize=7)

        # Arena boundary
        rect = mpatches.FancyBboxPatch(
            (-half_w, -half_h), arena_w, arena_h,
            boxstyle='square,pad=0', linewidth=1.2,
            edgecolor='#444466', facecolor='none', linestyle='--',
        )
        self._ax.add_patch(rect)

        # Grid guides
        cols = int(config.get('grid_cols', 2))
        rows = int(config.get('grid_rows', 2))
        cell_w = arena_w / cols
        cell_h = arena_h / rows
        for c in range(1, cols):
            x = -half_w + c * cell_w
            self._ax.axvline(x, color='#333355', linewidth=0.6, linestyle=':')
        for r in range(1, rows):
            y = -half_h + r * cell_h
            self._ax.axhline(y, color='#333355', linewidth=0.6, linestyle=':')

        # ── scatter plots (start empty) ──
        for tier, style in _TIER_STYLES.items():
            sc = self._ax.scatter([], [], **style)
            self._scatters[tier] = sc

        self._ax.legend(
            loc='upper right', fontsize=7,
            facecolor='#222222', edgecolor='#444444', labelcolor='#CCCCCC',
        )

        # Status overlay (top-left corner of arena)
        self._status_text = self._ax.text(
            -half_w + 0.4, half_h - 0.5, '',
            fontsize=7, va='top', family='monospace',
            color='#AADDAA',
        )

        # Title
        self._ax.set_title(
            'ASCS_3D  |  q=quit  arrows=steer  m=manual toggle',
            color='#CCCCCC', fontsize=8, pad=4,
        )

        # ── Reynolds sliders ──
        slider_left   = 0.15
        slider_width  = 0.68
        slider_height = 0.03
        slider_base   = 0.03
        slider_gap    = 0.07

        for row, (key, label, vmin, vmax, vinit) in enumerate(_SLIDER_SPECS):
            ax_sl = self._fig.add_axes([
                slider_left,
                slider_base + row * slider_gap,
                slider_width,
                slider_height,
            ])
            ax_sl.set_facecolor('#222222')

            import matplotlib.widgets as mwidgets
            sl = mwidgets.Slider(
                ax_sl, label, vmin, vmax,
                valinit=vinit,
                color='#3366AA', track_color='#333333',
            )
            sl.label.set_color('#AAAAAA')
            sl.label.set_fontsize(7)
            sl.valtext.set_color('#AAAAAA')
            sl.valtext.set_fontsize(7)

            def _make_cb(k):
                def _cb(val):
                    self._weights[k] = float(val)
                return _cb

            sl.on_changed(_make_cb(key))
            self._sliders[key] = sl

        # ── event connections ──
        self._fig.canvas.mpl_connect('key_press_event',  self._on_key)
        self._fig.canvas.mpl_connect('close_event',      self._on_close)

        plt.ion()
        plt.show(block=False)
        plt.pause(0.05)

        self._running = True
        print('MatplotlibBackend: window open  (q to quit, arrow keys to steer)')

    # ── interface ─────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        if not self._running:
            return False
        try:
            return bool(self._plt.fignum_exists(self._fig.number))
        except Exception:
            return False

    def get_user_input(self) -> dict:
        return {
            'quit':      self._quit,
            'weights':   dict(self._weights),
            'manual':    self._manual,
            'direction': list(self._direction),
            'altitude':  self._altitude,
        }

    def update(
        self,
        positions: dict,
        status:    dict,
        debug_info: dict,
    ) -> None:
        # ── update scatter positions ──
        for tier, pos_list in positions.items():
            sc = self._scatters.get(tier)
            if sc is None:
                continue
            if pos_list:
                pts = np.array(pos_list, dtype=float)
                sc.set_offsets(pts[:, :2])
            else:
                sc.set_offsets(np.empty((0, 2), dtype=float))

        # ── status overlay text ──
        step    = status.get('step_count',    0)
        phase   = status.get('mission_phase', '?')
        health  = status.get('overall_health', 1.0)
        op_phs  = debug_info.get('op_phases',  [])
        modes   = debug_info.get('node_states', [])
        manual  = debug_info.get('manual_mode', False)
        bias    = debug_info.get('bias',  [0, 0])

        lines = [f'step={step}  phase={phase}  health={health:.2f}']
        if manual:
            lines.append(f'MANUAL  bias=[{bias[0]:.1f},{bias[1]:.1f}]')
        for i, (mode, op) in enumerate(zip(modes, op_phs)):
            lines.append(f'N{i}: {mode} | {op}')
        self._status_text.set_text('\n'.join(lines))

        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def teardown(self) -> None:
        self._running = False
        try:
            self._plt.close('all')
        except Exception:
            pass

    # ── keyboard handler ──────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        k = event.key
        if k in ('q', 'Q', 'escape'):
            self._quit    = True
            self._running = False
        elif k == 'up':
            self._manual    = True
            self._direction = [0.0,  5.0]
        elif k == 'down':
            self._manual    = True
            self._direction = [0.0, -5.0]
        elif k == 'left':
            self._manual    = True
            self._direction = [-5.0, 0.0]
        elif k == 'right':
            self._manual    = True
            self._direction = [5.0,  0.0]
        elif k == 'm':
            self._manual    = not self._manual
            self._direction = [0.0,  0.0]

    def _on_close(self, event) -> None:
        self._quit    = True
        self._running = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _tk_available() -> bool:
    """Return True if Tkinter can be imported (TkAgg will work)."""
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False
