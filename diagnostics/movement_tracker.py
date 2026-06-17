"""
movement_tracker.py
-------------------
Records drone positions every N frames to a CSV for analysis.
Used to diagnose coverage and movement patterns.
"""

import csv
import os
import time
import numpy as np


class MovementTracker:
    """
    Records drone positions every N frames to a CSV for analysis.
    Used to diagnose coverage and movement patterns.
    """

    def __init__(self, swarm, log_dir='diagnostics/logs',
                 sample_every=10):
        self.swarm = swarm
        self.sample_every = sample_every
        self.frame = 0
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'movement_{ts}.csv')
        self.rows = []
        # Capture zone bounds for reference
        self.zone_bounds = {}
        for zh in swarm._zone_map.active_zones():
            c = swarm._zone_map.get_zone_centre(zh)
            self.zone_bounds[zh] = c

    def record(self):
        if self.frame % self.sample_every == 0:
            t = self.frame / 60.0
            # General
            g = self.swarm._general._agent
            self.rows.append([t, 'GENERAL', 'general',
                              g.pos[0], g.pos[1], g.pos[2], -1])
            # Nodes
            for zh, nc in self.swarm._nodes.items():
                n = nc._agent
                self.rows.append([t, 'NODE', f'node_{zh}',
                                  n.pos[0], n.pos[1], n.pos[2], zh])
            # Scouts — record which zone they belong to
            for nc in self.swarm._nodes.values():
                zh = nc._agent.zone_hash
                for sc in nc._scout_controllers:
                    s = sc._agent
                    self.rows.append([t, 'SCOUT', s.scout_id,
                                      s.pos[0], s.pos[1], s.pos[2], zh])
            # Workers
            for nc in self.swarm._nodes.values():
                zh = nc._agent.zone_hash
                for wc in nc._worker_controllers:
                    w = wc._agent
                    self.rows.append([t, 'WORKER', 'worker',
                                      w.pos[0], w.pos[1], w.pos[2], zh])
        self.frame += 1

    def save(self):
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time','tier','drone_id','x','y','z','zone_hash'])
            writer.writerows(self.rows)
        print(f'[MovementTracker] Saved {len(self.rows)} rows to {self.path}')
        return self.path

    def print_coverage_analysis(self):
        """Print per-zone and per-scout coverage stats."""
        print('\n=== COVERAGE ANALYSIS ===')

        # Arena bounds
        arena_w = self.swarm._config.get('arena_w', 15)
        arena_h = self.swarm._config.get('arena_h', 15)
        print(f'Arena: {arena_w}x{arena_h}  '
              f'bounds X[{-arena_w/2:.1f},{arena_w/2:.1f}] '
              f'Y[{-arena_h/2:.1f},{arena_h/2:.1f}]')

        print('\nZone centers:')
        for zh, c in self.zone_bounds.items():
            print(f'  Zone {zh}: ({c[0]:.1f}, {c[1]:.1f})')

        # Per scout: bounding box of where it actually went
        scout_positions = {}
        for row in self.rows:
            t, tier, did, x, y, z, zh = row
            if tier == 'SCOUT':
                scout_positions.setdefault(did, []).append((x, y))

        print('\nPer-scout movement bounding box:')
        for did, positions in list(scout_positions.items())[:8]:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            print(f'  {did[:16]}: '
                  f'X[{min(xs):.1f},{max(xs):.1f}] '
                  f'Y[{min(ys):.1f},{max(ys):.1f}] '
                  f'area={(max(xs)-min(xs))*(max(ys)-min(ys)):.1f}m2')

        # Overall scout coverage
        all_x = [p[0] for ps in scout_positions.values() for p in ps]
        all_y = [p[1] for ps in scout_positions.values() for p in ps]
        covered_area = (max(all_x)-min(all_x)) * (max(all_y)-min(all_y))
        total_area = arena_w * arena_h
        print(f'\nOverall scout coverage:')
        print(f'  X spread: [{min(all_x):.1f}, {max(all_x):.1f}]')
        print(f'  Y spread: [{min(all_y):.1f}, {max(all_y):.1f}]')
        print(f'  Covered bounding area: {covered_area:.1f}m2 '
              f'of {total_area:.1f}m2 '
              f'({100*covered_area/total_area:.0f}%)')

        # The key diagnostic — are scouts staying near their zone center
        # or spreading to fill their zone?
        print('\nScout distance from assigned zone center:')
        scout_zone = {}
        for row in self.rows:
            t, tier, did, x, y, z, zh = row
            if tier == 'SCOUT':
                scout_zone.setdefault(did, (zh, []))
                scout_zone[did][1].append((x, y))
        for did, (zh, positions) in list(scout_zone.items())[:8]:
            if zh in self.zone_bounds:
                zc = self.zone_bounds[zh]
                dists = [np.sqrt((x-zc[0])**2 + (y-zc[1])**2)
                         for x, y in positions]
                print(f'  {did[:16]} (zone {zh}): '
                      f'mean dist from center={np.mean(dists):.1f}m '
                      f'max={max(dists):.1f}m')
