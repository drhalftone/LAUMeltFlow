"""Time BeadMPGNN vs reference symplectic Euler solver.

Reports wall-clock seconds per rollout step and GNN speedup factor across
several chain lengths. Both CPU and (if available) CUDA.

Usage:
    python time_speedup.py
    python time_speedup.py --n_steps 2000 --chain_lengths 8,16,32
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from bead_mpgnn import BeadMPGNN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation import create_chain, compute_forces


def step_ref(state, dt, gravity):
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = BeadMPGNN(node_dim=4, edge_dim=6, output_dim=4,
                  hidden_dim=a["hidden_dim"],
                  n_message_passes=a.get("n_message_passes", 1)).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def time_ref(n_beads, n_steps, dt, gravity, taper, warmup):
    params = dict(n_nodes=n_beads, total_length=2.0, total_mass=0.5,
                  stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=taper)
    state = create_chain(**params)
    anchor = state.pos[0].copy()
    for _ in range(warmup):
        step_ref(state, dt, gravity)
        state.pos[0] = anchor
        state.vel[0] = 0.0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_ref(state, dt, gravity)
        state.pos[0] = anchor
        state.vel[0] = 0.0
    t1 = time.perf_counter()
    return (t1 - t0) / n_steps


def time_gnn(n_beads, n_steps, model, device, taper, warmup):
    """Time the GNN with tensors staying on-device across steps."""
    params = dict(n_nodes=n_beads, total_length=2.0, total_mass=0.5,
                  stiffness=1e4, damping=0.5, drag=0.02, taper_ratio=taper)
    state = create_chain(**params)

    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest = torch.from_numpy(state.rest_lengths).float().to(device)
    anchor = pos[0].clone()

    def one_step():
        nonlocal pos, vel
        pos, vel = model.step_chain(pos, vel, mass, is_fixed, edges, rest)
        pos[0] = anchor
        vel[0] = 0.0

    with torch.no_grad():
        for _ in range(warmup):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_steps


def main(a):
    chain_lengths = [int(n) for n in a.chain_lengths.split(",")]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"Timing with {a.n_steps} steps per chain (warmup {a.warmup}), taper {a.taper_ratio}")
    print()

    for device in devices:
        model = load_model(a.model_path, device)
        print(f"=== device: {device} ===")
        print(f"{'N':>4} | {'ref (µs/step)':>14} | {'gnn (µs/step)':>14} | {'speedup':>8}")
        print("-" * 54)
        for n in chain_lengths:
            t_ref = time_ref(n, a.n_steps, 1e-4, 9.81, a.taper_ratio, a.warmup)
            t_gnn = time_gnn(n, a.n_steps, model, device, a.taper_ratio, a.warmup)
            ratio = t_ref / t_gnn if t_gnn > 0 else float("inf")
            print(f"{n:>4} | {t_ref*1e6:>14.2f} | {t_gnn*1e6:>14.2f} | {ratio:>7.2f}×")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="outputs/mpgnn_best.pt")
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--chain_lengths", type=str, default="8,16,32,64")
    p.add_argument("--taper_ratio", type=float, default=10.0)
    main(p.parse_args())
