"""
Runner for bead chain simulation.

Runs the FEA simulation, exports GNN-ready data, and visualizes results.

Usage:
    python main.py                    # run with defaults
    python main.py --n_nodes 30 --dt 0.0005 --save_anim chain.gif
"""

import argparse
import os
import numpy as np
from simulation import run_simulation
from visualize import animate_chain, plot_energy


def main():
    parser = argparse.ArgumentParser(description="Bead chain gravity simulation")
    parser.add_argument("--n_nodes", type=int, default=20,
                        help="Number of mass nodes in chain")
    parser.add_argument("--total_length", type=float, default=2.0,
                        help="Total chain length (m)")
    parser.add_argument("--total_mass", type=float, default=0.5,
                        help="Total chain mass (kg)")
    parser.add_argument("--gravity", type=float, default=9.81,
                        help="Gravitational acceleration (m/s^2)")
    parser.add_argument("--stiffness", type=float, default=1e4,
                        help="Rod element stiffness (N/m)")
    parser.add_argument("--damping", type=float, default=0.5,
                        help="Damping coefficient")
    parser.add_argument("--dt", type=float, default=0.0001,
                        help="Simulation timestep (s)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration (s)")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save every N steps")
    parser.add_argument("--output", type=str, default="data/chain_trajectory.npz",
                        help="Output path for trajectory data")
    parser.add_argument("--save_anim", type=str, default=None,
                        help="Save animation to file (.gif or .mp4)")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip visualization")
    args = parser.parse_args()

    n_steps = int(args.duration / args.dt)

    print(f"Bead Chain Simulation")
    print(f"  Nodes: {args.n_nodes}")
    print(f"  Length: {args.total_length} m")
    print(f"  Mass: {args.total_mass} kg")
    print(f"  Stiffness: {args.stiffness} N/m")
    print(f"  Damping: {args.damping}")
    print(f"  dt: {args.dt} s")
    print(f"  Steps: {n_steps}")
    print()

    # Run simulation
    print("Running simulation...")
    trajectory = run_simulation(
        n_nodes=args.n_nodes,
        total_length=args.total_length,
        total_mass=args.total_mass,
        gravity=args.gravity,
        stiffness=args.stiffness,
        damping=args.damping,
        dt=args.dt,
        n_steps=n_steps,
        save_interval=args.save_interval,
    )

    n_saved = len(trajectory["times"])
    print(f"Done. Saved {n_saved} frames over {trajectory['times'][-1]:.2f} s")

    # Export data
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        **trajectory,
    )
    print(f"Trajectory saved to {args.output}")

    # Visualize
    if not args.no_viz:
        print("\nShowing animation (close window to continue)...")
        animate_chain(trajectory, save_path=args.save_anim)

        print("Showing energy plot...")
        plot_energy(trajectory, gravity=args.gravity)


if __name__ == "__main__":
    main()
