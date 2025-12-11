#!/usr/bin/env python3
"""
Batch runner for MeltFlow simulations.

Runs multiple configurations and generates comparison plots.

Usage:
    python run_batch.py                # Run all methods, no plots during sim
    python run_batch.py --plot         # Show live plots for each simulation
    python run_batch.py --compare      # Run all and show comparison at end
"""

import argparse
import subprocess
import sys
import os


def run_simulation(config, show_plot=False):
    """Run a single simulation."""
    cmd = [sys.executable, '-m', 'meltflow', '--config', config]
    if not show_plot:
        cmd.append('--no-plot')

    print(f"\n{'='*60}")
    print(f"Running: {config}")
    print('='*60)

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Batch run MeltFlow simulations')
    parser.add_argument('--plot', action='store_true',
                        help='Show live plots during simulation')
    parser.add_argument('--compare', action='store_true',
                        help='Show comparison plot at the end')
    parser.add_argument('--methods', nargs='+',
                        default=['in_1Dsod1fl', 'in_1Dsod1fl_dg0', 'in_1Dsod1fl_dg', 'in_1Dsod1fl_dg2'],
                        help='List of configs to run')
    args = parser.parse_args()

    print("="*60)
    print("MeltFlow Batch Runner")
    print("="*60)
    print(f"Methods to run: {args.methods}")

    # Run all simulations
    results = {}
    for config in args.methods:
        success = run_simulation(config, show_plot=args.plot)
        results[config] = success

    # Summary
    print("\n" + "="*60)
    print("BATCH RUN COMPLETE")
    print("="*60)
    for config, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {config}: {status}")

    # Run comparison
    if args.compare:
        print("\n" + "="*60)
        print("Running comparison...")
        print("="*60)
        subprocess.run([sys.executable, 'compare_results.py', '--all'])


if __name__ == '__main__':
    main()
