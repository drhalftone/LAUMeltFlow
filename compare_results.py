#!/usr/bin/env python3
"""
Compare FVM and DG simulation results.

Usage:
    python compare_results.py                    # Print numerical comparison
    python compare_results.py --plot             # Show comparison plots
    python compare_results.py --all              # Compare all available methods
    python compare_results.py --save             # Save plot without displaying
"""

import numpy as np
import argparse
import os

def load_data(filename):
    """Load simulation output file."""
    if not os.path.exists(filename):
        return None
    data = np.loadtxt(filename, skiprows=1)
    return {
        'x': data[:, 0],
        'rho': data[:, 1],
        'u': data[:, 2],
        'p': data[:, 3],
        'phi': data[:, 4]
    }


def compare_two(fvm_file, dg_file):
    """Compare two result files numerically."""
    fvm = load_data(fvm_file)
    dg = load_data(dg_file)

    if fvm is None:
        print(f"Error: {fvm_file} not found.")
        return None
    if dg is None:
        print(f"Error: {dg_file} not found.")
        return None

    print("=" * 60)
    print(f"Comparing: {os.path.basename(fvm_file)} vs {os.path.basename(dg_file)}")
    print("=" * 60)

    variables = ['rho', 'u', 'p']
    var_names = ['Density (rho)', 'Velocity (u)', 'Pressure (p)']

    print(f"\n{'Variable':<20} {'Max Diff':<15} {'L2 Norm':<15} {'Rel. Error':<15}")
    print("-" * 65)

    total_rel_error = 0
    for var, name in zip(variables, var_names):
        diff = np.abs(fvm[var] - dg[var])
        max_diff = np.max(diff)
        l2_norm = np.sqrt(np.mean(diff**2))
        rel_error = l2_norm / (np.sqrt(np.mean(fvm[var]**2)) + 1e-10)
        total_rel_error += rel_error
        print(f"{name:<20} {max_diff:<15.6e} {l2_norm:<15.6e} {rel_error:<15.6e}")

    print("-" * 65)
    avg_rel_error = total_rel_error / len(variables)
    print(f"Average Relative Error: {avg_rel_error:.6e}")

    if avg_rel_error < 1e-10:
        print("Status: IDENTICAL (machine precision)")
    elif avg_rel_error < 1e-3:
        print("Status: VERY SIMILAR (< 0.1% difference)")
    elif avg_rel_error < 1e-1:
        print("Status: MODERATE differences")
    else:
        print("Status: SIGNIFICANT differences")

    return {'fvm': fvm, 'dg': dg, 'error': avg_rel_error}


def plot_comparison(datasets, labels, title="Method Comparison", save_path=None, show=True):
    """
    Plot multiple datasets overlaid for comparison.

    Parameters
    ----------
    datasets : list of dict
        List of data dictionaries with 'x', 'rho', 'u', 'p' keys
    labels : list of str
        Labels for each dataset
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt

    # Color scheme for different methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Density
    ax = axes[0, 0]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        ax.plot(data['x'], data['rho'], color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('Density ρ [kg/m³]', fontsize=11)
    ax.set_title('Density', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        ax.plot(data['x'], data['u'], color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('Velocity u [m/s]', fontsize=11)
    ax.set_title('Velocity', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Pressure
    ax = axes[1, 0]
    for i, (data, label) in enumerate(zip(datasets, labels)):
        ax.plot(data['x'], data['p'] / 1000, color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('Pressure p [kPa]', fontsize=11)
    ax.set_title('Pressure', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Differences from first dataset (baseline)
    ax = axes[1, 1]
    baseline = datasets[0]
    for i, (data, label) in enumerate(zip(datasets[1:], labels[1:]), 1):
        diff_rho = np.abs(baseline['rho'] - data['rho'])
        ax.plot(data['x'], diff_rho, color=colors[i % len(colors)],
                linestyle='-', label=f'|Δρ| {label}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('x [m]', fontsize=11)
    ax.set_ylabel('|Δρ| [kg/m³]', fontsize=11)
    ax.set_title(f'Density Difference from {labels[0]}', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    if len(datasets) > 1:
        max_diff = max(np.max(np.abs(baseline['rho'] - d['rho'])) for d in datasets[1:])
        if max_diff > 1e-10:
            ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compare_all(show_plot=True, save_plot=False):
    """Compare all available simulation results."""

    # Define available result files and labels
    files = [
        ('data/flow_1Dsod1fl.d', 'FVM (baseline)'),
        ('data/flow_1Dsod1fl_dg0.d', 'DG (p=0)'),
        ('data/flow_1Dsod1fl_dg.d', 'DG (p=1)'),
        ('data/flow_1Dsod1fl_dg2.d', 'DG (p=2)'),
    ]

    # Load available datasets
    datasets = []
    labels = []
    for filepath, label in files:
        data = load_data(filepath)
        if data is not None:
            datasets.append(data)
            labels.append(label)
            print(f"Loaded: {filepath}")
        else:
            print(f"Not found: {filepath}")

    if len(datasets) < 2:
        print("\nNeed at least 2 datasets to compare. Run simulations first:")
        print("  ./run_simulation.sh --config in_1Dsod1fl --no-plot")
        print("  ./run_simulation.sh --config in_1Dsod1fl_dg --no-plot")
        return

    # Print numerical comparison
    print("\n" + "=" * 70)
    print("NUMERICAL COMPARISON (all methods vs FVM baseline)")
    print("=" * 70)

    baseline = datasets[0]
    print(f"\n{'Method':<15} {'Max |Δρ|':<14} {'Max |Δu|':<14} {'Max |Δp|':<14} {'Avg Rel Err':<14}")
    print("-" * 71)

    for data, label in zip(datasets, labels):
        drho = np.max(np.abs(baseline['rho'] - data['rho']))
        du = np.max(np.abs(baseline['u'] - data['u']))
        dp = np.max(np.abs(baseline['p'] - data['p']))

        # Average relative error
        err = 0
        for var in ['rho', 'u', 'p']:
            diff = np.abs(baseline[var] - data[var])
            l2 = np.sqrt(np.mean(diff**2))
            err += l2 / (np.sqrt(np.mean(baseline[var]**2)) + 1e-10)
        err /= 3

        print(f"{label:<15} {drho:<14.6e} {du:<14.6e} {dp:<14.6e} {err:<14.6e}")

    print("-" * 71)

    # Value ranges
    print(f"\n{'Method':<15} {'ρ_min':<10} {'ρ_max':<10} {'u_min':<10} {'u_max':<10} {'p_min':<12} {'p_max':<12}")
    print("-" * 79)
    for data, label in zip(datasets, labels):
        print(f"{label:<15} {data['rho'].min():<10.4f} {data['rho'].max():<10.4f} "
              f"{data['u'].min():<10.2f} {data['u'].max():<10.2f} "
              f"{data['p'].min():<12.1f} {data['p'].max():<12.1f}")

    # Plot
    if show_plot or save_plot:
        try:
            import matplotlib
            if save_plot and not show_plot:
                matplotlib.use('Agg')

            save_path = 'data/method_comparison.png' if save_plot else None
            plot_comparison(datasets, labels,
                          title="1D Sod Shock Tube: FVM vs DG Methods",
                          save_path=save_path,
                          show=show_plot)
        except ImportError:
            print("\nMatplotlib not available for plotting")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare FVM and DG simulation results')
    parser.add_argument('--plot', action='store_true', help='Show comparison plots')
    parser.add_argument('--save', action='store_true', help='Save plot to file')
    parser.add_argument('--all', action='store_true', help='Compare all available methods')
    parser.add_argument('--fvm', default='data/flow_1Dsod1fl.d', help='FVM output file')
    parser.add_argument('--dg', default='data/flow_1Dsod1fl_dg.d', help='DG output file')
    args = parser.parse_args()

    if args.all or args.plot or args.save:
        compare_all(show_plot=args.plot or args.all, save_plot=args.save)
    else:
        compare_two(args.fvm, args.dg)
