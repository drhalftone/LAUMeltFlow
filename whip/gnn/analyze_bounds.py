"""Analyze bead training CSV to find per-feature bounds for volume sampling."""

import sys
import csv
import math

csv_path = sys.argv[1] if len(sys.argv) > 1 else "../qt/bead_training.csv"

print(f"Loading {csv_path}...")

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    headers = next(reader)

skip = {"bead_id", "fixed", "has_left", "has_right"}
indices = [(i, h) for i, h in enumerate(headers) if h not in skip]

mins = [math.inf] * len(indices)
maxs = [-math.inf] * len(indices)
count = 0

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        for j, (i, _) in enumerate(indices):
            v = float(row[i])
            if v < mins[j]:
                mins[j] = v
            if v > maxs[j]:
                maxs[j] = v
        count += 1
        if count % 1000000 == 0:
            print(f"  {count:,} rows...")

print(f"  {count:,} samples, {len(headers)} columns\n")

print(f"{'feature':<22} {'min':>14} {'max':>14} {'range':>14}")
print("-" * 66)
for j, (i, name) in enumerate(indices):
    lo, hi = mins[j], maxs[j]
    print(f"{name:<22} {lo:>14.6e} {hi:>14.6e} {hi - lo:>14.6e}")
