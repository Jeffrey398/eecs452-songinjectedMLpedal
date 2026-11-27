import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
results = []

print(f"{'Drive':>7} {'Tone':>7} {'Train ESR':>12} {'Test ESR':>12} {'Converged':>10}")
print("-" * 55)

for g in range(6):
    for t in range(6):
        path = ROOT / f"bluesdriver_wh_g{g}_t{t}.json"

        if not path.exists():
            print(f"Missing: {path.name}")
            continue

        with open(path) as f:
            data = json.load(f)

        results.append(data)

        print(
            f"{data['drive']:7.2f} "
            f"{data['tone']:7.2f} "
            f"{data['train_esr']:12.6f} "
            f"{data['test_esr']:12.6f} "
            f"{str(data['optimizer_success']):>10}"
        )

if not results:
    raise SystemExit("No fitted models found")

train_esr = np.array([r["train_esr"] for r in results])
test_esr = np.array([r["test_esr"] for r in results])

best = min(results, key=lambda r: r["test_esr"])
worst = max(results, key=lambda r: r["test_esr"])
converged = sum(r["optimizer_success"] for r in results)

print("\nOverall performance")
print("-" * 55)
print(f"Models trained: {len(results)}/36")
print(f"Models converged: {converged}/{len(results)}")
print(f"Mean train ESR: {train_esr.mean():.6f}")
print(f"Mean test ESR:  {test_esr.mean():.6f}")
print(f"Median test ESR: {np.median(test_esr):.6f}")

print("\nBest configuration:")
print(f"Drive: {best['drive']:.2f}")
print(f"Tone:  {best['tone']:.2f}")
print(f"ESR:   {best['test_esr']:.6f}")

print("\nWorst configuration:")
print(f"Drive: {worst['drive']:.2f}")
print(f"Tone:  {worst['tone']:.2f}")
print(f"ESR:   {worst['test_esr']:.6f}")

print("\nConfigurations with test ESR > 0.10:")
for r in results:
    if r["test_esr"] > 0.10:
        print(
            f"Drive={r['drive']:.2f}, "
            f"Tone={r['tone']:.2f}, "
            f"ESR={r['test_esr']:.6f}"
        )

print("\nConfigurations that did not converge:")
for r in results:
    if not r["optimizer_success"]:
        print(
            f"Drive={r['drive']:.2f}, "
            f"Tone={r['tone']:.2f}, "
            f"ESR={r['test_esr']:.6f}"
        )
