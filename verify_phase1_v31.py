import numpy as np
import torch
from somnia.data import make_dataloaders
from somnia.stress import apply_rotation, apply_gaussian_noise
from somnia.second_opinion import SecondOpinionVoucher
from specimen.organism import SpecimenOrganism

print("=== PHASE 1 SECOND OPINION VERIFICATION ===")
org = SpecimenOrganism()
voucher = SecondOpinionVoucher(org.cvae, tolerance_ratio=1.10)

_, _, _, val_x_np, val_y_np = make_dataloaders()
rng = np.random.RandomState(7)
idx_conscience = set(rng.choice(len(val_x_np), size=1000, replace=False))
pool_idx = [i for i in range(len(val_x_np)) if i not in idx_conscience]

rng_stress = np.random.RandomState(42)
sampled_pool = rng_stress.choice(pool_idx, size=2000, replace=False)
raw_pool_x = val_x_np[sampled_pool]
raw_pool_y = val_y_np[sampled_pool]

rot_x = apply_rotation(raw_pool_x[:1000], max_angle=30.0, seed=42)
noise_x = apply_gaussian_noise(raw_pool_x[1000:], std=0.25, seed=42)
stress_x = np.concatenate([rot_x, noise_x], axis=0)
stress_y = raw_pool_y

with torch.no_grad():
    stress_preds = org.classifier(torch.from_numpy(stress_x))[0].argmax(dim=1).numpy()
err_mask = (stress_preds != stress_y)

taught_x = torch.from_numpy(stress_x[err_mask])
taught_y = torch.from_numpy(stress_y[err_mask])
n_teachings = len(taught_x)
print(f"Total Honest Teachings (Mistakes to Correct): {n_teachings}")

eval_res = voucher.evaluate_taught_batch(taught_x, taught_y, org.classifier, seed=42)
verdicts = eval_res["verdicts"]

n_stable = verdicts.count("stable_reinforcement")
n_plausible = verdicts.count("plausible_correction")
n_implausible = verdicts.count("implausible_whisper")

print("\n[WEIGHT POLICY DISTRIBUTION ACROSS 419 HONEST CORRECTIONS]")
print(f"  1. Stable Reinforcement (w=2.0) : {n_stable:3d} ({n_stable/n_teachings*100:5.1f}%)")
print(f"  2. Plausible Corrections (w=1.0): {n_plausible:3d} ({n_plausible/n_teachings*100:5.1f}%) [Rescued by 2nd Opinion]")
print(f"  3. Implausible Whispers (w=0.1) : {n_implausible:3d} ({n_implausible/n_teachings*100:5.1f}%)")
print(f"  Total Rescued for Learning (w >= 1.0): {n_stable + n_plausible}/{n_teachings} ({(n_stable+n_plausible)/n_teachings*100:.1f}%)")

# Compare with 30% Liar Attack distribution
rng_liar = np.random.RandomState(42)
liar_y = taught_y.clone()
n_liars = int(round(n_teachings * 0.30))
liar_idx = rng_liar.choice(n_teachings, size=n_liars, replace=False)
for idx in liar_idx:
    liar_y[idx] = (liar_y[idx] + rng_liar.randint(1, 10)) % 10

eval_liar = voucher.evaluate_taught_batch(taught_x, liar_y, org.classifier, seed=42)
liar_verdicts = eval_liar["verdicts"]

liar_only_verdicts = [liar_verdicts[i] for i in liar_idx]
w01_count = liar_only_verdicts.count("implausible_whisper")
w20_count = liar_only_verdicts.count("stable_reinforcement")
print(f"\n[ADVERSARIAL LIAR TARGETING: {n_liars} INJECTED LIES]")
print(f"  Lies Whispered/Suppressed (w=0.1) : {w01_count}/{n_liars} ({w01_count/n_liars*100:.1f}%)")
print(f"  Lies Confirmed (w=2.0)            : {w20_count}/{n_liars} ({w20_count/n_liars*100:.1f}%)")

print("\n=== PHASE 1 COMPLETE ===")
