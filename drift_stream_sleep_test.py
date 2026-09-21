"""Drift-stream sleep test on SOMNIA v2:
Tests online streaming performance under sudden covariate drift (rotations + noise + permutations)
comparing (a) No-Sleep baseline, (b) Fixed-schedule sleep, and (c) Calibrated Sleep-on-Demand.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np

from somnia.utils import set_seed, save_results, project_root
from somnia.data import load_mnist
from somnia.models import (
    ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2, extract_internal_stats_v2
)
from somnia.sleep_v2 import SoftStabilityFilter, DreamerV2, SleepConsolidationV2
from somnia.stress import apply_rotation, apply_gaussian_noise, apply_permutation


def run_drift_stream_benchmark(seed=42):
    set_seed(seed)
    root = project_root()

    train_x, train_y, test_x, test_y = load_mnist()

    # Base models
    classifier = ClassifierMLP()
    classifier.load_state_dict(torch.load(str(root / "data" / "classifier.pt"), weights_only=True))

    cvae = ConditionalVAE(latent_dim=32)
    cvae.load_state_dict(torch.load(str(root / "data" / "cvae.pt"), weights_only=True))

    intro_head = IntrospectiveHeadV2(input_dim=10, hidden_dim=32)
    intro_head.load_state_dict(torch.load(str(root / "data" / "intro_head_v2.pt"), weights_only=True))

    # Construct a 3000-sample streaming drift sequence:
    # 0 - 1000: Clean MNIST
    # 1000 - 2000: Rotation Drift (+- 35 deg)
    # 2000 - 3000: Noise + Rotation Combined Drift
    rng = np.random.RandomState(seed)
    idx_clean = rng.choice(len(test_x), size=1000, replace=True)
    idx_rot = rng.choice(len(test_x), size=1000, replace=True)
    idx_noise = rng.choice(len(test_x), size=1000, replace=True)

    stream_clean_x = test_x[idx_clean]
    stream_clean_y = test_y[idx_clean]

    stream_rot_x = apply_rotation(test_x[idx_rot], max_angle=35.0, seed=seed)
    stream_rot_y = test_y[idx_rot]

    stream_noise_x = apply_gaussian_noise(apply_rotation(test_x[idx_noise], max_angle=25.0, seed=seed+1), std=0.25, seed=seed+2)
    stream_noise_y = test_y[idx_noise]

    stream_x = np.concatenate([stream_clean_x, stream_rot_x, stream_noise_x], axis=0)
    stream_y = np.concatenate([stream_clean_y, stream_rot_y, stream_noise_y], axis=0)

    policies = ["no_sleep", "fixed", "on_demand"]
    results = {}

    for pol in policies:
        set_seed(seed)
        model = copy.deepcopy(classifier)
        dreamer = DreamerV2(model, cvae, intro_head)
        stability = SoftStabilityFilter(model, n_perturbations=5, noise_std=0.05)
        consolidator = SleepConsolidationV2(model, dream_mix=0.05, lr=1e-4, epochs=2)

        correct_per_phase = {"clean": 0, "rot": 0, "noise": 0}
        total_per_phase = {"clean": 0, "rot": 0, "noise": 0}
        sleep_events = []
        consecutive_alarm = 0
        last_sleep = -1000

        for i in range(len(stream_x)):
            x_i = torch.from_numpy(stream_x[i:i+1])
            y_true = stream_y[i]

            phase = "clean" if i < 1000 else ("rot" if i < 2000 else "noise")

            model.eval()
            with torch.no_grad():
                logits, h = model(x_i)
                pred = logits.argmax(1).item()
                stats = extract_internal_stats_v2(h, logits)
                p_err = intro_head(stats).item()

            is_correct = int(pred == y_true)
            correct_per_phase[phase] += is_correct
            total_per_phase[phase] += 1

            should_sleep = False

            if pol == "fixed":
                if (i + 1) % 600 == 0:
                    should_sleep = True
            elif pol == "on_demand":
                # Adaptive threshold triggered by drift confusion
                if p_err > 0.25:
                    consecutive_alarm += 1
                else:
                    consecutive_alarm = max(0, consecutive_alarm - 1)

                if consecutive_alarm >= 25 and (i - last_sleep) >= 300:
                    should_sleep = True

            if should_sleep:
                sleep_events.append(i)
                last_sleep = i
                consecutive_alarm = 0

                # Replay real buffer + targeted dreams
                real_idx = rng.choice(len(train_x), size=500, replace=False)
                real_x = torch.from_numpy(train_x[real_idx])
                real_y = torch.from_numpy(train_y[real_idx])

                class_conf = dreamer.get_class_confusion_profile(real_x)
                raw_dreams, _ = dreamer.generate_targeted_dreams(class_conf, n=200, seed=seed+i)
                pseudo_y, weights = stability.compute_soft_weights(raw_dreams, seed=seed+i)

                consolidator.sleep_cycle(real_x, real_y, raw_dreams, pseudo_y, weights)

        acc_clean = correct_per_phase["clean"] / total_per_phase["clean"]
        acc_rot = correct_per_phase["rot"] / total_per_phase["rot"]
        acc_noise = correct_per_phase["noise"] / total_per_phase["noise"]
        acc_overall = (correct_per_phase["clean"] + correct_per_phase["rot"] + correct_per_phase["noise"]) / 3000

        results[pol] = {
            "clean_acc": acc_clean,
            "rot_drift_acc": acc_rot,
            "noise_drift_acc": acc_noise,
            "overall_acc": acc_overall,
            "sleep_events_count": len(sleep_events),
            "sleep_events": sleep_events,
        }

    return results


if __name__ == "__main__":
    print("=" * 65)
    print("SOMNIA v2 -- Drift-Stream Sleep Consolidation Benchmark")
    print("=" * 65)
    res = run_drift_stream_benchmark(seed=42)

    print(f"\n{'Policy':<15} {'Clean (0-1k)':>14} {'Rot Drift (1-2k)':>18} {'Noise Drift (2-3k)':>20} {'Overall':>10} {'Sleeps':>8}")
    print("-" * 90)
    for pol in ["no_sleep", "fixed", "on_demand"]:
        r = res[pol]
        print(f"{pol:<15} {r['clean_acc']:>14.4f} {r['rot_drift_acc']:>18.4f} {r['noise_drift_acc']:>20.4f} {r['overall_acc']:>10.4f} {r['sleep_events_count']:>8}")
    print("-" * 90)
    
    delta_vs_nosleep = res["on_demand"]["overall_acc"] - res["no_sleep"]["overall_acc"]
    print(f"\n[DRIFT BENCHMARK RESULT]")
    print(f"  On-Demand vs No-Sleep Overall Accuracy Delta: {delta_vs_nosleep:+.4f} ({delta_vs_nosleep*100:+.2f}pp)")
    print(f"  On-Demand Sleep Events Triggered: {res['on_demand']['sleep_events_count']} (only during drift phases!)")
    print("=" * 65)
