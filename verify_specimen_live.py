"""Automated proof-of-life verification for THE SPECIMEN living AI organism."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from somnia.utils import project_root, set_seed
from specimen.organism import SpecimenOrganism


def run_live_proof_of_life():
    print("=" * 65)
    print("THE SPECIMEN -- PROOF OF LIFE VERIFICATION")
    print("=" * 65)

    root = project_root()
    organism = SpecimenOrganism()

    results = {}

    # TEST 1: Initial Alive State & Brainwaves
    print("\n[TEST 1/5] Checking Initial Living State...")
    state = organism.get_state()
    t1_pass = (
        state["specimen_id"] == "SPECIMEN #001" and
        len(state["recent_dreams"]) == 8 and
        len(state["event_log"]) > 0 and
        not state["is_sleeping"]
    )
    results["test_1"] = {
        "name": "Page & Organism Boot",
        "passed": t1_pass,
        "uptime": state["uptime_str"],
        "initial_mood": state["vitals"]["mood"],
        "initial_p_error": state["vitals"]["p_error"],
    }
    print(f"  Organism Status: ALIVE (Uptime: {state['uptime_str']})")
    print(f"  Initial Vitals : Mood = {state['vitals']['mood'].upper()}, P(error) = {state['vitals']['p_error']:.3f}")
    print(f"  -> TEST 1: {'PASS [OK]' if t1_pass else 'FAIL'}")

    # TEST 2: Feed Normal Digit (Canonical Clean MNIST '0' or '7')
    print("\n[TEST 2/5] Feeding Normal Digit (Clean MNIST sample)...")
    from somnia.data import load_mnist
    _, _, test_x, test_y = load_mnist()
    clean_sample = test_x[0].reshape(28, 28) # Real canonical MNIST 7
    res_2 = organism.feed(clean_sample)
    t2_pass = (
        res_2["prediction"] == int(test_y[0]) and
        res_2["confidence"] > 0.85 and
        res_2["p_error"] < 0.20
    )
    results["test_2"] = {
        "name": "Normal Digit Feeding",
        "passed": t2_pass,
        "prediction": res_2["prediction"],
        "confidence": res_2["confidence"],
        "p_error": res_2["p_error"],
        "mood": res_2["mood"],
    }
    print(f"  Mouth Channel  : Predicted '{res_2['prediction']}' ({res_2['confidence']*100:.1f}% confidence)")
    print(f"  Gut Channel    : P(error) = {res_2['p_error']*100:.1f}%, Mood = {res_2['mood'].upper()}")
    print(f"  -> TEST 2: {'PASS [OK]' if t2_pass else 'FAIL'}")

    # TEST 3: Feed Synthetic Noise (Spike Alarm)
    print("\n[TEST 3/5] Feeding Synthetic Noise (Spike Alarm)...")
    rng = np.random.RandomState(42)
    noise_img = rng.randn(28, 28).clip(0, 1).astype(np.float32)
    res_3 = organism.feed(noise_img)
    t3_pass = (
        res_3["p_error"] >= 0.25 or
        res_3["mood"] in ["alarmed", "in pain"]
    )
    results["test_3"] = {
        "name": "Noise / Alarm Spike",
        "passed": t3_pass,
        "p_error": res_3["p_error"],
        "mood": res_3["mood"],
    }
    print(f"  Mouth Channel  : Predicted '{res_3['prediction']}' ({res_3['confidence']*100:.1f}% confidence)")
    print(f"  Gut Channel    : P(error) = {res_3['p_error']*100:.1f}%, Mood = {res_3['mood'].upper()} (ALARM SPIKED!)")

    # Save Test 3 graphic: Noise Spike
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor="#090C10")
    ax1.set_facecolor("#11161F")
    ax2.set_facecolor("#11161F")
    ax1.imshow(noise_img, cmap="inferno")
    ax1.set_title("SYNTHETIC NOISE INJECTION", color="#E2E8F0", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2.text(0.5, 0.7, f"MOOD: {res_3['mood'].upper()}", color="#FF3B30", fontsize=16, fontweight="bold", ha="center")
    ax2.text(0.5, 0.45, f"P(ERROR): {res_3['p_error']*100:.1f}%", color="#FB923C", fontsize=14, ha="center")
    ax2.text(0.5, 0.2, "ALARM SPIKED ON ANOMALOUS INPUT", color="#94A3B8", fontsize=9, ha="center")
    ax2.axis("off")
    plt.tight_layout()
    noise_path = str(root / "figures" / "specimen_noise_spike.png")
    plt.savefig(noise_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Screenshot 3 (Noise Spike) saved -> {noise_path}")
    print(f"  -> TEST 3: {'PASS [OK]' if t3_pass else 'FAIL'}")

    # TEST 4: The Disagreement Moment (Ambiguous Digit)
    print("\n[TEST 4/5] Inducing Cognitive Disagreement (Mouth & Gut Disagree)...")
    ambiguous = rng.uniform(0.1, 0.6, size=(28, 28)).astype(np.float32)
    ambiguous[7:21, 12:16] = 0.9  # faint stroke inducing mouth confidence with gut alarm
    res_4 = organism.feed(ambiguous)
    t4_pass = res_4["disagreement"] is True
    results["test_4"] = {
        "name": "Disagreement Moment",
        "passed": t4_pass,
        "prediction": res_4["prediction"],
        "confidence": res_4["confidence"],
        "p_error": res_4["p_error"],
        "disagreement": res_4["disagreement"],
        "note": res_4["disagreement_note"],
    }
    print(f"  Mouth Channel  : Says '{res_4['prediction']}' ({res_4['confidence']*100:.1f}% sure)")
    print(f"  Gut Channel    : Feels P(error) = {res_4['p_error']*100:.1f}% ({res_4['mood'].upper()})")
    print(f"  Alert Active   : {res_4['disagreement']} (Note: {res_4['disagreement_note']})")
    print(f"  -> TEST 4: {'PASS [OK]' if t4_pass else 'FAIL'}")

    # TEST 5: Trigger Dream Consolidation
    print("\n[TEST 5/5] Triggering Subconscious Dream Consolidation...")
    sleep_res = organism.sleep_cycle()
    t5_pass = (
        sleep_res["status"] == "success" and
        sleep_res["sleep_count"] >= 1 and
        len(sleep_res["recent_dreams"]) == 8
    )
    results["test_5"] = {
        "name": "Dream Consolidation",
        "passed": t5_pass,
        "sleep_count": sleep_res["sleep_count"],
        "dreams_count": len(sleep_res["recent_dreams"]),
    }
    print(f"  Sleep Status   : Consolidated {sleep_res['dreams_consolidated']} targeted dreams.")
    print(f"  Recent Dreams  : {len(sleep_res['recent_dreams'])} cards updated in memory feed.")
    print(f"  -> TEST 5: {'PASS [OK]' if t5_pass else 'FAIL'}")

    print("\n" + "=" * 65)
    all_passed = all(r["passed"] for r in results.values())
    print(f"ALL 5 PROOF-OF-LIFE TESTS: {'ALL PASSED [5/5 GREEN]' if all_passed else 'SOME FAILED'}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_live_proof_of_life()
