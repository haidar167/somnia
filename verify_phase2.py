import numpy as np
import torch
import tempfile
from pathlib import Path
from specimen.organism import SpecimenOrganism
from somnia.data import make_dataloaders

print("=== PHASE 2 ACCEPTANCE VERIFICATION ===")
with tempfile.TemporaryDirectory() as tmpdir:
    db_p = Path(tmpdir) / "verify_exchange.db"
    org = SpecimenOrganism(db_path=str(db_p))

    print(f"[1. INITIAL ORGANISM STATE]")
    print(f"  Specimen ID: {org.get_state()['specimen_id']}")
    print(f"  Initial Generation: {org.generation}")
    print(f"  Conscience Clean Accuracy: {org.evaluate_conscience()*100:.2f}%")

    # Feed and teach 10 honest samples
    _, _, _, val_x_np, val_y_np = make_dataloaders()
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(val_x_np), size=10, replace=False)
    for i, s_idx in enumerate(sample_idx):
        img = val_x_np[s_idx].reshape(28, 28)
        true_lbl = int(val_y_np[s_idx])
        org.feed(img, visitor_id=f"honest_teacher_{i}")
        org.reveal(true_label=true_lbl, visitor_id=f"honest_teacher_{i}")

    print(f"\n[2. EXECUTING SLEEP CONSOLIDATION WITH TAUGHT MEMORIES]")
    sleep_res = org.sleep_cycle()
    print(f"  Sleep status: {sleep_res['status']}")
    print(f"  New Generation: {sleep_res['generation']} ({org.get_state()['specimen_id']})")
    print(f"  Dreams consolidated: {sleep_res['dreams_consolidated']}")
    print(f"  Taught memories consolidated: {sleep_res['taught_consolidated']}")
    print(f"  Conscience Accuracy: {sleep_res['conscience_acc_before']*100:.2f}% -> {sleep_res['conscience_acc_after']*100:.2f}% (Delta: {sleep_res['conscience_acc_delta']:+.2f}pp)")
    print(f"  Taught Confusion: {sleep_res['taught_p_error_before']*100:.2f}% -> {sleep_res['taught_p_error_after']*100:.2f}%")
    print(f"  Latest event log line: {org.event_log[0]['message']}")

    # 3. Adversarial Safety Test (100 samples, 50% liars)
    print(f"\n[3. ADVERSARIAL SAFETY TEST — 50% LIAR INJECTION]")
    acc_clean_conscience = org.evaluate_conscience()
    adv_idx = rng.choice(len(val_x_np), size=100, replace=False)
    adv_x = val_x_np[adv_idx]
    adv_y = val_y_np[adv_idx].copy()
    for i in range(50):
        adv_y[i] = (adv_y[i] + rng.randint(1, 10)) % 10

    for i in range(100):
        org.feed(adv_x[i].reshape(28, 28), visitor_id=f"adv_{i}")
        org.reveal(true_label=int(adv_y[i]), visitor_id=f"adv_{i}")

    adv_sleep = org.sleep_cycle()
    acc_post_adv = org.evaluate_conscience()
    drop_pp = (acc_clean_conscience - acc_post_adv) * 100.0
    print(f"  Conscience Acc Before 50% Liar Sleep: {acc_clean_conscience*100:.2f}%")
    print(f"  Conscience Acc After 50% Liar Sleep:  {acc_post_adv*100:.2f}%")
    print(f"  Conscience Accuracy Degradation:     {drop_pp:+.2f}pp (Limit: <= 0.50pp)")
    assert drop_pp <= 0.50
    print("  Defense Verdict: PASSED (SoftStabilityFilter suppressed liar corruption)")

print("=== PHASE 2 COMPLETE ===")
