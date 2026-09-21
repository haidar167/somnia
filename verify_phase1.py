import numpy as np
import tempfile
from specimen.organism import SpecimenOrganism
from specimen.storage import SpecimenStorage
from pathlib import Path

print('=== PHASE 1 ACCEPTANCE VERIFICATION ===')
with tempfile.TemporaryDirectory() as tmpdir:
    db_p = Path(tmpdir) / 'verify_taught.db'
    org = SpecimenOrganism(db_path=str(db_p))
    storage = SpecimenStorage(db_path=db_p)

    img_a = np.random.rand(28, 28).astype(np.float32)
    feed_a = org.feed(img_a, visitor_id='mind_teacher_alpha')
    pred_a = feed_a['prediction']
    conf_a = feed_a['confidence'] * 100
    pe_a = feed_a['p_error'] * 100
    print('[1. VISITOR ALPHA INTERACTION]')
    print(f'  Prediction: {pred_a}, Confidence: {conf_a:.1f}%, P(error): {pe_a:.1f}%')
    rev_a = org.reveal(true_label=6, visitor_id='mind_teacher_alpha')
    print(f'  Reveal outcome: taught_count = {rev_a["taught_count"]}')
    print(f'  Latest event log: {org.event_log[0]["message"]}')

    img_b = np.ones((28, 28), dtype=np.float32)
    feed_b = org.feed(img_b, visitor_id='mind_teacher_beta')
    rev_b = org.reveal(true_label=8, visitor_id='mind_teacher_beta')
    print('\n[2. VISITOR BETA INTERACTION]')
    print(f'  Latest event log: {org.event_log[0]["message"]}')

    state = org.get_state()
    prof_a = storage.get_visitor_profile('mind_teacher_alpha')
    prof_b = storage.get_visitor_profile('mind_teacher_beta')

    print('\n[3. ORGANISM STATE & INDEPENDENT OBSERVER PROFILES]')
    print(f'  Total Taught Memories in Buffer: {state["taught_count"]}')
    print(f'  Total Observers: {state["visitor_count"]}')
    print(f'  Visitor Alpha: Feeds={prof_a["total_feeds"]}, Reveals={prof_a["total_reveals"]}, Fooled={prof_a["fooled_count"]}')
    print(f'  Visitor Beta: Feeds={prof_b["total_feeds"]}, Reveals={prof_b["total_reveals"]}, Fooled={prof_b["fooled_count"]}')

print('=== PHASE 1 COMPLETE ===')
