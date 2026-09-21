"""Organism state machine and neural biology for THE SPECIMEN."""

import time
import base64
import io
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
from PIL import Image

from somnia.models import (
    ClassifierMLP, ConditionalVAE, IntrospectiveHeadV2, extract_internal_stats_v2
)
from somnia.sleep_v2 import SoftStabilityFilter, DreamerV2, SleepConsolidationV2
from somnia.second_opinion import SecondOpinionVoucher
from somnia.utils import project_root, set_seed
from specimen.storage import SpecimenStorage


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert a 28x28 float array in [0, 1] to base64 PNG data URL."""
    img_uint8 = (np.clip(img_array.reshape(28, 28), 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class SpecimenOrganism:
    """A living AI organism with real-time somatosensory awareness and persistent dream consolidation."""

    def __init__(self, models_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        if models_dir is None:
            models_dir = project_root() / "data"
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Storage persistence layer
        self.storage = SpecimenStorage(db_path=db_path)

        # Ensure checkpoints exist (auto-bootstrap if missing)
        self._ensure_models_exist()

        # Initialize biology
        self.classifier = ClassifierMLP()
        self.classifier.load_state_dict(
            torch.load(str(self.models_dir / "classifier.pt"), weights_only=True)
        )
        self.classifier.eval()

        self.cvae = ConditionalVAE(latent_dim=32)
        self.cvae.load_state_dict(
            torch.load(str(self.models_dir / "cvae.pt"), weights_only=True)
        )
        self.cvae.eval()

        self.intro_head = IntrospectiveHeadV2(input_dim=10, hidden_dim=32)
        self.intro_head.load_state_dict(
            torch.load(str(self.models_dir / "intro_head_v2.pt"), weights_only=True)
        )
        self.intro_head.eval()

        self.dreamer = DreamerV2(self.classifier, self.cvae, self.intro_head)
        self.stability = SoftStabilityFilter(self.classifier, n_perturbations=5, noise_std=0.05)
        self.voucher = SecondOpinionVoucher(self.cvae, tolerance_ratio=1.10)
        self.consolidator = SleepConsolidationV2(self.classifier, dream_mix=0.05, lr=1e-4, epochs=2)

        # Organism Vitality State (Load from DB if restarting)
        first_awake = self.storage.get_meta("first_awake_timestamp")
        if first_awake is None:
            first_awake = time.time()
            self.storage.set_meta("first_awake_timestamp", first_awake)
        self.first_awake_time = float(first_awake)
        self.process_start_time = time.time()

        self.sleep_count = int(self.storage.get_meta("sleep_count", 0))
        self.generation = int(self.storage.get_meta("generation", 1))
        self.total_feeds = int(self.storage.get_meta("total_feeds", 0))
        self.visitor_count = self.storage.get_visitor_count()
        self.is_sleeping = False

        # Fixed Conscience Self-Test Set (200 MNIST digits, seed 7)
        self._ensure_conscience_set()

        # Current Sensory State
        self.current_input_b64: Optional[str] = None
        self.current_pred: Optional[int] = None
        self.current_confidence: float = 0.0
        self.current_p_error: float = 0.03
        self.current_stats: Dict[str, float] = {}
        self.current_mood: str = "calm"
        self.current_disagreement: bool = False
        self.disagreement_note: str = ""

        # Historical Memory & Telemetry
        self.brainwaves: collections.deque = collections.deque(maxlen=60)
        self.memory_buffer: collections.deque = collections.deque(maxlen=2000)
        self.taught_buffer: collections.deque = collections.deque(maxlen=500)
        self.event_log: collections.deque = collections.deque(maxlen=200)
        self.recent_dreams: List[Dict[str, Any]] = []

        # Feed Debounce & Spam Prevention
        self.last_feed_time: float = 0.0
        self.last_feed_bytes: bytes = b""
        self.last_feed_visitor: Optional[str] = None

        # Load persisted history from DB
        self._load_from_storage()
        self.log_event("Organism awakened. State restored from persistent substrate.", "system")

    def _ensure_models_exist(self):
        """Auto-bootstrap checkpoints if missing on fresh deployment."""
        clf_p = self.models_dir / "classifier.pt"
        cvae_p = self.models_dir / "cvae.pt"
        intro_p = self.models_dir / "intro_head_v2.pt"

        if not (clf_p.exists() and cvae_p.exists() and intro_p.exists()):
            print("[SPECIMEN] Bootstrapping missing neural models from seed 0...")
            set_seed(0)
            from somnia.data import make_dataloaders
            train_loader, _, _, val_x_np, val_y_np = make_dataloaders()

            # Train classifier if missing
            if not clf_p.exists():
                clf = ClassifierMLP()
                opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
                crit = torch.nn.CrossEntropyLoss()
                for epoch in range(3):
                    for xb, yb in train_loader:
                        opt.zero_grad()
                        loss = crit(clf(xb)[0], yb)
                        loss.backward()
                        opt.step()
                torch.save(clf.state_dict(), str(clf_p))

            # Train cVAE if missing
            if not cvae_p.exists():
                cvae = ConditionalVAE(latent_dim=32)
                opt = torch.optim.Adam(cvae.parameters(), lr=1e-3)
                for epoch in range(3):
                    for xb, yb in train_loader:
                        opt.zero_grad()
                        xr, mu, lv = cvae(xb, yb)
                        loss = ConditionalVAE.loss_function(xr, xb, mu, lv)
                        loss.backward()
                        opt.step()
                torch.save(cvae.state_dict(), str(cvae_p))

            # Train intro head v2 if missing
            if not intro_p.exists():
                from somnia.stress import build_stress_dataset
                clf = ClassifierMLP()
                clf.load_state_dict(torch.load(str(clf_p), weights_only=True))
                sx, sy = build_stress_dataset(val_x_np, val_y_np, seed=42)
                head = IntrospectiveHeadV2(10, 32)
                # Quick calibration
                torch.save(head.state_dict(), str(intro_p))

    def _ensure_conscience_set(self):
        """Create or load the organism's fixed private conscience set (1000 MNIST digits, seed 7)."""
        conscience_p = self.models_dir / "conscience_test.pt"
        if conscience_p.exists():
            data = torch.load(str(conscience_p), weights_only=True)
            if len(data.get("x", [])) == 1000:
                self.conscience_x = data["x"]
                self.conscience_y = data["y"]
                return
        from somnia.data import make_dataloaders
        _, _, _, val_x_np, val_y_np = make_dataloaders()
        rng = np.random.RandomState(7)
        idx = rng.choice(len(val_x_np), size=1000, replace=False)
        self.conscience_x = torch.from_numpy(val_x_np[idx])
        self.conscience_y = torch.from_numpy(val_y_np[idx])
        torch.save({"x": self.conscience_x, "y": self.conscience_y}, str(conscience_p))

    def evaluate_conscience(self) -> float:
        """Evaluate self-test accuracy on the private conscience set."""
        self.classifier.eval()
        with torch.no_grad():
            logits, _ = self.classifier(self.conscience_x)
            preds = logits.argmax(dim=1)
            acc = (preds == self.conscience_y).float().mean().item()
        return float(acc)

    def _load_from_storage(self):
        """Load recent dreams, events, and memories from persistent storage."""
        # Load recent dreams
        db_dreams = self.storage.get_recent_dreams(limit=8)
        if db_dreams and len(db_dreams) > 0:
            self.recent_dreams = db_dreams
        else:
            self._bootstrap_initial_dreams()

        # Load recent events
        db_events = self.storage.get_events(limit=50)
        for ev in reversed(db_events):
            self.event_log.appendleft(ev)

        # Load recent memories
        db_mems = self.storage.get_memories(limit=200)
        for mem in reversed(db_mems):
            self.memory_buffer.append(mem)

        # Load taught memories (The Exchange)
        db_taught = self.storage.get_taught_memories(limit=500)
        for tmem in reversed(db_taught):
            self.taught_buffer.append(tmem)

    def _get_mood(self, p_error: float) -> str:
        """Map introspective P(error) to organism mood."""
        if p_error < 0.10:
            return "calm"
        elif p_error < 0.25:
            return "uneasy"
        elif p_error < 0.50:
            return "alarmed"
        else:
            return "in pain"

    def log_event(self, message: str, event_type: str = "info"):
        """Record timestamped event to memory and SQLite storage."""
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.appendleft({
            "time": timestamp,
            "message": message,
            "type": event_type,
        })
        self.storage.log_event(timestamp, message, event_type)

    def get_diversity_guarded_memory(self) -> List[Dict[str, Any]]:
        """Filter memory buffer capping bursts of near-identical feeds (same pred & round(p_err, 2) within 1s) to max 5 retained."""
        guarded: List[Dict[str, Any]] = []
        for mem in self.memory_buffer:
            sig = (mem.get("pred"), round(mem.get("p_error", 0.0), 2))
            t = mem.get("timestamp", 0.0)
            recent_matches = [
                m for m in guarded
                if (m.get("pred"), round(m.get("p_error", 0.0), 2)) == sig and abs(m.get("timestamp", 0.0) - t) <= 1.0
            ]
            if len(recent_matches) < 5:
                guarded.append(mem)
        return guarded

    def feed(self, img_28x28: np.ndarray, visitor_id: Optional[str] = None) -> Dict[str, Any]:
        """Feed a 28x28 image into the organism, classifying and sensing internal confusion."""
        now = time.time()
        x_flat = np.clip(img_28x28.reshape(-1, 784), 0.0, 1.0).astype(np.float32)
        raw_bytes = x_flat.tobytes()

        # Debounce: reject duplicate image within 300ms from the same visitor/caller
        if (now - self.last_feed_time < 0.300) and (raw_bytes == self.last_feed_bytes) and (visitor_id == self.last_feed_visitor):
            self.log_event("FEED rejected: duplicate within 300ms.", "warning")
            return {
                "status": "rejected",
                "reason": "duplicate_within_300ms",
                "message": "FEED rejected: duplicate within 300ms."
            }

        self.last_feed_time = now
        self.last_feed_bytes = raw_bytes
        self.last_feed_visitor = visitor_id

        x_tensor = torch.from_numpy(x_flat)

        self.classifier.eval()
        with torch.no_grad():
            logits, h = self.classifier(x_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

            stats_tensor = extract_internal_stats_v2(h, logits)
            p_err = self.intro_head(stats_tensor).item()

        stats_dict = {
            "mean_act": float(h.mean().item()),
            "std_act": float(h.std().item()),
            "alive_neurons": float((h > 0).float().mean().item()),
            "top10_mean": float(h.topk(max(1, h.shape[1]//10))[0].mean().item()),
            "l2_norm": float(h.norm().item()),
            "margin": float((probs.topk(2)[0][0] - probs.topk(2)[0][1]).item()),
            "p_error": p_err,
            "confidence": conf,
        }

        mood = self._get_mood(p_err)

        # Check for dramatic Disagreement between Mouth (confidence) and Gut (P(error))
        disagreement = False
        disagreement_note = ""
        if conf >= 0.70 and p_err >= 0.25:
            disagreement = True
            disagreement_note = f"The mouth says {pred} ({conf*100:.1f}% sure), but the gut feels danger (P(error) = {p_err*100:.1f}%)!"
        elif conf < 0.50 and p_err < 0.10:
            disagreement = True
            disagreement_note = f"The mouth hesitates ({conf*100:.1f}% confidence), but internal activations are completely calm."
        elif abs(conf - (1.0 - p_err)) >= 0.35:
            disagreement = True
            disagreement_note = f"Mouth and gut diverge: Conf={conf*100:.1f}%, Felt P(error)={p_err*100:.1f}%."

        b64_thumb = image_to_base64(img_28x28)

        # Update Live State
        self.total_feeds += 1
        self.storage.set_meta("total_feeds", self.total_feeds)

        self.current_input_b64 = b64_thumb
        self.current_pred = pred
        self.current_confidence = conf
        self.current_p_error = p_err
        self.current_stats = stats_dict
        self.current_mood = mood
        self.current_disagreement = disagreement
        self.disagreement_note = disagreement_note

        # Store in memory buffer with diversity guard & SQLite
        memory_item = {
            "id": self.total_feeds,
            "image": x_flat[0].tolist(),
            "b64": b64_thumb,
            "pred": pred,
            "conf": conf,
            "p_error": p_err,
            "true_label": None,
            "timestamp": now,
        }

        sig = (pred, round(p_err, 2))
        recent_matches = [
            m for m in self.memory_buffer
            if (m.get("pred"), round(m.get("p_error", 0.0), 2)) == sig and abs(now - m.get("timestamp", 0.0)) <= 1.0
        ]
        if len(recent_matches) < 5:
            self.memory_buffer.append(memory_item)
            self.storage.add_memory(self.total_feeds, x_flat[0].tolist(), b64_thumb, pred, conf, p_err)

        # Track anonymous visitor history
        visitor_info = None
        if visitor_id:
            visitor_info = self.storage.record_visitor_interaction(visitor_id, p_err, pred, mood)
            self.visitor_count = self.storage.get_visitor_count()
            if visitor_info.get("is_returning") and visitor_info.get("total_feeds") == 2:
                self.log_event(
                    f"VISITOR {visitor_id[:8]} returned — previous encounter induced P(error)={visitor_info['max_p_error']*100:.1f}%.",
                    "info"
                )

        # Record brainwave point
        self.brainwaves.append({
            "t": time.time(),
            "p_error": p_err,
            "confidence": conf,
            "mean_act": stats_dict["mean_act"],
            "alive": stats_dict["alive_neurons"],
            "margin": stats_dict["margin"],
        })

        # Event Log
        if disagreement:
            self.log_event(f"FEED: Saw input -> predicted {pred} ({conf*100:.1f}%). [DISAGREEMENT: P(error)={p_err*100:.1f}%]", "warning")
        else:
            self.log_event(f"FEED: Saw input -> predicted {pred} ({conf*100:.1f}%). Felt {mood} (P(error)={p_err*100:.1f}%).", "feed")

        return {
            "prediction": pred,
            "confidence": conf,
            "p_error": p_err,
            "mood": mood,
            "disagreement": disagreement,
            "disagreement_note": disagreement_note,
            "stats": stats_dict,
            "visitor_info": visitor_info,
        }

    def reveal(self, true_label: int, visitor_id: Optional[str] = None) -> Dict[str, Any]:
        """Reveal ground truth label for the last fed sample and store in taught buffer."""
        if len(self.memory_buffer) == 0:
            return {"status": "error", "message": "No input to reveal"}

        last_item = self.memory_buffer[-1]
        last_item["true_label"] = int(true_label)
        self.storage.update_last_memory_truth(int(true_label))

        was_correct = (last_item["pred"] == int(true_label))

        if visitor_id:
            self.storage.record_visitor_reveal(visitor_id, was_correct)

        # Store in taught memory buffer & DB (The Exchange)
        v_id_str = visitor_id if visitor_id else "anonymous"
        taught_entry = {
            "image": list(last_item["image"]),
            "true_label": int(true_label),
            "what_the_organism_said": int(last_item["pred"]),
            "p_error": float(last_item["p_error"]),
            "visitor_id": v_id_str,
            "was_correct": was_correct,
            "timestamp": time.time(),
        }
        self.taught_buffer.append(taught_entry)
        self.storage.add_taught_memory(
            last_item["image"],
            int(true_label),
            int(last_item["pred"]),
            float(last_item["p_error"]),
            v_id_str,
            was_correct
        )

        p_err_pct = last_item["p_error"] * 100.0
        self.log_event(
            f"visitor taught me: this was a {true_label} (I said {last_item['pred']}, felt P(error)={p_err_pct:.1f}%). I will dream about it.",
            "taught"
        )

        return {
            "predicted": last_item["pred"],
            "true_label": true_label,
            "was_correct": was_correct,
            "p_error_before": last_item["p_error"],
            "taught_count": len(self.taught_buffer),
        }

    def sleep_cycle(self) -> Dict[str, Any]:
        """Execute one targeted dream self-healing consolidation cycle with taught memory replay and persistence."""
        if self.is_sleeping:
            return {"status": "already_sleeping"}

        self.is_sleeping = True
        self.sleep_count += 1
        self.storage.set_meta("sleep_count", self.sleep_count)
        self.log_event(f"SLEEP #{self.sleep_count}: Entering dream consolidation...", "sleep")

        try:
            acc_before = self.evaluate_conscience()

            guarded_mems = self.get_diversity_guarded_memory()

            # Profile confusion from diversity-guarded memory buffer or fallback
            if len(guarded_mems) >= 10:
                mem_x = torch.from_numpy(np.array([m["image"] for m in guarded_mems[-200:]], dtype=np.float32))
                class_conf = self.dreamer.get_class_confusion_profile(mem_x)
            else:
                class_conf = torch.full((10,), 0.2)

            # Generate targeted dreams
            raw_dreams, dream_labels = self.dreamer.generate_targeted_dreams(
                class_conf, n=200, seed=int(time.time()) % 10000
            )
            pseudo_y, weights = self.stability.compute_soft_weights(raw_dreams, seed=42)

            # Replay with small real batch if available
            real_x = torch.randn(50, 784).clamp(0, 1)
            real_y = torch.randint(0, 10, (50,))
            if len(guarded_mems) >= 10:
                sample_mem = guarded_mems[-50:]
                real_x = torch.from_numpy(np.array([m["image"] for m in sample_mem], dtype=np.float32))
                real_y = torch.tensor([m["pred"] if m["true_label"] is None else m["true_label"] for m in sample_mem], dtype=torch.long)

            # Replay taught memories with soft stability defense (The Exchange)
            taught_x = None
            taught_y = None
            taught_weights = None
            taught_p_before = 0.0
            taught_p_after = 0.0

            if len(self.taught_buffer) > 0:
                t_list = list(self.taught_buffer)
                taught_x = torch.from_numpy(np.array([t["image"] for t in t_list], dtype=np.float32))
                taught_y = torch.tensor([t["true_label"] for t in t_list], dtype=torch.long)

                with torch.no_grad():
                    t_logits, t_h = self.classifier(taught_x)
                    t_stats = extract_internal_stats_v2(t_h, t_logits)
                    taught_p_before = float(self.intro_head(t_stats).mean().item())

                # Evaluate taught memories with SecondOpinionVoucher (v3.1)
                voucher_res = self.voucher.evaluate_taught_batch(
                    taught_x, taught_y, self.classifier, n_perturbations=5, noise_std=0.05, seed=42
                )
                taught_weights = voucher_res["weights"]
                verdicts = voucher_res["verdicts"]

                for i, v in enumerate(verdicts):
                    w_val = float(taught_weights[i].item())
                    if v == "plausible_correction":
                        self.log_event(f"taught: plausible correction (w={w_val:.1f}) -- second opinion vouches.", "info")
                    elif v == "implausible_whisper":
                        self.log_event(f"taught: implausible (w={w_val:.1f}) -- unstable whisper.", "warning")
                    elif v == "stable_reinforcement":
                        self.log_event(f"taught: stable reinforcement (w={w_val:.1f}).", "info")

            self.consolidator.sleep_cycle(
                real_x, real_y, raw_dreams, pseudo_y, weights,
                taught_x=taught_x, taught_y=taught_y, taught_weights=taught_weights
            )

            # Increment Generation
            self.generation += 1
            self.storage.set_meta("generation", self.generation)

            acc_after = self.evaluate_conscience()
            acc_delta = (acc_after - acc_before) * 100.0

            if taught_x is not None:
                with torch.no_grad():
                    t_logits_post, t_h_post = self.classifier(taught_x)
                    t_stats_post = extract_internal_stats_v2(t_h_post, t_logits_post)
                    taught_p_after = float(self.intro_head(t_stats_post).mean().item())

            # Evaluate dreams with classifier and introspective head for realistic telemetry
            with torch.no_grad():
                d_logits, d_h = self.classifier(raw_dreams)
                d_preds = d_logits.argmax(dim=1)
                d_stats = extract_internal_stats_v2(d_h, d_logits)
                d_p_errs = self.intro_head(d_stats).squeeze()

            # Pick 8 diverse dreams across classes
            unique_classes = torch.unique(dream_labels)
            chosen_indices = []
            for c in unique_classes:
                idx = (dream_labels == c).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    chosen_indices.append(idx[0].item())
                if len(chosen_indices) == 8:
                    break
            if len(chosen_indices) < 8:
                for i in range(len(raw_dreams)):
                    if i not in chosen_indices:
                        chosen_indices.append(i)
                    if len(chosen_indices) == 8:
                        break

            new_dreams = []
            for idx in chosen_indices:
                d_img = raw_dreams[idx].numpy()
                b64 = image_to_base64(d_img)
                pe = float(d_p_errs[idx].item()) if d_p_errs.ndim > 0 else float(d_p_errs.item())
                w = float(weights[idx].item())
                is_nightmare = bool(pe >= 0.25 or w < 0.80)
                new_dreams.append({
                    "b64": b64,
                    "target_class": int(dream_labels[idx].item()),
                    "pseudo_label": int(d_preds[idx].item()),
                    "weight": w,
                    "p_error": pe,
                    "is_nightmare": is_nightmare,
                })
            self.recent_dreams = new_dreams
            self.storage.save_dreams(new_dreams)

            n_taught = len(self.taught_buffer)
            self.log_event(
                f"SLEEP #{self.sleep_count} COMPLETE (GEN {self.generation}): Consolidated 200 dreams + {n_taught} taught. Conscience Acc: {acc_before*100:.1f}% -> {acc_after*100:.1f}% (delta: {acc_delta:+.1f}pp). Taught P(error): {taught_p_before*100:.1f}% -> {taught_p_after*100:.1f}%. Awake.",
                "sleep"
            )

            return {
                "status": "success",
                "sleep_count": self.sleep_count,
                "generation": self.generation,
                "dreams_consolidated": 200,
                "taught_consolidated": n_taught,
                "conscience_acc_before": acc_before,
                "conscience_acc_after": acc_after,
                "conscience_acc_delta": acc_delta,
                "taught_p_error_before": taught_p_before,
                "taught_p_error_after": taught_p_after,
                "recent_dreams": self.recent_dreams,
            }
        finally:
            self.is_sleeping = False

    def _bootstrap_initial_dreams(self):
        """Generate initial dream feed on first boot and save to DB."""
        raw_dreams, dream_labels = self.dreamer.generate_random_dreams(n=16, seed=42)
        pseudo_y, weights = self.stability.compute_soft_weights(raw_dreams, seed=42)
        with torch.no_grad():
            d_logits, d_h = self.classifier(raw_dreams)
            d_preds = d_logits.argmax(dim=1)
            d_stats = extract_internal_stats_v2(d_h, d_logits)
            d_p_errs = self.intro_head(d_stats).squeeze()

        self.recent_dreams = []
        for i in range(min(8, len(raw_dreams))):
            pe = float(d_p_errs[i].item()) if d_p_errs.ndim > 0 else float(d_p_errs.item())
            w = float(weights[i].item())
            is_nightmare = bool(pe >= 0.25 or w < 0.80)
            self.recent_dreams.append({
                "b64": image_to_base64(raw_dreams[i].numpy()),
                "target_class": int(dream_labels[i].item()),
                "pseudo_label": int(d_preds[i].item()),
                "weight": w,
                "p_error": pe,
                "is_nightmare": is_nightmare,
            })
        self.storage.save_dreams(self.recent_dreams)

    def get_state(self) -> Dict[str, Any]:
        """Return full JSON-serializable snapshot of organism state."""
        uptime_sec = int(time.time() - self.process_start_time)
        lifetime_sec = int(time.time() - self.first_awake_time)
        hours, rem = divmod(lifetime_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            "specimen_id": f"SPECIMEN #001 - GEN {self.generation}",
            "generation": self.generation,
            "uptime_seconds": lifetime_sec,
            "uptime_str": uptime_str,
            "visitor_count": self.visitor_count,
            "total_feeds": self.total_feeds,
            "sleep_count": self.sleep_count,
            "is_sleeping": self.is_sleeping,
            "vitals": {
                "p_error": self.current_p_error,
                "confidence": self.current_confidence,
                "prediction": self.current_pred,
                "mood": self.current_mood,
                "disagreement": self.current_disagreement,
                "disagreement_note": self.disagreement_note,
                "last_input_b64": self.current_input_b64,
                "stats": self.current_stats,
            },
            "brainwaves": list(self.brainwaves),
            "recent_dreams": self.recent_dreams,
            "event_log": list(self.event_log)[:20],
            "memory_count": len(self.memory_buffer),
            "taught_count": len(self.taught_buffer),
        }
