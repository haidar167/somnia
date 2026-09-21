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
from somnia.utils import project_root, set_seed


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert a 28x28 float array in [0, 1] to base64 PNG data URL."""
    img_uint8 = (np.clip(img_array.reshape(28, 28), 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class SpecimenOrganism:
    """A living AI organism with real-time somatosensory awareness and dream consolidation."""

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            models_dir = project_root() / "data"
        self.models_dir = Path(models_dir)

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
        self.consolidator = SleepConsolidationV2(self.classifier, dream_mix=0.05, lr=1e-4, epochs=2)

        # Organism Vitality State
        self.start_time = time.time()
        self.visitor_count = 0
        self.sleep_count = 0
        self.total_feeds = 0
        self.is_sleeping = False

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
        self.event_log: collections.deque = collections.deque(maxlen=200)
        self.recent_dreams: List[Dict[str, Any]] = []

        # Bootstrap initial state & dreams
        self._bootstrap_initial_dreams()
        self.log_event("Organism awakened. Neural circuitry calibrated.", "system")

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
        """Record timestamped event to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.appendleft({
            "time": timestamp,
            "message": message,
            "type": event_type,
        })

    def feed(self, img_28x28: np.ndarray) -> Dict[str, Any]:
        """Feed a 28x28 image into the organism, classifying and sensing internal confusion."""
        x_flat = np.clip(img_28x28.reshape(-1, 784), 0.0, 1.0).astype(np.float32)
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
        if conf >= 0.75 and p_err >= 0.25:
            disagreement = True
            disagreement_note = f"The mouth says {pred} ({conf*100:.1f}% sure), but the gut feels danger (P(error) = {p_err*100:.1f}%)!"
        elif conf < 0.50 and p_err < 0.10:
            disagreement = True
            disagreement_note = f"The mouth hesitates ({conf*100:.1f}% confidence), but internal activations are completely calm."

        b64_thumb = image_to_base64(img_28x28)

        # Update Live State
        self.total_feeds += 1
        self.current_input_b64 = b64_thumb
        self.current_pred = pred
        self.current_confidence = conf
        self.current_p_error = p_err
        self.current_stats = stats_dict
        self.current_mood = mood
        self.current_disagreement = disagreement
        self.disagreement_note = disagreement_note

        # Store in memory buffer
        memory_item = {
            "id": self.total_feeds,
            "image": x_flat[0],
            "b64": b64_thumb,
            "pred": pred,
            "conf": conf,
            "p_error": p_err,
            "true_label": None,
            "timestamp": time.time(),
        }
        self.memory_buffer.append(memory_item)

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
        }

    def reveal(self, true_label: int) -> Dict[str, Any]:
        """Reveal ground truth label for the last fed sample."""
        if len(self.memory_buffer) == 0:
            return {"status": "error", "message": "No input to reveal"}

        last_item = self.memory_buffer[-1]
        last_item["true_label"] = int(true_label)
        was_correct = (last_item["pred"] == int(true_label))

        if was_correct:
            self.log_event(
                f"REVEAL: Confirmed correct! Label is {true_label}. (I felt P(error)={last_item['p_error']*100:.1f}%).",
                "success"
            )
        else:
            self.log_event(
                f"REVEAL: I was WRONG. Said {last_item['pred']}, but true label is {true_label}. I felt P(error)={last_item['p_error']*100:.1f}% BEFORE I knew!",
                "error"
            )

        return {
            "predicted": last_item["pred"],
            "true_label": true_label,
            "was_correct": was_correct,
            "p_error_before": last_item["p_error"],
        }

    def sleep_cycle(self) -> Dict[str, Any]:
        """Execute one targeted dream self-healing consolidation cycle."""
        if self.is_sleeping:
            return {"status": "already_sleeping"}

        self.is_sleeping = True
        self.sleep_count += 1
        self.log_event(f"SLEEP #{self.sleep_count}: Entering dream consolidation...", "sleep")

        try:
            # Profile confusion from memory buffer or fallback
            if len(self.memory_buffer) >= 10:
                mem_x = torch.from_numpy(np.array([m["image"] for m in list(self.memory_buffer)[-200:]]))
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
            if len(self.memory_buffer) >= 10:
                sample_mem = list(self.memory_buffer)[-50:]
                real_x = torch.from_numpy(np.array([m["image"] for m in sample_mem]))
                real_y = torch.tensor([m["pred"] if m["true_label"] is None else m["true_label"] for m in sample_mem])

            self.consolidator.sleep_cycle(real_x, real_y, raw_dreams, pseudo_y, weights)

            # Store recent 8 dreams with confusion badges
            new_dreams = []
            for i in range(min(8, len(raw_dreams))):
                d_img = raw_dreams[i].numpy()
                b64 = image_to_base64(d_img)
                p_err = 1.0 - weights[i].item()
                new_dreams.append({
                    "b64": b64,
                    "target_class": int(dream_labels[i].item()),
                    "pseudo_label": int(pseudo_y[i].item()),
                    "weight": float(weights[i].item()),
                    "p_error": float(p_err),
                    "is_nightmare": bool(weights[i].item() < 0.80),
                })
            self.recent_dreams = new_dreams

            post_conf = float(np.mean([m["p_error"] for m in self.memory_buffer])) if len(self.memory_buffer) > 0 else 0.05
            self.log_event(
                f"SLEEP #{self.sleep_count} COMPLETE: Consolidated 200 targeted dreams. Mean memory confusion: {post_conf*100:.1f}%. Awake.",
                "sleep"
            )

            return {
                "status": "success",
                "sleep_count": self.sleep_count,
                "dreams_consolidated": 200,
                "recent_dreams": self.recent_dreams,
            }
        finally:
            self.is_sleeping = False

    def _bootstrap_initial_dreams(self):
        """Generate initial dream feed on boot."""
        raw_dreams, dream_labels = self.dreamer.generate_random_dreams(n=8, seed=42)
        pseudo_y, weights = self.stability.compute_soft_weights(raw_dreams, seed=42)
        self.recent_dreams = [
            {
                "b64": image_to_base64(raw_dreams[i].numpy()),
                "target_class": int(dream_labels[i].item()),
                "pseudo_label": int(pseudo_y[i].item()),
                "weight": float(weights[i].item()),
                "p_error": float(1.0 - weights[i].item()),
                "is_nightmare": bool(weights[i].item() < 0.80),
            }
            for i in range(8)
        ]

    def get_state(self) -> Dict[str, Any]:
        """Return full JSON-serializable snapshot of organism state."""
        uptime_sec = int(time.time() - self.start_time)
        hours, rem = divmod(uptime_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            "specimen_id": "SPECIMEN #001",
            "uptime_seconds": uptime_sec,
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
        }
