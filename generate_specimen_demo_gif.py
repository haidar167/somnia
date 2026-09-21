"""Generate a multi-frame scripted animated demo GIF for THE SPECIMEN web console."""

import time
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from somnia.utils import project_root, set_seed
from somnia.data import load_mnist
from specimen.organism import SpecimenOrganism


def render_console_frame(organism, input_img, title_step, status_text, disagreement=False, note=""):
    fig, (ax_in, ax_vit, ax_log) = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#090C10")
    for ax in [ax_in, ax_vit, ax_log]:
        ax.set_facecolor("#11161F")

    # Left: Input display
    if input_img is not None:
        ax_in.imshow(input_img, cmap="viridis")
        ax_in.set_title(f"SENSORY INPUT\n({title_step})", color="#E2E8F0", fontsize=10, fontweight="bold")
    else:
        ax_in.text(0.5, 0.5, "IDLE / WAITING", color="#64748B", fontsize=12, ha="center")
        ax_in.set_title("SENSORY INPUT", color="#E2E8F0", fontsize=10, fontweight="bold")
    ax_in.axis("off")

    # Center: Vitals & Disagreement
    v = organism.get_state()["vitals"]
    p_err = v.get("p_error", 0.03)
    conf = v.get("confidence", 0.0)
    pred = v.get("prediction", "--")
    mood = v.get("mood", "calm").upper()

    mood_color = "#00FF9D" if mood == "CALM" else ("#FACC15" if mood == "UNEASY" else "#FF3B30")

    ax_vit.text(0.5, 0.85, f"STATUS: {status_text}", color="#38BDF8", fontsize=11, fontweight="bold", ha="center")
    ax_vit.text(0.5, 0.65, f"MOUTH: {pred} ({conf*100:.1f}% CONF)", color="#E2E8F0", fontsize=12, ha="center")
    ax_vit.text(0.5, 0.45, f"GUT: P(error) = {p_err*100:.1f}% [{mood}]", color=mood_color, fontsize=13, fontweight="bold", ha="center")

    if disagreement:
        ax_vit.text(0.5, 0.20, "⚠️ NEURAL DISAGREEMENT DETECTED", color="#FF3B30", fontsize=10, fontweight="bold", ha="center")
        ax_vit.text(0.5, 0.08, note[:40], color="#F87171", fontsize=8, ha="center")
    else:
        ax_vit.text(0.5, 0.15, "CIRCUITRY SYNCHRONIZED", color="#10B981", fontsize=9, ha="center")
    ax_vit.axis("off")

    # Right: Subconscious / Log
    ax_log.set_title("SUBCONSCIOUS & LOGS", color="#E2E8F0", fontsize=10, fontweight="bold")
    logs = organism.get_state().get("event_log", [])[:5]
    y_pos = 0.8
    for entry in logs:
        msg = entry.get("message", "")[:35]
        t = entry.get("time", "")
        ax_log.text(0.05, y_pos, f"[{t}] {msg}", color="#94A3B8", fontsize=8)
        y_pos -= 0.16
    ax_log.axis("off")

    fig.suptitle("SPECIMEN #001 -- LIVING AI CONSOLE TELEMETRY STREAM",
                 color="#00FF9D", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame_np = np.asarray(buf)
    plt.close(fig)
    return Image.fromarray(frame_np)


def generate_specimen_demo():
    root = project_root()
    set_seed(42)
    organism = SpecimenOrganism()
    _, _, test_x, test_y = load_mnist()

    frames = []

    # 1. Idle Boot frames (3 frames)
    for _ in range(3):
        frames.append(render_console_frame(organism, None, "STANDBY", "ALIVE (ONLINE)"))

    # 2. Feed Normal 7 (4 frames)
    clean_7 = test_x[0].reshape(28, 28)
    res_clean = organism.feed(clean_7)
    for _ in range(4):
        frames.append(render_console_frame(organism, clean_7, "CLEAN '7'", "EVALUATING INPUT"))

    # 3. Feed Noise Spike (4 frames)
    rng = np.random.RandomState(42)
    noise_img = rng.randn(28, 28).clip(0, 1).astype(np.float32)
    res_noise = organism.feed(noise_img)
    for _ in range(4):
        frames.append(render_console_frame(organism, noise_img, "WHITE NOISE", "ALARM SPIKED!"))

    # 4. Disagreement Moment (5 frames)
    ambiguous = rng.uniform(0.1, 0.6, size=(28, 28)).astype(np.float32)
    ambiguous[7:21, 12:16] = 0.9
    res_dis = organism.feed(ambiguous)
    for _ in range(5):
        frames.append(render_console_frame(organism, ambiguous, "AMBIGUOUS DIGIT", "DISAGREEMENT!",
                                           disagreement=True, note=res_dis["disagreement_note"]))

    # 5. Sleep Consolidation (4 frames)
    sleep_res = organism.sleep_cycle()
    for _ in range(4):
        frames.append(render_console_frame(organism, None, "DREAM REPLAY", "SLEEP CONSOLIDATION"))

    save_path = str(root / "figures" / "specimen_demo.gif")
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"  Specimen demo GIF generated -> {save_path} ({len(frames)} frames)")


if __name__ == "__main__":
    generate_specimen_demo()
