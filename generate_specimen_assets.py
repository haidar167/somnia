"""Generate high-resolution visual cards / screenshots for THE SPECIMEN marketing & README."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from somnia.utils import project_root, set_seed
from specimen.organism import SpecimenOrganism


def generate_specimen_mockup_figures():
    root = project_root()
    organism = SpecimenOrganism()

    # 1. Generate Disagreement Moment Figure
    # Feed an ambiguous noise/digit to induce disagreement
    rng = np.random.RandomState(42)
    ambiguous = rng.uniform(0.1, 0.6, size=(28, 28)).astype(np.float32)
    ambiguous[7:21, 12:16] = 0.9  # faint stroke
    feed_res = organism.feed(ambiguous)

    # Render a high-res graphic representation of the Disagreement Moment
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#090C10")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#11161F")

    # Left: The Input
    ax1.imshow(ambiguous, cmap="magma")
    ax1.set_title("SENSORY INPUT (AMBIGUOUS '1/7')", color="#E2E8F0", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Middle: Mouth Channel (Softmax)
    pred = feed_res["prediction"]
    conf = feed_res["confidence"]
    ax2.text(0.5, 0.7, f"PREDICTED: {pred}", color="#38BDF8", fontsize=18, fontweight="bold", ha="center")
    ax2.text(0.5, 0.45, f"CONFIDENCE: {conf*100:.1f}%", color="#E2E8F0", fontsize=14, ha="center")
    ax2.text(0.5, 0.2, "CHANNEL: SOFTMAX MOUTH", color="#94A3B8", fontsize=10, ha="center")
    ax2.set_title("MOUTH CHANNEL", color="#38BDF8", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Right: Gut Channel (Introspection)
    p_err = feed_res["p_error"]
    mood = feed_res["mood"].upper()
    ax3.text(0.5, 0.7, f"P(ERROR): {p_err*100:.1f}%", color="#FF3B30", fontsize=18, fontweight="bold", ha="center")
    ax3.text(0.5, 0.45, f"MOOD: {mood}", color="#FB923C", fontsize=14, fontweight="bold", ha="center")
    ax3.text(0.5, 0.2, "CHANNEL: 10-STAT SOMATOSENSORY", color="#94A3B8", fontsize=10, ha="center")
    ax3.set_title("GUT FEELING (INTROSPECTION)", color="#FF3B30", fontsize=11, fontweight="bold")
    ax3.axis("off")

    fig.suptitle("THE DISAGREEMENT MOMENT: When The Mouth and Gut Disagree",
                 color="#FF3B30", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    disagree_path = str(root / "figures" / "specimen_disagreement.png")
    plt.savefig(disagree_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Disagreement figure saved -> {disagree_path}")


if __name__ == "__main__":
    generate_specimen_mockup_figures()
