"""Cost-accuracy plotting. See CLAUDE.md step S5+ (the 'money plot')."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cost_accuracy(points, out_path, title="Cost vs accuracy"):
    """points: list of dicts with 'policy', 'mean_cost', 'accuracy'. One series per method.

    'fixed@k' points are merged into a single 'fixed budget' line."""
    # Group: all fixed@k under one key, otherwise by policy string.
    groups = {}
    for p in points:
        key = "fixed budget" if p["policy"].startswith("fixed") else p["policy"]
        groups.setdefault(key, []).append(p)

    styles = {"fixed budget": "o-", "esc": "P", "adaptive-confidence": "s--",
              "adaptive-agreement": "^:"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, pts in groups.items():
        pts = sorted(pts, key=lambda p: p["mean_cost"])
        ax.plot([p["mean_cost"] for p in pts], [p["accuracy"] for p in pts],
                styles.get(key, "x-"), label=key, markersize=8 if key == "esc" else 6)
    ax.set_xlabel("mean samples / problem (compute)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
