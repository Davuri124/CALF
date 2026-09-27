# ============================================================
#  CALF — CHECKPOINT UTILITY (paste at top of every notebook)
#  Works with Google Colab + Google Drive
#  Guarantees: nothing is ever re-run from scratch
# ============================================================

import os, pickle, json, time
import torch
from google.colab import drive

# ── 1. Mount Drive (run once per session) ──────────────────
def mount_drive():
    drive.mount('/content/drive', force_remount=False)
    print("✅ Drive mounted")

# ── 2. Define all save paths ───────────────────────────────
BASE = "/content/drive/MyDrive/CALF"

PATHS = {
    "data"          : f"{BASE}/data/processed_dataset.pkl",
    "dag"           : f"{BASE}/graphs/causal_dag.pkl",
    "dag_fig"       : f"{BASE}/graphs/dag_figure.png",
    "t_x_learner"   : f"{BASE}/baselines/tx_learner.pkl",
    "bart"          : f"{BASE}/baselines/bart.pkl",
    "causal_forest" : f"{BASE}/baselines/causal_forest.pkl",
    "dragonnet"     : f"{BASE}/baselines/dragonnet.pth",
    "calf_best"     : f"{BASE}/calf_model/epoch_best.pth",
    "calf_epoch"    : f"{BASE}/calf_model/epoch_{{n}}.pth",  # use .format(n=N)
    "metrics"       : f"{BASE}/results/all_metrics.csv",
    "ablation"      : f"{BASE}/results/ablation_results.csv",
    "figures"       : f"{BASE}/figures/",
}

def make_dirs():
    """Create all CALF folders on Drive."""
    for folder in ["data", "graphs", "baselines", "calf_model", "results", "figures"]:
        os.makedirs(f"{BASE}/{folder}", exist_ok=True)
    print("✅ All CALF folders ready on Drive")

# ── 3. Generic save / load ─────────────────────────────────
def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"💾 Saved → {path}")

def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Not found: {path}  →  run the earlier phase first")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"✅ Loaded ← {path}")
    return obj

def exists(path):
    return os.path.exists(path)

# ── 4. PyTorch model checkpoint ────────────────────────────
def save_model_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "loss"        : loss,
        "timestamp"   : time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    print(f"💾 Model checkpoint → {path}  (epoch {epoch}, loss {loss:.4f})")

def load_model_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print(f"⚠️  No checkpoint at {path} — starting from scratch")
        return 0, float('inf')
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    timestamp_val = ckpt.get('timestamp')
    timestamp_info = f"  (saved {timestamp_val})" if timestamp_val is not None else ""
    print(f"✅ Resumed from epoch {ckpt['epoch']}{timestamp_info}")
    return ckpt["epoch"], ckpt["loss"]

# ── 5. Auto-find latest epoch checkpoint ──────────────────
def find_latest_epoch_checkpoint():
    """Scans /calf_model/ and returns path of the highest epoch saved."""
    folder = f"{BASE}/calf_model/"
    if not os.path.exists(folder):
        return None, 0
    files = [f for f in os.listdir(folder) if f.startswith("epoch_") and f.endswith(".pth")
             and f != "epoch_best.pth"]
    if not files:
        return None, 0
    epochs = [int(f.replace("epoch_","").replace(".pth","")) for f in files]
    latest = max(epochs)
    path = f"{folder}epoch_{latest}.pth"
    print(f"🔍 Found latest checkpoint: epoch_{latest}.pth")
    return path, latest

# ── 6. Training loop wrapper with auto-checkpointing ──────
def train_with_checkpoints(model, optimizer, train_fn, total_epochs=200,
                            save_every=10, resume=True):
    """
    Wraps your training loop with:
      - auto-resume from latest saved epoch
      - saves every `save_every` epochs
      - always saves best model separately
    
    train_fn(epoch) must return: loss (float)
    """
    start_epoch = 0
    best_loss = float('inf')

    if resume:
        ckpt_path, start_epoch = find_latest_epoch_checkpoint()
        if ckpt_path:
            start_epoch, best_loss = load_model_checkpoint(model, optimizer, ckpt_path)

    print(f"\n🚀 Training from epoch {start_epoch + 1} → {total_epochs}")
    print(f"   Saving every {save_every} epochs to Drive\n")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        loss = train_fn(epoch)

        # Save every N epochs
        if epoch % save_every == 0:
            ckpt_path = PATHS["calf_epoch"].format(n=epoch)
            save_model_checkpoint(model, optimizer, epoch, loss, ckpt_path)

        # Save best model separately
        if loss < best_loss:
            best_loss = loss
            save_model_checkpoint(model, optimizer, epoch, loss, PATHS["calf_best"])
            print(f"   ⭐ New best model at epoch {epoch}")

    print(f"\n✅ Training complete. Best loss: {best_loss:.4f}")
    print(f"   Best model at: {PATHS['calf_best']}")

# ── 7. Phase completion tracker ───────────────────────────
PHASE_LOG = f"{BASE}/phase_log.json"

def mark_phase_done(phase_name):
    log = {}
    if os.path.exists(PHASE_LOG):
        with open(PHASE_LOG) as f:
            log = json.load(f)
    log[phase_name] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PHASE_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"✅ Phase '{phase_name}' marked complete")

def check_phases_done():
    if not os.path.exists(PHASE_LOG):
        print("No phases completed yet.")
        return
    with open(PHASE_LOG) as f:
        log = json.load(f)
    print("\n📋 CALF Phase Completion Log:")
    for phase, ts in log.items():
        print(f"   ✅ {phase:25s} → {ts}")
    print()

# ── 8. Quick status check (run at start of every session) ──
def status():
    print("\n" + "="*55)
    print("  CALF PROJECT STATUS")
    print("="*55)
    checks = {
        "Processed dataset"  : PATHS["data"],
        "Causal DAG"         : PATHS["dag"],
        "T/X-Learner"        : PATHS["t_x_learner"],
        "BART"               : PATHS["bart"],
        "Causal Forest"      : PATHS["causal_forest"],
        "DragonNet"          : PATHS["dragonnet"],
        "CALF best model"    : PATHS["calf_best"],
        "Metrics CSV"        : PATHS["metrics"],
    }
    for name, path in checks.items():
        icon = "✅" if os.path.exists(path) else "⬜"
        print(f"  {icon}  {name}")
    print("="*55 + "\n")

# ── USAGE (paste at top of every new Colab session) ────────
# from calf_checkpoint_utils import *
# mount_drive()
# make_dirs()
# status()          ← shows exactly what's done and what's left
