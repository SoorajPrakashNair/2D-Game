# 2D Game: Train an AI Agent for Flappy Bird (PLE)

## ✅ Goal
Train a reinforcement learning (RL) agent to autonomously play a simple 2D game (Flappy Bird or Chrome Dino). The agent should learn the game mechanics, pick optimal actions, and **maximize score**.  
This implementation uses **Flappy Bird** via **PyGame Learning Environment (PLE)** and a **Double DQN** agent in TensorFlow/Keras.

---

## 🎮 Why Flappy Bird?
**Flappy Bird** is ideal for this coursework:
- **Simple, deterministic mechanics** (flap or don’t flap).
- **Fast episodes** → more learning signal.
- **Rich reward feedback** (survival time & pipe scores).
- **Low state dimensionality** → fast to train on CPU.
- **Well-supported environment** via PLE.

(Chrome Dino is also suitable, but Flappy Bird offers more immediate shaping signals and shorter iteration loops.)

---

## 🧠 Method: Double DQN + Experience Replay
We implement a **Double Deep Q-Network** with:
- **Online network** (Q): chooses actions.
- **Target network** (Q′): provides stable targets (periodically synced).
- **Experience Replay**: randomly samples past transitions for decorrelated training.
- **Epsilon-Greedy**: balances exploration vs exploitation (ε decays from 1.0 → 0.01).
- **Reward Shaping**: small survival reward to encourage staying alive.

### State Representation
From `game.getGameState()` (PLE), we use:
- `player_y`
- `player_vel`
- `next_pipe_dist_to_player`
- `next_pipe_top_y`

These 4 scalars form the input to the neural network.

### Action Space
`env.getActionSet()` returns 2 actions:
- **Do nothing** (coast)
- **Flap** (jump)

### Rewards
- **Native PLE reward** from the step
- **+ SURVIVAL_REWARD (0.1)** to encourage longer survival

---

## 🧩 Architecture
- MLP with two hidden layers (128 ReLU units each), output size = number of actions
- Loss: MSE over Q-targets
- Optimizer: Adam, `LR = 5e-4`

---

## ⚙️ Hyperparameters (from the code)
```
STATE_KEYS = ["player_y", "player_vel", "next_pipe_dist_to_player", "next_pipe_top_y"]
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000         # steps
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 200000
MAX_EPISODE_STEPS = 10000
SURVIVAL_REWARD = 0.1
```

---

## 🗺️ Action Plan & Rationale
1. **Pick Environment** → PLE Flappy Bird for simplicity and speed.
2. **Design State & Actions** → minimal features to reduce overfitting & speed up learning.
3. **Choose Algorithm** → Double DQN for overestimation bias reduction vs vanilla DQN.
4. **Implement Replay Buffer** → stabilize learning with batched updates.
5. **Set Epsilon Schedule** → start random, slowly rely on learned Q-values.
6. **Add Reward Shaping** → survival reward to avoid local minima.
7. **Checkpoint Often** → avoid losing progress; support resume.
8. **Build UX** → In-game **menu**, **Human vs AI** modes, **countdown**, **on-screen score**, and **“Boom!!!”** overlay on death.
9. **Test/Iterate** → validate that AI can clear pipes and improve average score.
10. **Document & Package** → this README with full commands.

---

## 📦 Project Files
- `qn_flappy.py` — **single merged script** with:
  - Training (`--train`)
  - AI test/play (`--test`)
  - Human play (`--human`)
  - In-game **menu & buttons**, **countdown**, **score overlay**, **Boom!!!** effect
- `model_flappy.keras` — saved model (native Keras format)
- `training_rewards.csv` — reward log (episode vs reward)
- `training_rewards.png` — training curve

> Note: We consolidated features from `flappy_test.py` into `qn_flappy.py`. You can delete `flappy_test.py` to avoid confusion.

---

## 🖥️ Setup
```bash
# (optional) create venv
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# install deps
pip install pygame ple tensorflow==2.* matplotlib numpy
# If ple fails via pip, use:
# pip install git+https://github.com/ntasfi/PyGame-Learning-Environment
```

> GPU is **not required**. If you see CUDA warnings, TensorFlow will fall back to CPU.

---

## ▶️ How to Run

### 1) Start the Menu UI (recommended)
```bash
python qn_flappy.py
```
- Use buttons: **Train Mode**, **AI Mode**, **Human Mode**, **Exit**.
- ESC during a round → returns to menu.

### 2) Direct Commands
- **Train:**
```bash
python qn_flappy.py --train --episodes 2500 --model_path model_flappy.keras
```
- **AI Mode (watch agent):**
```bash
python qn_flappy.py --test --model_path model_flappy.keras
```
- **Human Mode (you play):**
```bash
python qn_flappy.py --human
```

### Controls (Human Mode)
- **Spacebar** = flap
- **ESC** = back to menu

---

## 💾 Saving, Resuming & Formats
- The script **auto-saves** every 200 episodes and at the end:
  - `model_flappy.keras` (native Keras format — recommended)
- You can convert HDF5 → Keras format inside Python REPL (already done in your session):
```python
from tensorflow import keras
model = keras.models.load_model("model_flappy.h5", compile=False)
model.save("model_flappy.keras")
```

### Stopping Safely
- Press **Ctrl+C**; training loop will stop at a safe point soon.
- Because we save every 200 episodes, your progress is preserved.
- For immediate manual save, reduce the checkpoint interval in code if desired.

---

## 📈 Results Snapshot (from your logs)
The agent improved substantially over time. Example excerpts:

- Early training: `Avg(last100) ~ 4–6`
- Mid training: `Avg(last100) ~ 12–22`
- Peaks observed: rewards and scores like **91.6**, **98.4**, **103.5**, **169.8** on certain episodes.
- Later stabilization & variance as ε → 0.01 with occasional high scores.

These indicate the DQN is learning useful policies (timed flaps, safe gaps).

---

## 🧰 Troubleshooting
- **`NameError: FPS not defined`** → We define a constant `FPS = 30` in the merged script; ensure you’re running the latest `qn_flappy.py`.
- **CUDA warnings / `cuInit` errors** → Safe to ignore if you don’t have a GPU; TensorFlow runs on CPU.
- **`libpng warning: iCCP: known incorrect sRGB profile`** → benign Pygame image metadata warning; can be ignored.
- **Window doesn’t show buttons** → Make sure you run `python qn_flappy.py` without `--test/--human`, and that the script you’re running is the merged one.
- **Agent doesn’t improve** → Increase episodes, raise `MEMORY_SIZE`, or tune `LR`, `SURVIVAL_REWARD`, `TARGET_UPDATE_FREQ`.

---

## 🧪 Evaluation
- **Per-episode score**: reported on screen and printed after each episode in test mode.
- **Average over last 100 episodes**: printed during training for a stability signal.
- **Visual validation**: watch AI mode clear pipes reliably.

---

## 🛠️ Extensions (Future Work)
- **Prioritized Experience Replay** for better sampling of rare/valuable transitions.
- **Dueling DQN** architecture to better estimate state-value vs advantage.
- **Frame stacks / pixel input** to train end-to-end from images.
- **Curriculum**: gradually increase pipe difficulty or speed.
- **Model checkpoint best-by-score** and TensorBoard logging.
- **Self-play variants** aren’t relevant here, but **randomized seeds** improve robustness.

---

## 🧾 References (conceptual)
- Mnih et al., 2015 — Playing Atari with Deep RL (DQN)
- van Hasselt et al., 2016 — Double Q-learning for stability

---

## ✅ Summary
- Built a **Double DQN** agent that learns Flappy Bird in **PLE**.
- Includes a **clean UI**: menu buttons, **Human/AI** modes, countdown, on-screen score, “**Boom!!!**” overlay.
- Offers **safe saving**, resume, and a simple **CLI**.
- Demonstrated learning progress via logs and scores.

Good luck, and have fun breaking your high score! 🐥🚀
