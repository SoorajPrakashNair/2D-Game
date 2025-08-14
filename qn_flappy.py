# qn_flappy.py
# Flappy Bird (PLE) — Double DQN + On-screen UI (Menu, Countdown, Score HUD, Boom overlay)
# One file: Train / AI Play / Human Play / Exit

import os
import time
import csv
import random
from collections import deque
import argparse

import numpy as np
import pygame
import matplotlib.pyplot as plt

# TF/Keras kept minimal in import path to reduce verbose logs
from tensorflow.keras import layers, models, optimizers
from tensorflow import keras

from ple import PLE
from ple.games.flappybird import FlappyBird

# ---------------------------
# UI / Game constants
# ---------------------------
FPS = 30
WINDOW_W, WINDOW_H = 512, 512  # PLE will create its own window; we draw overlays on it
FONT_NAME = "Arial"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY  = (40, 40, 40)
RED   = (220, 60, 60)
YELLOW= (240, 200, 40)
CYAN  = (90, 220, 220)

# ---------------------------
# RL Hyperparameters
# ---------------------------
STATE_KEYS = ["player_y", "player_vel", "next_pipe_dist_to_player", "next_pipe_top_y"]
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_FREQ = 1_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 200_000
MAX_EPISODE_STEPS = 10_000
SURVIVAL_REWARD = 0.1

# ---------------------------
# Small utils
# ---------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, float(done)))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d
    def __len__(self): return len(self.buffer)

def build_model(input_size, n_actions):
    m = models.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_actions, activation="linear"),
    ])
    m.compile(optimizer=optimizers.Adam(learning_rate=LR), loss="mse")
    return m

def state_to_array(state_dict):
    return np.array([float(state_dict[k]) for k in STATE_KEYS], dtype=np.float32)

def save_rewards_to_csv(filename, rewards):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Episode", "Reward"])
        for i, r in enumerate(rewards, 1):
            w.writerow([i, r])

# ---------------------------
# UI helpers (pygame overlays)
# ---------------------------
def get_display_surface():
    # After PLE env.init(), pygame has an active display
    return pygame.display.get_surface()

def draw_text(surface, text, size, color, center):
    font = pygame.font.SysFont(FONT_NAME, size, bold=True)
    label = font.render(text, True, color)
    rect = label.get_rect(center=center)
    surface.blit(label, rect)

def draw_hud(score, mode):
    surf = get_display_surface()
    if surf is None: return
    # top HUD bar
    pygame.draw.rect(surf, (0, 0, 0), (0, 0, surf.get_width(), 40))
    draw_text(surf, f"Score: {int(score)}", 28, YELLOW, (surf.get_width() // 2, 20))
    draw_text(surf, f"{mode}", 18, CYAN, (70, 20))
    pygame.display.update(pygame.Rect(0, 0, surf.get_width(), 40))

def countdown_overlay():
    surf = get_display_surface()
    if surf is None: return
    for num in ["3", "2", "1", "Go!"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        surf.fill((0, 0, 0))
        draw_text(surf, num, 72, WHITE, (surf.get_width() // 2, surf.get_height() // 2))
        pygame.display.flip()
        time.sleep(1.0)

def boom_overlay():
    surf = get_display_surface()
    if surf is None: return
    # darken screen
    overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surf.blit(overlay, (0, 0))
    draw_text(surf, "Boom!!!", 72, RED, (surf.get_width() // 2, surf.get_height() // 2))
    pygame.display.flip()
    time.sleep(2.0)

# ---------------------------
# Menus (pygame)
# ---------------------------
class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
    def draw(self, surf, hover):
        color = (70, 70, 70) if not hover else (100, 100, 100)
        pygame.draw.rect(surf, color, self.rect, border_radius=12)
        pygame.draw.rect(surf, (140, 140, 140), self.rect, 2, border_radius=12)
        draw_text(surf, self.label, 28, WHITE, self.rect.center)
    def is_hover(self, pos): return self.rect.collidepoint(pos)

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("FlappyBird — DQN Menu")

    title_font = pygame.font.SysFont(FONT_NAME, 36, bold=True)

    buttons = [
        Button((WINDOW_W//2 - 140, 160, 280, 56), "Train"),
        Button((WINDOW_W//2 - 140, 230, 280, 56), "AI Play"),
        Button((WINDOW_W//2 - 140, 300, 280, 56), "Human Play"),
        Button((WINDOW_W//2 - 140, 370, 280, 56), "Exit"),
    ]

    clock = pygame.time.Clock()
    while True:
        mouse = pygame.mouse.get_pos()
        screen.fill(GREY)

        title = title_font.render("Flappy Bird — Double DQN", True, WHITE)
        screen.blit(title, (WINDOW_W//2 - title.get_width()//2, 80))

        clicked_idx = None
        for i, b in enumerate(buttons):
            b.draw(screen, b.is_hover(mouse))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, b in enumerate(buttons):
                    if b.is_hover(mouse):
                        clicked_idx = i
        if clicked_idx is not None:
            return clicked_idx  # 0=train, 1=AI, 2=human, 3=exit

        clock.tick(60)

# ---------------------------
# Training
# ---------------------------
def train(episodes=2000, model_path="model_flappy.keras"):
    # Headless training (render=False)
    game = FlappyBird()
    env = PLE(game, fps=FPS, display_screen=False)
    env.init()
    action_set = env.getActionSet()
    n_actions = len(action_set)

    dummy_state = state_to_array(game.getGameState())
    input_size = dummy_state.shape[0]

    q_net = build_model(input_size, n_actions)
    target_net = build_model(input_size, n_actions)

    # resume
    if os.path.exists(model_path):
        try:
            q_net.load_weights(model_path)
            print(f"Loaded existing weights from {model_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    target_net.set_weights(q_net.get_weights())

    replay = ReplayBuffer(MEMORY_SIZE)

    # warmup
    while len(replay) < MIN_REPLAY_SIZE:
        env.reset_game()
        s = state_to_array(game.getGameState())
        steps = 0
        while not env.game_over() and len(replay) < MIN_REPLAY_SIZE and steps < MAX_EPISODE_STEPS:
            a_idx = random.randrange(n_actions)
            r = env.act(action_set[a_idx]) + SURVIVAL_REWARD
            ns = state_to_array(game.getGameState())
            replay.add(s, a_idx, r, ns, env.game_over())
            s = ns
            steps += 1

    total_steps = 0
    eps = EPS_START
    eps_decay = (EPS_START - EPS_END) / EPS_DECAY_STEPS
    rewards_history = []

    for ep in range(1, episodes + 1):
        env.reset_game()
        s = state_to_array(game.getGameState())
        ep_reward = 0.0
        step = 0
        done = False

        while not done and step < MAX_EPISODE_STEPS:
            if random.random() < eps:
                a_idx = random.randrange(n_actions)
            else:
                q_vals = q_net.predict(s.reshape(1, -1), verbose=0)[0]
                a_idx = int(np.argmax(q_vals))

            r = env.act(action_set[a_idx]) + SURVIVAL_REWARD
            ns = state_to_array(game.getGameState())
            done = env.game_over()

            replay.add(s, a_idx, r, ns, done)
            s = ns
            ep_reward += r
            step += 1
            total_steps += 1

            if eps > EPS_END:
                eps = max(EPS_END, eps - eps_decay)

            if len(replay) >= BATCH_SIZE:
                s_b, a_b, r_b, ns_b, d_b = replay.sample(BATCH_SIZE)
                q_next_policy = q_net.predict(ns_b, verbose=0)
                next_a = np.argmax(q_next_policy, axis=1)
                q_next_target = target_net.predict(ns_b, verbose=0)
                q_target = q_net.predict(s_b, verbose=0)

                for i in range(BATCH_SIZE):
                    target_val = r_b[i] if d_b[i] == 1.0 else r_b[i] + GAMMA * q_next_target[i, next_a[i]]
                    q_target[i, int(a_b[i])] = target_val
                q_net.train_on_batch(s_b, q_target)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.set_weights(q_net.get_weights())

        rewards_history.append(ep_reward)
        if ep % 10 == 0:
            avg100 = np.mean(rewards_history[-100:])
            print(f"Episode {ep} | Reward: {ep_reward:.2f} | Avg(last100): {avg100:.2f} | Eps: {eps:.3f}")
        if ep % 200 == 0:
            q_net.save(model_path)
            print(f"Saved checkpoint: {model_path}")

    q_net.save(model_path)
    print(f"Training finished. Model saved to {model_path}")

    save_rewards_to_csv("training_rewards.csv", rewards_history)
    plt.figure()
    plt.plot(rewards_history, label="Episode Reward")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig("training_rewards.png")

# ---------------------------
# Model load helper (handles .keras or .h5)
# ---------------------------
def load_trained_model(model_path, input_size, n_actions):
    model = build_model(input_size, n_actions)
    # Try native Keras format
    try:
        loaded = keras.models.load_model(model_path, compile=False)
        # ensure same architecture output size
        if loaded.output_shape[-1] == n_actions and loaded.input_shape[-1] == input_size:
            return loaded
        else:
            # fallback to weights if shapes mismatch
            model.load_weights(model_path)
            return model
    except Exception:
        # Fallback: treat file as weights only
        model.load_weights(model_path)
        return model

# ---------------------------
# Play loops (AI / Human) with overlays
# ---------------------------
def play_ai(model_path):
    game = FlappyBird()
    env = PLE(game, fps=FPS, display_screen=True)
    env.init()
    action_set = env.getActionSet()
    n_actions = len(action_set)
    input_size = len(STATE_KEYS)

    # prepare pygame font (now that display exists)
    pygame.font.init()

    # countdown
    countdown_overlay()

    model = load_trained_model(model_path, input_size, n_actions)

    # Single-episode play, then return to menu
    env.reset_game()
    clock = pygame.time.Clock()
    while not env.game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

        state = state_to_array(game.getGameState()).reshape(1, -1)
        q_vals = model.predict(state, verbose=0)[0]
        action_idx = int(np.argmax(q_vals))
        env.act(action_set[action_idx])

        # HUD
        draw_hud(game.getScore(), "AI")

        clock.tick(FPS)

    boom_overlay()

def play_human():
    game = FlappyBird()
    env = PLE(game, fps=FPS, display_screen=True)
    env.init()
    action_set = env.getActionSet()

    pygame.font.init()
    countdown_overlay()

    env.reset_game()
    clock = pygame.time.Clock()
    while not env.game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

        keys = pygame.key.get_pressed()
        # flap on SPACE, otherwise do nothing
        action = action_set[1] if len(action_set) > 1 and keys[pygame.K_SPACE] else action_set[0]
        env.act(action)

        draw_hud(game.getScore(), "HUMAN")
        clock.tick(FPS)

    boom_overlay()

# ---------------------------
# App entry — menu-first experience
# ---------------------------
def run_menu(model_path, train_episodes):
    while True:
        choice = main_menu()  # 0=train, 1=AI, 2=human, 3=exit
        if choice == 0:
            # Close the menu window before training to free the display
            pygame.display.quit()
            train(episodes=train_episodes, model_path=model_path)
            # Re-open pygame window on return to menu
            # (loop will continue and recreate menu)
        elif choice == 1:
            play_ai(model_path)
        elif choice == 2:
            play_human()
        elif choice == 3:
            pygame.quit()
            return

# ---------------------------
# CLI (optional): default is GUI menu
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flappy Bird Double DQN with in-game UI")
    parser.add_argument("--menu", action="store_true", help="Start GUI menu (default if no args)")
    parser.add_argument("--train", action="store_true", help="Headless training (no GUI)")
    parser.add_argument("--test", action="store_true", help="AI play directly (no menu)")
    parser.add_argument("--human", action="store_true", help="Human play directly (no menu)")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    parser.add_argument("--model_path", type=str, default="model_flappy.keras", help="Model path (.keras or .h5)")
    args = parser.parse_args()

    # Default to menu if no flags
    if not any([args.train, args.test, args.human, args.menu]):
        args.menu = True

    if args.train:
        train(episodes=args.episodes, model_path=args.model_path)
    elif args.test:
        pygame.init()
        # Create a small window so overlays can render; PLE will replace it
        pygame.display.set_mode((WINDOW_W, WINDOW_H))
        play_ai(args.model_path)
        pygame.quit()
    elif args.human:
        pygame.init()
        pygame.display.set_mode((WINDOW_W, WINDOW_H))
        play_human()
        pygame.quit()
    else:
        run_menu(args.model_path, args.episodes)
