#!/usr/bin/env python3
"""
train.py — Isaac Lab Soccer Training (Isaac Lab 2.x)
=====================================================
Run from IsaacLab directory:
  .\isaaclab.bat -p C:\IsaacLab\soccer\train.py --num_envs 1024
  .\isaaclab.bat -p C:\IsaacLab\soccer\train.py --num_envs 1024 --play --checkpoint runs\soccer\<ts>\model_5000.pt
"""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TurtleBot3 Soccer — Isaac Lab")
parser.add_argument("--num_envs",   type=int,  default=1024)
parser.add_argument("--play",       action="store_true")
parser.add_argument("--checkpoint", type=str,  default=None)
parser.add_argument("--max_iters",  type=int,  default=5000)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soccer_env import SoccerEnv, SoccerEnvCfg


# ── PPO config (plain dict — works with all rsl_rl versions) ─────────────────
def make_runner_cfg(max_iters: int) -> dict:
    return {
        # ── Observation groups ────────────────────────────────────────
        # Must match keys returned by _get_observations() → {"policy": obs}
        "obs_groups": {
            "actor":  ["policy"],
            "critic": ["policy"],
        },

        # ── Actor — outputs stochastic actions via GaussianDistribution ─
        "actor": {
            "class_name":       "rsl_rl.models.MLPModel",
            "hidden_dims":      [256, 128, 64],
            "activation":       "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
            },
        },

        # ── Critic — outputs deterministic value estimate ─────────────
        "critic": {
            "class_name":       "rsl_rl.models.MLPModel",
            "hidden_dims":      [256, 128, 64],
            "activation":       "elu",
            "obs_normalization": False,
        },

        # ── PPO (keys match PPO.__init__ params exactly) ──────────────
        "algorithm": {
            "class_name":                      "rsl_rl.algorithms.PPO",
            "num_learning_epochs":             8,
            "num_mini_batches":                4,
            "clip_param":                      0.2,
            "gamma":                           0.99,
            "lam":                             0.95,
            "value_loss_coef":                 1.0,
            "entropy_coef":                    0.05,
            "learning_rate":                   1e-4,
            "max_grad_norm":                   1.0,
            "optimizer":                       "adam",
            "use_clipped_value_loss":          True,
            "schedule":                        "adaptive",
            "desired_kl":                      0.01,
            "normalize_advantage_per_mini_batch": False,
        },

        # ── Multi-GPU — pass None to disable (matches multi_gpu_cfg=None default) 
        "multi_gpu": None,

        # ── Runner ────────────────────────────────────────────────────
        "num_steps_per_env": 24,
        "max_iterations":    max_iters,
        "save_interval":     100,
        "experiment_name":   "soccer",
        "run_name":          "",
        "logger":            "tensorboard",
        "log_interval":      10,
    }


# ── Environment factory ───────────────────────────────────────────────────────
def make_env(num_envs: int) -> RslRlVecEnvWrapper:
    cfg                   = SoccerEnvCfg()
    cfg.num_envs          = num_envs
    cfg.scene.num_envs    = num_envs
    cfg.scene.env_spacing = cfg.env_spacing
    env = SoccerEnv(cfg=cfg)
    return RslRlVecEnvWrapper(env)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    env     = make_env(args.num_envs)
    cfg     = make_runner_cfg(args.max_iters)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)

    if args.checkpoint:
        print(f"Resuming from {args.checkpoint}")
        runner.load(args.checkpoint)

    n_steps = cfg["num_steps_per_env"] * args.num_envs
    print(f"\n{'='*50}")
    print(f"  TurtleBot3 Soccer — Isaac Lab")
    print(f"{'='*50}")
    print(f"  Device:          {device}")
    print(f"  Envs:            {args.num_envs:,}")
    print(f"  Steps/rollout:   {n_steps:,}")
    print(f"  Max iterations:  {args.max_iters:,}")
    print(f"  TensorBoard:     tensorboard --logdir {log_dir}")
    print(f"{'='*50}\n")

    runner.learn(
        num_learning_iterations=args.max_iters,
        init_at_random_ep_len=True,
    )
    env.close()
    print("Training complete.")


# ── Play ──────────────────────────────────────────────────────────────────────
def play(args):
    if not args.checkpoint:
        print("ERROR: --play requires --checkpoint <path>")
        sys.exit(1)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    env     = make_env(num_envs=1)
    cfg     = make_runner_cfg(max_iters=1)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    print(f"Loaded: {args.checkpoint}\n")
    print(f"{'Episode':>8}  {'Outcome':<22}  {'Reward':>10}")
    print("-" * 45)

    obs, _       = env.reset()
    episode      = 0
    goals        = 0
    total_reward = 0.0

    while simulation_app.is_running():
        with torch.no_grad():
            action = policy(obs)
        obs, reward, done, extras = env.step(action)
        total_reward += reward.sum().item()

        if done.any():
            episode += 1
            timed_out = extras.get("time_outs", torch.zeros_like(done))
            scored    = (done & ~timed_out).any().item()
            if scored:
                goals += 1
            outcome = "GOAL" if scored else "timeout/oob"
            pct     = 100 * goals / max(episode, 1)
            print(f"{episode:>8}  {outcome:<22}  {total_reward:>10.1f}"
                  f"  [{goals}/{episode} = {pct:.0f}%]")
            total_reward = 0.0
            obs, _       = env.reset()

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if args.play:
        play(args)
    else:
        train(args)
