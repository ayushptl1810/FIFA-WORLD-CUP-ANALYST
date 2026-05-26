import os
import sys
import numpy as np
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from rl_agent.environment import TactiqEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_gated_policy():
    # 1. Load the environment and model
    env = TactiqEnv()
    env.reset(seed=42)
    model = PPO.load("rl_agent/tactiq_policy.zip", env=env)
    
    logger.info("Evaluating Gated PPO Policy over 50 matches (seeded)...")
    
    # 2. Run deterministic evaluation
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=50, 
        deterministic=True,
        warn=False
    )
    
    # Calculate Human Coach baseline return over the first 50 matches
    coach_returns = []
    for match_id in env.match_ids[:50]:
        trajectory = env.match_trajectories[match_id]
        coach_return = sum(state['payload'].get('reward', 0.0) for state in trajectory)
        coach_returns.append(coach_return)
    mean_coach = np.mean(coach_returns)
    
    print("\n" + "=" * 60)
    print("🏆 FINAL EVALUATION RESULTS (50 MATCHES)")
    print("=" * 60)
    print(f"  🤖 Gated PPO Policy Mean Match Reward:  {mean_reward:+.4f} (± {std_reward:.4f})")
    print(f"  👥 Human Coach Baseline Mean Reward:      {mean_coach:+.4f}")
    print(f"  📈 Policy Performance Improvement:         {mean_reward - mean_coach:+.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_gated_policy()

