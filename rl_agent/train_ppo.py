import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Bootstrap project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from rl_agent.environment import TactiqEnv
from rl_agent.networks import GatedTacticalExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Sets the random seed across Python's random, NumPy, and PyTorch for 100% deterministic reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducible PyTorch convolution/matrix multiply algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")


class BCClassifier(nn.Module):
    """
    Supervised classifier using our Learned Gating Mask to mimic human coach decisions.
    """
    def __init__(self, observation_space) -> None:
        super().__init__()
        self.extractor = GatedTacticalExtractor(observation_space, features_dim=64)
        self.classifier = nn.Linear(64, 4)  # 4 discrete action logits
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        return self.classifier(features)


def pretrain_behavior_cloning(env: TactiqEnv, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3) -> BCClassifier:
    """
    Supervised Behavior Cloning (BC) Pre-training Phase.
    Directly trains a PyTorch classification network to predict the empirical coach actions (0-3)
    from the 24D state vector. This initializes our PPO Actor to follow realistic human coaching.
    """
    logger.info("--- Starting Behavior Cloning Pre-Training Phase ---")
    
    # 1. Extract all 8,568 empirical states and action labels from the environment memory
    states = []
    actions = []
    
    for match_id in env.match_trajectories:
        for step_data in env.match_trajectories[match_id]:
            states.append(env.normalize_obs(step_data['vector']))
            payload = step_data['payload']
            assert payload is not None
            actions.append(payload.get('action_taken', 0))
            
    X = torch.tensor(np.array(states), dtype=torch.float32)
    y = torch.tensor(np.array(actions), dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate our Gated Policy Classifier using our Learned Gating Mask
    bc_model = BCClassifier(env.observation_space)
    optimizer = optim.Adam(bc_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    bc_model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = bc_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        avg_loss = total_loss / total
        accuracy = (correct / total) * 100
        logger.info(f"Epoch {epoch:02d}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
        
    logger.info("Behavior Cloning Pre-Training Complete.")
    return bc_model


def train_ppo():
    # Set global seed first for PyTorch/Numpy parameter and data shuffling determinism
    set_seed(42)
    
    # 1. Instantiate the in-memory offline environment and seed it
    env = TactiqEnv()
    env.reset(seed=42)
    
    # 2. Pre-train Actor parameters using Behavior Cloning (Supervised Phase)
    bc_model = pretrain_behavior_cloning(env, epochs=15)
    
    # 3. Configure the PPO training algorithm
    policy_kwargs = dict(
        features_extractor_class=GatedTacticalExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64], vf=[64])  # Actor and Critic heads
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None,
        seed=42  # Seed PPO exploration and optimization
    )
    
    # 4. Copy the pre-trained BC weights directly into the PPO Actor policy network
    logger.info("Initializing PPO Actor weights with pre-trained Behavior Cloning weights...")
    
    # Assert model.policy has ActorCriticPolicy type for static analyzer validation
    policy = model.policy
    assert isinstance(policy, ActorCriticPolicy), "PPO policy must be an instance of ActorCriticPolicy"
    
    # Load feature extractor and classifier weights
    ppo_actor_state = policy.features_extractor.state_dict()
    bc_extractor_state = bc_model.extractor.state_dict()
    
    # Align and copy extractor parameters
    for name, param in bc_extractor_state.items():
        if name in ppo_actor_state:
            ppo_actor_state[name].copy_(param)
            
    # Copy action logit classifier parameters to PPO policy action net using standard state dicts
    policy.action_net.load_state_dict(bc_model.classifier.state_dict())
    
    logger.info("PPO Actor weights successfully initialized.")
    
    # 5. Run reinforcement learning fine-tuning
    logger.info("--- Starting PPO Reinforcement Learning Phase ---")
    model.learn(total_timesteps=100000)
    
    # 6. Save the final trained model
    os.makedirs("rl_agent", exist_ok=True)
    save_path = "rl_agent/tactiq_policy.zip"
    model.save(save_path)
    logger.info(f"Training complete! Gated PPO policy successfully saved to {save_path}")
    
    # 7. Final overall average values evaluation
    logger.info("Computing final deterministic performance metrics against human baseline...")
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Evaluate PPO policy deterministically
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=100,
        deterministic=True,
        warn=False
    )
    
    # Calculate Human Coach baseline return over the first 100 matches
    coach_returns = []
    for match_id in env.match_ids[:100]:
        trajectory = env.match_trajectories[match_id]
        coach_return = sum(state['payload'].get('reward', 0.0) for state in trajectory)
        coach_returns.append(coach_return)
    mean_coach = np.mean(coach_returns)
    
    print("\n" + "=" * 60)
    print("🏆 FINAL POST-TRAINING PERFORMANCE SUMMARY (100 MATCHES)")
    print("=" * 60)
    print(f"  🤖 Gated PPO Policy Mean Match Reward:  {mean_reward:+.4f} (± {std_reward:.4f})")
    print(f"  👥 Human Coach Baseline Mean Reward:      {mean_coach:+.4f}")
    print(f"  📈 Policy Performance Improvement:         {mean_reward - mean_coach:+.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_ppo()

