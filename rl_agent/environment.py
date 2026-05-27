import os
import sys
import random
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple

# Bootstrap project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class TactiqEnv(gym.Env):
    """
    Gymnasium environment for offline RL training of the Tactiq substitution advisor
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
        qdrant_grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
        q_client = QdrantClient(url=qdrant_url, check_compatibility=False, prefer_grpc=True, grpc_port=qdrant_grpc_port)
        
        logger.info("Loading offline game states from Qdrant into memory...")
        res = q_client.scroll(
            collection_name="historical_states",
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
        points = res[0]
        logger.info(f"Loaded {len(points)} transitions successfully.")
        
        # Group states by match_id and sort chronologically by minute
        self.match_trajectories: Dict[int, List[Dict[str, Any]]] = {}
        for p in points:
            payload = p.payload
            if payload is None:
                continue
            match_id = int(payload['match_id'])

            if match_id not in self.match_trajectories:
                self.match_trajectories[match_id] = []
            
            self.match_trajectories[match_id].append({
                'id': p.id,
                'vector': p.vector,
                'payload': payload
            })
            
        for match_id in self.match_trajectories:
            self.match_trajectories[match_id].sort(key=lambda x: x['payload']['minute'] if x['payload'] else 0)
            
        self.match_ids = list(self.match_trajectories.keys())
        
        # Compute dataset-wide running Mean and Standard Deviation for all 21 features
        all_vectors = []
        for match_id in self.match_trajectories:
            for state in self.match_trajectories[match_id]:
                all_vectors.append(state['vector'])
                
        all_vectors_arr = np.array(all_vectors, dtype=np.float32)
        self.obs_mean = np.mean(all_vectors_arr, axis=0)
        self.obs_std = np.std(all_vectors_arr, axis=0)
        # Prevent division by zero for static or zero-padded features
        self.obs_std[self.obs_std == 0.0] = 1.0
        
        # Define Gymnasium Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(24,), 
            dtype=np.float32
        )
        
        # Action Space: Discrete(4) -> [0: Wait, 1: Sub Attacker, 2: Sub Midfielder, 3: Sub Defender]
        self.action_space = spaces.Discrete(4)
        
        # Environment Tracking State
        self.current_match_id: int = self.match_ids[0] if self.match_ids else 0
        self.current_step: int = 0
        
    def normalize_obs(self, vector: List[float]) -> np.ndarray:
        """
        Normalizes a raw 24D state vector using the dataset-wide mean and standard deviation.
        """
        return (np.array(vector, dtype=np.float32) - self.obs_mean) / self.obs_std
        
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean array where True represents a valid action.
        """
        state = self.match_trajectories[self.current_match_id][self.current_step]
        payload = state['payload']
        assert payload is not None, "State payload cannot be None"
        subs_used = payload.get('subs_used', 0.0)
        
        if subs_used >= 3.0:
            return np.array([True, False, False, False], dtype=np.bool_)
            
        # Dynamic Action Masking for Critical Players (Antigravity Extension)
        # Vector index 23 corresponds to the continuous fatigue_state
        fatigue_state = state['vector'][23]
        if fatigue_state >= 0.8:
            # Force substitution (Wait is illegal)
            return np.array([False, True, True, True], dtype=np.bool_)
            
        return np.array([True, True, True, True], dtype=np.bool_)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Select a random match for this episode using the environment's seeded generator
        self.current_match_id = int(self.np_random.choice(self.match_ids))
        self.current_step = 0
        
        state = self.match_trajectories[self.current_match_id][self.current_step]
        payload = state['payload']
        assert payload is not None, "State payload cannot be None"
        
        obs = self.normalize_obs(state['vector'])
        
        info = {
            "match_id": self.current_match_id,
            "minute": payload['minute'],
            "action_mask": self.get_action_mask()
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state = self.match_trajectories[self.current_match_id][self.current_step]
        payload = state['payload']
        assert payload is not None, "State payload cannot be None"
        
        # Retrieve empirical reward and action taken
        empirical_action = payload.get('action_taken', 0)
        empirical_reward = payload.get('reward', 0.0)
        
        # Determine reward based on alignment with historical choice
        reward = 0.0
        if action == empirical_action:
            reward = empirical_reward
        else:
            # Action does not match what the coach historically did
            if action == 0:
                # Agent chose to Wait but coach historically substituted.
                # Compute Wait Inaction Penalty dynamically:
                minute = payload['minute']
                drift = payload.get('candidate_drift', 0.0)
                momentum = payload.get('momentum_15m', 0.0)
                subs_used = payload.get('subs_used', 0.0)
                
                if minute > 60:
                    if drift > 0.7:
                        reward -= 0.2
                    if momentum < -0.3 and subs_used < 3.0:
                        reward -= 0.3
            else:
                # Agent chose to substitute (1, 2, or 3) but coach historically waited.
                # If they have no subs remaining, this is an illegal action (normally masked out)
                subs_used = payload.get('subs_used', 0.0)
                if subs_used >= 3.0:
                    reward = -0.8
                else:
                    # Penalize unobserved actions slightly to encourage staying close to empirical data
                    reward = -0.3
        
        # Advance state chronologically
        self.current_step += 1
        trajectory = self.match_trajectories[self.current_match_id]
        
        terminated = self.current_step >= len(trajectory)
        truncated = False
        
        if terminated:
            # Episode ended, return zero observation
            obs = np.zeros((24,), dtype=np.float32)
            info = {
                "match_id": self.current_match_id,
                "minute": 90,
                "action_mask": np.array([True, False, False, False], dtype=np.bool_)
            }
        else:
            next_state = trajectory[self.current_step]
            next_payload = next_state['payload']
            assert next_payload is not None, "Next state payload cannot be None"
            obs = self.normalize_obs(next_state['vector'])
            info = {
                "match_id": self.current_match_id,
                "minute": next_payload['minute'],
                "action_mask": self.get_action_mask()
            }
            
        return obs, float(reward), terminated, truncated, info
