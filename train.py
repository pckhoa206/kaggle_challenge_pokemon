import os
import sys
import argparse

# Check requirements
try:
    import gymnasium as gym
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
except ImportError as e:
    print(f"Error: Required library is missing ({e}).")
    print("Please install requirements using the following command:")
    print("  pip install gymnasium stable-baselines3 sb3-contrib onnx onnxruntime torch")
    sys.exit(1)

import numpy as np
from env import PokemonTCGEnv
from model import PokemonFeatureExtractor
from main import agent as heuristic_agent_main, score_option
from cg.api import to_observation_class

class OpponentPolicy:
    """Wrapper for selecting opponent moves during self-play and evaluation."""
    def __init__(self, mode="heuristic", model=None):
        self.mode = mode
        self.model = model

    def __call__(self, obs_dict):
        select_data = obs_dict.get("select")
        if not select_data:
            return []
        
        # Mode 1: Heuristic opponent
        if self.mode == "heuristic":
            try:
                # Wrap the existing main.py agent function
                return heuristic_agent_main(obs_dict)
            except Exception:
                # Fallback to random
                return self._random_policy(select_data)
                
        # Mode 2: Self-play neural network opponent
        elif self.mode == "rl" and self.model is not None:
            try:
                # Extract state representation
                env_temp = PokemonTCGEnv()
                obs_obj = to_observation_class(obs_dict)
                state_vec = env_temp._extract_state(obs_obj)
                
                # Get action mask
                options = select_data.get("option", [])
                mask = np.zeros(50, dtype=bool)
                for i in range(min(len(options), 50)):
                    mask[i] = True
                    
                # Predict action
                action, _ = self.model.predict(state_vec, action_masks=mask, deterministic=True)
                
                # Fill remaining actions using heuristics if maxCount > 1
                max_count = select_data.get("maxCount", 1)
                primary_choice = min(int(action), len(options) - 1)
                choices = [primary_choice]
                
                if max_count > 1 and len(options) > 1:
                    your_idx = obs_obj.current.yourIndex
                    scored_options = []
                    for i in range(len(options)):
                        if i == primary_choice:
                            continue
                        try:
                            score = score_option(obs_obj, options[i], obs_obj.select.context, your_idx)
                        except Exception:
                            score = 100.0
                        scored_options.append((score, i))
                    scored_options.sort(key=lambda x: x[0], reverse=True)
                    for _, idx in scored_options[:max_count - 1]:
                        choices.append(idx)
                return choices[:max_count]
            except Exception:
                return self._random_policy(select_data)
                
        return self._random_policy(select_data)

    def _random_policy(self, select_data):
        max_count = select_data.get("maxCount", 1)
        options_len = len(select_data.get("option", []))
        import random
        return random.sample(list(range(options_len)), max_count)

def evaluate_agent(eval_env, model, eval_games=20) -> float:
    """Evaluate current RL model against Heuristic baseline."""
    wins = 0
    for _ in range(eval_games):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action_masks = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
        current = eval_env.unwrapped.obs_dict.get("current") if eval_env.unwrapped.obs_dict is not None else None
        if current and current.get("result") == 0:
            wins += 1
            
    win_rate = wins / eval_games
    return win_rate

def export_to_onnx(model, onnx_path="model.onnx"):
    """Export the trained SB3 PPO model to ONNX for lightweight inference on Kaggle."""
    import torch as th
    
    # Define a helper wrapper class to output policy action probabilities
    class ONNXWrapper(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.extractor = policy.features_extractor
            self.mlp = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, x):
            features = self.extractor(x)
            latent_pi, _ = self.mlp(features)
            action_logits = self.action_net(latent_pi)
            return action_logits

    # Wrap the policy
    onnx_wrapper = ONNXWrapper(model.policy)
    onnx_wrapper.eval()
    
    # Create dummy input tensor
    dummy_input = th.randn(1, 144)
    
    # Export to ONNX file
    th.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Successfully exported model to ONNX: {onnx_path} (size ~ {os.path.getsize(onnx_path) / 1024:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="Self-play DRL training pipeline for Pokemon TCG.")
    parser.add_argument("--steps", type=int, default=50000, help="Total number of steps to train.")
    args = parser.parse_args()

    # 1. Setup Environment
    # Default opponent is the Heuristic baseline agent
    opp_policy = OpponentPolicy(mode="heuristic")
    env = PokemonTCGEnv(opponent_policy=opp_policy)

    # 2. Define Custom Neural Network policy model structure
    policy_kwargs = dict(
        features_extractor_class=PokemonFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # 3. Initialize Masked PPO Agent
    print("Initializing MaskablePPO Agent...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99
    )
    
    # Attach model back to env for evaluation wrapper
    env.model = model

    # 4. Training loop with self-play evaluation
    total_steps = args.steps
    steps_per_epoch = 10000
    current_step = 0
    opponent_mode = "heuristic"

    # Create a completely separate environment for evaluation to avoid state pollution
    eval_env = PokemonTCGEnv(opponent_policy=OpponentPolicy(mode="heuristic"))

    print(f"Starting Training: Total steps = {total_steps}, Epoch steps = {steps_per_epoch}")
    
    while current_step < total_steps:
        # Train for 1 epoch
        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False)
        current_step += steps_per_epoch
        print(f"Completed step {current_step}/{total_steps}")
        
        # Evaluate performance against heuristic baseline using the dedicated eval_env
        win_rate = evaluate_agent(eval_env, model, eval_games=20)
        print(f"Evaluation against Heuristic Baseline: Win Rate = {win_rate * 100:.1f}%")
        
        # Check if we should update opponent for Self-play
        if win_rate > 0.60:
            print("RL Agent achieved > 60% win rate! Updating opponent to current RL checkpoint (Self-play)...")
            opp_policy.mode = "rl"
            opp_policy.model = model
            opponent_mode = "self-play (rl)"
        else:
            print("Win rate under 60%. Continuing training against current opponent...")
            
        # Re-set training opponent to active self-play or baseline pool
        env.opponent_policy = opp_policy
        
        # Force reset the training environment and update model's last observation buffer
        # to clear any C++ global state pollution caused by the evaluation environment games.
        model._last_obs = model.env.reset()
        
        # Save checkpoints
        model.save("model_checkpoint.zip")
        
    print("Training Complete. Saving final model and exporting to ONNX...")
    model.save("model_final.zip")
    export_to_onnx(model, "model.onnx")

if __name__ == "__main__":
    main()
