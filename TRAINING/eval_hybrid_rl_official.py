import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_rl_env import HybridKanonEnv
from train_hybrid_ppo import ActorCritic

def eval_official():
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        
    model = ActorCritic(state_dim=10, action_dim=3, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    env = HybridKanonEnv(dt_macro=0.02, dt_micro=0.001)
    
    cond_names = [
        "Cond 1 (mu=0.85, No Wind)",
        "Cond 2 (mu=0.85, Wind 1600N)",
        "Cond 3 (mu-change, No Wind)",
        "Cond 4 (mu-change, Wind 1600N)"
    ]
    
    total_scores = []
    print("=" * 78)
    print("  EVALUATION OF TRAINED HYBRID RL ON 4 OFFICIAL MVC 2027 CONDITIONS")
    print("=" * 78)
    
    for c_id in [1, 2, 3, 4]:
        obs = env.reset(condition_id=c_id)
        done = False
        max_y = 0.0
        while not done:
            action, _, _ = model.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if abs(info['PosY']) > max_y:
                max_y = abs(info['PosY'])
                
        total_scores.append(info['total_Phi'])
        print("%-32s -> Phi=%7.2f (Phi1=%5.1f, Phi2*60=%5.2f, Phi3*3=%5.1f, Phi4*10=%5.1f, max|y|=%5.1fmm)" % (
            cond_names[c_id - 1], info['total_Phi'],
            info['cum_Phi1'], 60.0 * info['cum_Phi2'],
            3.0 * info['cum_Phi3'], 10.0 * info['cum_Phi4'],
            max_y * 1000.0
        ))
        
    print("-" * 78)
    print("TOTAL SCORE PHI (SUM OF 4 CONDITIONS): %.2f" % sum(total_scores))
    print("=" * 78)

if __name__ == '__main__':
    eval_official()
