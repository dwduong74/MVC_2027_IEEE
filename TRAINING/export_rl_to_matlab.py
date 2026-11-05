"""
Export Hybrid RL Policy to Standalone MATLAB Script & JSON
===========================================================
Extracts weights and biases from the trained PyTorch ActorCritic model
and generates:
  1. `rl_policy_weights.json`: dictionary of arrays for all layers.
  2. `c:\project\IEEE\MVC_2027\SIMULATION_FILES\rl_policy_inference.m`:
     pure MATLAB Function compatible with Simulink without Neural Network Toolbox!
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_hybrid_ppo import ActorCritic


def export_model_to_matlab(model_path=None, output_matlab_path=None, output_json_path=None):
    save_dir = os.path.dirname(__file__)
    if model_path is None:
        model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
            
    if output_matlab_path is None:
        output_matlab_path = os.path.abspath(os.path.join(save_dir, "..", "SIMULATION_FILES", "rl_policy_inference.m"))
        
    if output_json_path is None:
        output_json_path = os.path.join(save_dir, "rl_policy_weights.json")
        
    # Load model
    model = ActorCritic(state_dim=10, action_dim=3, hidden_dim=64)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Model path {model_path} not found. Exporting initialized zero-residual policy.")
        
    # Extract weights
    w1 = model.fc1.weight.detach().numpy()    # (64, 10)
    b1 = model.fc1.bias.detach().numpy()      # (64,)
    w2 = model.fc2.weight.detach().numpy()    # (64, 64)
    b2 = model.fc2.bias.detach().numpy()      # (64,)
    w_mu = model.mu_head.weight.detach().numpy() # (3, 64)
    b_mu = model.mu_head.bias.detach().numpy()   # (3,)
    
    # Save to JSON
    weights_dict = {
        'W1': w1.tolist(),
        'b1': b1.tolist(),
        'W2': w2.tolist(),
        'b2': b2.tolist(),
        'W_mu': w_mu.tolist(),
        'b_mu': b_mu.tolist(),
    }
    with open(output_json_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print(f"Saved weights JSON to: {output_json_path}")
    
    # Format MATLAB matrix strings
    def mat2str(mat):
        rows = []
        for row in mat:
            rows.append(" ".join([f"{val:+.8e}" for val in row]))
        return "[" + ";\n       ".join(rows) + "]"
        
    def vec2str(vec):
        return "[" + "; ".join([f"{val:+.8e}" for val in vec]) + "]"

    w1_str = mat2str(w1)
    b1_str = vec2str(b1)
    w2_str = mat2str(w2)
    b2_str = vec2str(b2)
    w_mu_str = mat2str(w_mu)
    b_mu_str = vec2str(b_mu)

    matlab_code = f"""function [alpha_throttle, r_front, delta_Nz_rl] = rl_policy_inference(obs)
%#codegen
% RL_POLICY_INFERENCE
% Auto-generated Standalone Feedforward Neural Network Inference Function
% for IEEE VTS Motor Vehicle Challenge 2027 (MVC 2027)
%
% Architecture: MLP [10 -> 64 -> 64 -> 3] with Tanh activations
% Inputs:
%   obs: [10x1] normalized observation vector:
%     1: Vx / 20.0
%     2: (Vref - Vx) / 5.0
%     3: PosY / 0.10
%     4: Vy / 2.0
%     5: phi / 0.10
%     6: gamma / 0.30
%     7: ay / 2.0
%     8: max_slip_excess / 0.15
%     9: in_dwpt_flag (1 or 0)
%    10: PosX / 2022.0
%
% Outputs:
%   alpha_throttle: Throttle scaling factor in [0.8, 1.2]
%   r_front:        Front/Rear torque distribution ratio in [0.35, 0.65]
%   delta_Nz_rl:    Auxiliary yaw moment command in [-500, 500] [Nm]
%
% Dependencies: NONE. Runs in <0.05ms. Fully compatible with MATLAB R2022b.

persistent W1 b1 W2 b2 W_mu b_mu is_init

if isempty(is_init)
    W1 = {w1_str};
    b1 = {b1_str};
    W2 = {w2_str};
    b2 = {b2_str};
    W_mu = {w_mu_str};
    b_mu = {b_mu_str};
    is_init = true;
end

% Forward Pass
u = obs(:);

% Layer 1
h1 = tanh(W1 * u + b1);

% Layer 2
h2 = tanh(W2 * h1 + b2);

% Output Layer
a_raw = tanh(W_mu * h2 + b_mu);

% Scale and Map Actions to Physical Ranges
% a_raw(1) in [-1, 1] -> alpha_throttle in [0.8, 1.2]
alpha_throttle = 1.0 + 0.20 * a_raw(1);

% a_raw(2) in [-1, 1] -> r_front in [0.35, 0.65]
r_front = 0.50 + 0.15 * a_raw(2);

% a_raw(3) in [-1, 1] -> delta_Nz_rl in [-500, 500] Nm
delta_Nz_rl = 500.0 * a_raw(3);

end
"""

    with open(output_matlab_path, 'w', encoding='utf-8') as f:
        f.write(matlab_code)
        
    print(f"Generated standalone MATLAB inference code: {output_matlab_path}")
    return output_matlab_path


if __name__ == '__main__':
    export_model_to_matlab()
