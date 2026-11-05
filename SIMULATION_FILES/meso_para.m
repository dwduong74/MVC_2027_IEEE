%
% Parameter of the Energy Management & Motion Control Strategy
% IEEE-VTS Motor Vehicle Challenge 2027 (MVC2027) - Phase 2 Advanced
%

% 0. Environment and Driving Condition Safeguard
% Ensures Road and Wind structures exist in base workspace even if path is misconfigured
if ~exist('Road', 'var') || ~isfield(Road, 'Mu_l')
    if exist('driving_condition', 'file')
        driving_condition;
    end
end
if ~exist('Road', 'var') || ~isfield(Road, 'Mu_l')
    Road.Mu_h    =  0.85;                       % friction coefficient (high)
    Road.Mu_l    =  0.25;                       % friction coefficient (low)
    Road.Mu_m    =  0.60;                       % friction coefficient (medium)
    Road.L_in    =  12.5;                       % position the vehicle enters the low-friction-surface [m]
    Road.L_out   =  42.5;                       % position the vehicle leaves the low-friction-surface [m]
    Road.M_in    =  1976.99;                    % position the vehicle enters the medium-friction-surface [m]
    Road.M_out   =  2011.03;                    % position the vehicle leaves the medium-friction-surface [m]
end
if ~exist('Wind', 'var') || ~isfield(Wind, 'force')
    Wind.time_start = 45;                       % starting time [s]
    Wind.time_stop  = 55;                       % stoping time [s]
    Wind.force      = 1600;                     % wind lateral force [N]
    Wind.moment     = 160;                      % wind lateral moment [N.m]
end

if (control_strategy == 1)
    flag.DYC = 0; flag.current = 0;
end
if (control_strategy == 2)
    flag.DYC = 1; flag.current = 1;
end
% "1": Baseline strategy (BS): 
% The vehicle is controlled by active front steering
% The torque distribution ratios: tdis_FR = tdis_FL = 0.27; 
%                                 tdis_RR = tdis_RL = 0.23
%                            (No yaw moment control is applied)
% The d-axis currents of all motors are zero

% "2": Improved strategy (IS) & Hybrid MESO (Phase 2 Advanced):
% The vehicle is controlled by active front steering + Direct Yaw Moment Control (DYC)
% 1. Bidirectional Cascade Traction Control (TCS) & Anti-Lock Braking (ABS) (Ueno et al., 2025)
% 2. Full-Cycle Regenerative Braking with 70.5/29.5 Front/Rear Loss-Minimization (LMC)
% 3. Smooth DWPT Centering Barrier with Band-Limited DYC
% 4. Adaptive Torque-Dependent d-Axis Current Field Weakening (Morimoto et al., 1994)

% 1. Active front steering
Steer.Kp_pos =  0.02;                      % P gain of the lateral position control via front steering
Steer.Kp_yaw =  0.2;                       % P gain of the yaw angle control via front steering

% 2. Yaw moment control (Improved strategy)
DYC.Kp_pos       =  4000;                      % P gain of lateral position control via DYC
DYC.Ki_pos       =  35;                        % I gain of lateral position control via DYC (fast wind rejection)
DYC.Kd_pos       =  1000;                      % D gain of lateral position control via DYC
DYC.Taud_pos     =  0.005;                     % Time constant of Derivative terms
DYC.Kp_yaw       =  5000;                      % P gain of yaw angle control via DYC
DYC.damping_gain =  1500;                      % High-speed yaw damping gain [Nm/(rad/s)]
DYC.tau_filter   =  0.015;                     % DYC low-pass filter time constant [s]
DYC.delta_T_max  =  250;                       % Differential torque clamp per wheel [Nm]
DYC.Y_barrier    =  0.035;                     % Precision DWPT lateral barrier threshold [m]
DYC.K_barrier    =  8000.0;                    % Restoring moment gain [Nm/m]

% 3. Torque distribution (Baseline strategy)
tdis_FR          =  0.27;                      % torque distribution gain (front right)
tdis_FL          =  0.27;                      % torque distribution gain (front left)
tdis_RR          =  0.23;                      % torque distribution gain (rear right)
tdis_RL          =  1 - (tdis_FR + tdis_FL + tdis_RR);  % torque distribution gain (rear left)

% 4. Cascade Traction Control System (TCS)
% Ref: T. Ueno, B.-M. Nguyen, S. Nagai and H. Fujimoto, "Wheel Velocity Based 
% Cascade Driving Force Control for Electric Vehicles," IEEE/ASME T-Mech, 2025.
TCS.K_tcs            =  800.0;                 % TCS feedback gain [Nm / (m/s)]
TCS.slip_ratio_allow =  0.07;                  % Maximum allowable slip ratio (7% optimal)
TCS.slip_speed_min   =  0.12;                  % Minimum allowed slip speed [m/s]

% 5. Regenerative Braking & Anti-Lock System (ABS)
Regen.enabled        =  1;                     % 1: Enable bidirectional regenerative braking
Regen.slip_allow     =  0.06;                  % Allowable slip ratio during braking (6%)
Regen.K_abs          =  800.0;                 % Anti-lock proportional reduction gain [Nm / (m/s)]
Regen.k_f            =  0.705;                 % Front regeneration bias ratio

% 6. Electronic Stability Control (ESC) Barrier
% Limits maximum allowable yaw rate to prevent spinout on extreme low-friction transitions
ESC.gamma_limit      =  0.25;                  % Maximum allowable yaw rate limit [rad/s]
ESC.directional      =  1;                     % Only clamp when moment destabilizes (Nz * gamma > 0)
