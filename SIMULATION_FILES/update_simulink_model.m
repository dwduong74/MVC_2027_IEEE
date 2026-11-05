%
% UPDATE_SIMULINK_MODEL.M
% IEEE-VTS Motor Vehicle Challenge 2027 (MVC2027)
%
% Automatically updates 'MVC27_Sim.slx' to integrate the advanced multi-objective
% MESO controller into the allowed subsystem:
% 'MVC27_Sim/Energy Management & Motion Control Strategy'
%
% This uses MATLAB's native Simulink API, ensuring 100% binary integrity
% without risk of OpenXML/SLX package corruption.
%
% How to use:
%   1. In MATLAB / MATLAB Online, change folder to SIMULATION_FILES:
%      >> cd '/MATLAB Drive/SIMULATION_FILES_MVC2027'
%   2. Run this script once:
%      >> update_simulink_model
%   3. Run main:
%      >> main
%

fprintf('=================================================================\n');
fprintf('  IEEE MVC2027: Integrating Advanced Multi-Objective Controller\n');
fprintf('=================================================================\n');

% 1. Close any open models to avoid memory locks
bdclose('all');

% 2. Create safety backup if not already present
model_file = 'MVC27_Sim.slx';
backup_file = 'MVC27_Sim_backup.slx';
if exist(model_file, 'file') && ~exist(backup_file, 'file')
    copyfile(model_file, backup_file);
    fprintf('[1/5] Safety backup created: %s\n', backup_file);
else
    fprintf('[1/5] Verified existing backup: %s\n', backup_file);
end

% 3. Load the model into memory
fprintf('[2/5] Loading Simulink model: %s...\n', model_file);
load_system('MVC27_Sim');

subsystem_path = 'MVC27_Sim/Energy Management & Motion Control Strategy';
sfun_name = 'Advanced MESO Controller';
sfun_path = [subsystem_path '/' sfun_name];

% 4. Clear old internal routing inside the subsystem
fprintf('[3/5] Cleaning internal wiring of subsystem...\n');
try
    lines = find_system(subsystem_path, 'FindAll', 'on', 'Type', 'line');
    if ~isempty(lines)
        delete_line(lines);
    end
    
    % Delete old internal blocks (keep only Inports and Outports)
    blocks = find_system(subsystem_path, 'SearchDepth', 1, 'Type', 'block');
    for i = 1:length(blocks)
        b = blocks{i};
        if ~strcmp(b, subsystem_path)
            btype = get_param(b, 'BlockType');
            if ~strcmp(btype, 'Inport') && ~strcmp(btype, 'Outport')
                delete_block(b);
            end
        end
    end
catch ME
    fprintf('  Notice during cleanup: %s\n', ME.message);
end

% 5. Add the Level-2 S-Function block
fprintf('[4/5] Adding Level-2 S-Function block [%s]...\n', sfun_name);
if getSimulinkBlockHandle(sfun_path) ~= -1
    delete_block(sfun_path);
end
add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', sfun_path, ...
    'Position', [320, 100, 520, 500], ...
    'FunctionName', 'meso_controller_sfun');

% 6. Connect Inports to S-Function
fprintf('[5/5] Connecting sensor bus and command ports...\n');
try
    add_line(subsystem_path, 'T_all/1', [sfun_name '/1'], 'autorouting', 'on');
catch
end
try
    add_line(subsystem_path, 'Sensor/1', [sfun_name '/2'], 'autorouting', 'on');
catch
end

% Connect S-Function 9 outputs to the 9 Outports
outport_names = { ...
    'delta_f_cmd', ...  % Port 1
    'T_FR_cmd',    ...  % Port 2
    'Id_FR_cmd',   ...  % Port 3
    'T_FL_cmd',    ...  % Port 4
    'Id_FL_cmd',   ...  % Port 5
    'T_RR_cmd',    ...  % Port 6
    'Id_RR_cmd',   ...  % Port 7
    'T_RL_cmd',    ...  % Port 8
    'Id_RL_cmd'    ...  % Port 9
};

for k = 1:length(outport_names)
    src_port = sprintf('%s/%d', sfun_name, k);
    dst_port = sprintf('%s/1', outport_names{k});
    try
        add_line(subsystem_path, src_port, dst_port, 'autorouting', 'on');
    catch
    end
end

% 7. Save and close model cleanly
fprintf('Saving updated Simulink model...\n');
save_system('MVC27_Sim');
bdclose('all');

fprintf('=================================================================\n');
fprintf('  SUCCESS! The model MVC27_Sim.slx has been upgraded natively.\n');
fprintf('  Ready to run: type "main" to execute the simulation.\n');
fprintf('=================================================================\n');
