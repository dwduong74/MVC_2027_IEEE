import zipfile
import re

z_orig = zipfile.ZipFile(r'c:\project\IEEE\MVC_2027\SIMULATION_FILES\MVC27_Sim_original.slx', 'r')
xml_6111 = z_orig.read('simulink/systems/system_6111.xml').decode('utf-8')

subsystems = re.findall(r'<Block BlockType="SubSystem".*?</Block>', xml_6111, re.DOTALL)
for s in subsystems:
    print('--- Subsystem block in original 6111:')
    print(s)
