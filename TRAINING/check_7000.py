import zipfile
import re

z = zipfile.ZipFile(r'c:\project\IEEE\MVC_2027\SIMULATION_FILES\MVC27_Sim.slx', 'r')
xml_6111 = z.read('simulink/systems/system_6111.xml').decode('utf-8')
xml_7000 = z.read('simulink/systems/system_7000.xml').decode('utf-8')
rels_6111 = z.read('simulink/systems/_rels/system_6111.xml.rels').decode('utf-8')

print('system_7000 in rels:', 'system_7000' in rels_6111)
print('system_7000 block in 6111:', '7000' in xml_6111)

inports = re.findall(r'<Block BlockType="Inport"[^>]+>', xml_7000)
outports = re.findall(r'<Block BlockType="Outport"[^>]+>', xml_7000)
print('Inports count in 7000:', len(inports))
print('Outports count in 7000:', len(outports))

lines = re.findall(r'<Line>.*?</Line>', xml_6111, re.DOTALL)
lines_7000 = [l for l in lines if '7000' in l]
print('Lines connecting to 7000 in 6111:', len(lines_7000))
for l in lines_7000:
    cleaned = ' '.join(l.split())
    print('  ', cleaned)
