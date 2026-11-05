import zipfile
import re

z = zipfile.ZipFile(r'c:\project\IEEE\MVC_2027\SIMULATION_FILES\MVC27_Sim.slx', 'r')
xml_6111 = z.read('simulink/systems/system_6111.xml').decode('utf-8')

# Get all blocks and their SIDs in 6111
blocks = re.findall(r'<Block BlockType="([^"]+)" Name="([^"]+)" SID="([^"]+)"', xml_6111)
block_sids = {b[2]: (b[0], b[1]) for b in blocks}

print("Block SIDs in 6111:")
for sid, (btype, bname) in sorted(block_sids.items(), key=lambda x: int(x[0])):
    print(f"  SID {sid}: {btype} '{bname}'")

# Check all Lines and their Src / Dst
lines = re.findall(r'<Line>(.*?)</Line>', xml_6111, re.DOTALL)
print(f"\nTotal lines in 6111: {len(lines)}")

issues = []
for i, l in enumerate(lines):
    src = re.findall(r'<P Name="Src">([^<]+)</P>', l)
    dsts = re.findall(r'<P Name="Dst">([^<]+)</P>', l)
    
    if not src:
        issues.append(f"Line {i} has no Src: {l}")
    else:
        src_sid = src[0].split('#')[0]
        if src_sid not in block_sids:
            issues.append(f"Line {i} invalid Src SID {src_sid}")
            
    for d in dsts:
        dst_sid = d.split('#')[0]
        if dst_sid not in block_sids:
            issues.append(f"Line {i} invalid Dst SID {dst_sid}")

if issues:
    print("\nISSUES FOUND:")
    for iss in issues:
        print(" ", iss)
else:
    print("\nALL LINES IN system_6111 ARE 100% VALID!")
