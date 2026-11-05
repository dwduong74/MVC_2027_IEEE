import zipfile
import os

slx_path = r'c:\project\IEEE\MVC_2027\SIMULATION_FILES\MVC27_Sim.slx'
temp_path = r'c:\project\IEEE\MVC_2027\SIMULATION_FILES\MVC27_Sim_fixed.slx'

new_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="system_4919" Target="system_4919.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/system"/>
  <Relationship Id="system_4931" Target="system_4931.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/system"/>
  <Relationship Id="system_6162" Target="system_6162.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/system"/>
  <Relationship Id="system_6165" Target="system_6165.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/system"/>
  <Relationship Id="system_7000" Target="system_7000.xml" Type="http://schemas.mathworks.com/simulink/2010/relationships/system"/>
</Relationships>"""

with zipfile.ZipFile(slx_path, 'r') as zin:
    with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'simulink/systems/_rels/system_6111.xml.rels':
                zout.writestr(item, new_rels)
            else:
                zout.writestr(item, zin.read(item.filename))

os.replace(temp_path, slx_path)
print("Updated system_6111.xml.rels in MVC27_Sim.slx successfully!")
