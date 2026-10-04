CoD4 (IW3) map -> Blender exporter
==================================
Usage:   python3 export_map.py "<Call of Duty 4 folder>" <mapname> "<output folder>"
Example: python3 export_map.py "F:/SteamLibrary/steamapps/common/Call of Duty 4" mp_backlot out/mp_backlot
Needs:   Python 3 with numpy and Pillow.
Options: --no-textures, --all-maps (also normal + specular maps)

Reads zone/english/<map>.ff (plus common_mp.ff for shared props) and main/*.iwd from
your own install. Output: <map>.gltf + <map>.bin + textures/*.png,
<map>_materials.json (material -> texture names) and <map>_entities.txt (spawns etc.).

iw3types.json and zonecmds/ are derived from the IW3 struct definitions and zone
command files of OpenAssetTools (github.com/Laupetin/OpenAssetTools, GPL-3.0).
