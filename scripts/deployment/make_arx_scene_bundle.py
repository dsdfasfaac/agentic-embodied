#!/usr/bin/env python3
"""Create a Zetta ARX scene-bundle manifest from a Real2Sim run."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); root=a.run.expanduser().resolve(); final=json.loads((root/'output/final_scene.json').read_text()); assembly=json.loads((root/'scene/assembly_report.json').read_text()); layout=json.loads((root/'agent/active_layout.json').read_text())
    required=[str(x['id']) for x in layout['objects']]; targets=['tube_01']
    robot=Path('/data4/zhengyikai/ARX_Model/AC one/URDF/AC one.7z'); xml=Path('/data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d.xml'); mapping=Path('/data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d_mapping.json')
    def ref(p): return {'path':str(p.relative_to(root)),'sha256':sha(p)}
    out={'schema_version':'zetta_real2sim_scene_bundle_v1','scene_id':'tubes_2_mujoco_modified','task_name':'pickup_test_tube','source':{'real2sim_commit':'tubes_2_mujoco_modified','run_id':final['run_id'],'kind':'attempt','index':0,'retry':0,'bundle_root':str(root)},'artifacts':{'active_layout':ref(root/'agent/active_layout.json'),'assembly_report':ref(root/'scene/assembly_report.json'),'final_scene':ref(root/'output/final_scene.json'),'scene_xml':ref(root/'scene/scene.xml'),'model_mjb':ref(root/'output/model.mjb'),'settled_state':ref(root/'output/settled_state.npz')},'objects':{'required_ids':required,'target_ids':targets},'composition':{'frame_transform':'identity_v1','robot_source_archive':str(robot),'robot_source_archive_sha256':sha(robot),'robot_xml':str(xml),'robot_xml_sha256':sha(xml),'mapping':str(mapping),'mapping_sha256':sha(mapping),'cameras':'arx_task7_cameras_v1'},'reset':{'source':'settled_state','settle_after_composition_steps':0}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(a.output)
if __name__=='__main__': main()
