"""Add controller attribution BELOW original RGB pixels, from action-source log."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);a=p.parse_args()
    sources=['initial_observation']+np.load(a.session/'trajectory.npz')['command_sources'].tolist()
    replay_frames=0
    if (a.session/'approved_replay.jsonl').exists():
        replay_frames=json.loads((a.session/'approved_replay.jsonl').read_text().splitlines()[-1])['frames']
    labels={'initial_observation':'INITIAL OBSERVATION',
            'original_prefix':'COSMOS | original action prefix replay',
            'set_gripper':'C&R | Agent-approved gripper command',
            'move_eef':'C&R | Agent-approved EEF move',
            'hold':'Agent-approved hold',
            'cosmos_after_recovery':'COSMOS | fresh inference AFTER recovery',
            'cosmos_before_recovery':'COSMOS | fresh inference before recovery'}
    font_path=next((p for p in map(Path, [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf']) if p.exists()), Path('/nonexistent-font'))
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    ffmpeg=shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
    decoder=subprocess.Popen([ffmpeg,'-v','error','-i',str(a.session/'three_view.mp4'),
        '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    out=a.session/'three_view_controller_labels.mp4'
    encoder=subprocess.Popen([ffmpeg,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s','960x288','-r','15','-i','-','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    for i,source in enumerate(sources):
        raw=decoder.stdout.read(960*240*3)
        if len(raw)!=960*240*3:raise ValueError(f'missing video frame {i}')
        canvas=Image.new('RGB',(960,288),(20,25,30))
        canvas.paste(Image.frombytes('RGB',(960,240),raw),(0,0))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,242),f'frame {i:04d} | sim {i/15:05.2f}s | {labels.get(source,source)}',font=font,
                  fill=(255,206,102) if source in ('move_eef','set_gripper','hold') else (130,215,250))
        note=('RECORDED ACTIONS (not a new Agent trial) | handoff is NOT task success'
              if replay_frames and i<=replay_frames else
              'RGB views only | pauses for Agent review omitted | handoff is NOT task success')
        draw.text((12,264),note,font=font,fill=(225,225,225))
        encoder.stdin.write(canvas.tobytes())
    assert not decoder.stdout.read(1),'extra unlogged video frame'
    encoder.stdin.close();assert encoder.wait()==0;assert decoder.wait()==0
    print(json.dumps({'video':str(out),'frames':len(sources),'fps':15,'original_pixels_preserved_before_encoding':True}))


if __name__=='__main__':main()
