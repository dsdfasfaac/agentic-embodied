import io,json,time,statistics
from pathlib import Path
import numpy as np
from robots.arx.gateway.journal import Journal
from robots.arx.gateway.public import ImageStore,atomic_write
source=Path('/mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/preflight-sensors/snapshot.npz')
with np.load(source) as data:
 frames={k:data[k] for k in data.files}
results={}
for name,root in [('hdd',Path('/mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/io-benchmark')),('nvme',Path('/home/dodo/chenfu/.cache/arx-live-evidence/io-benchmark-20261009'))]:
 root.mkdir(parents=True,exist_ok=False)
 journal=Journal(root/'journal.sqlite3')
 samples=[]
 for index in range(5):
  start=time.monotonic()
  # Force three new durable image publications, as distinct real frames do.
  ImageStore(root/f'images-{index}').publish(frames)
  images_done=time.monotonic()
  stream=io.BytesIO();np.savez_compressed(stream,**frames)
  encoded=time.monotonic()
  atomic_write(root/f'sensor-{index}.npz',stream.getvalue())
  sensors_done=time.monotonic()
  for record in range(12):
   journal.record('benchmark',{'sample':index,'record':record,'state':[0.]*14})
  done=time.monotonic()
  samples.append({'images_ms':(images_done-start)*1000,'compression_ms':(encoded-images_done)*1000,'sensor_write_ms':(sensors_done-encoded)*1000,'journal_ms':(done-sensors_done)*1000,'total_ms':(done-start)*1000})
 journal.db.close()
 results[name]={'path':str(root),'samples':samples,'median_ms':{k:statistics.median(v[k] for v in samples) for k in samples[0]}}
output=source.parent.parent/'io-benchmark.json';output.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
