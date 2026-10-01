# Copyright (c) 2026 Zetta Contributors
"""Content-addressed public evidence and durable ARX learner state."""
from __future__ import annotations

import json
from pathlib import Path

from zetta.evolution.jsonio import atomic_write_json, canonical_sha256

from .contracts import CampaignContract, Checkpoint, Evidence


class LearningStore:
 def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); (self.root/'evidence').mkdir(exist_ok=True)
 def put(self,kind,payload,visibility='public'):
  data={'kind':kind,'payload':payload,'visibility':visibility}; cid=canonical_sha256(data); atomic_write_json(self.root/'evidence'/f'{cid}.json',data); return cid
 def read(self,cid,offline=False):
  data=json.loads((self.root/'evidence'/f'{cid}.json').read_text())
  if data['visibility']=='private' or (data['visibility']=='offline_authorized' and not offline): raise PermissionError('evidence is not learner-visible')
  return Evidence(content_id=cid,**data)
 def save_contract(self,c):
  target=self.root/'campaign_contract.json'
  if target.exists():
   existing=CampaignContract.model_validate_json(target.read_text())
   if existing != c: raise ValueError('campaign root contains a different contract')
   return target
  return atomic_write_json(target,c.model_dump())
 def contract(self): return CampaignContract.model_validate_json((self.root/'campaign_contract.json').read_text())
 def checkpoint(self,c): return atomic_write_json(self.root/'checkpoint.json',c.model_dump(),overwrite=True)
 def load_checkpoint(self): return Checkpoint.model_validate_json((self.root/'checkpoint.json').read_text()) if (self.root/'checkpoint.json').exists() else None
