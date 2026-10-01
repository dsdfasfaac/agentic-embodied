#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run automated or interactive ARX high-level learning session."""
import argparse
import json
from pathlib import Path

from zetta.evolution.arx.contracts import CampaignContract
from zetta.evolution.arx.session import LearningSession
from zetta.planner.base import build_planner
from zetta.evolution.arx.coordinator import CampaignCoordinator


def main():
 p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--interactive',action='store_true');a=p.parse_args(); contract=CampaignContract.model_validate_json(a.contract.read_text()); session=LearningSession(a.root,contract,runner=CampaignCoordinator(a.root,contract).run)
 planner=lambda: build_planner(contract.learner.get('planner_type','api'),output_dir=a.root,recipe_tag='arx-learning',env_name='arx',model=contract.learner.get('model','gpt-5.6-sol'),reasoning_effort=contract.learner.get('reasoning_effort','low'),max_tokens=contract.learner.get('max_tokens',4096),planner_timeout_s=contract.learner.get('timeout_s',120))
 session.planner=planner
 try:
  if a.interactive: session.interactive()
  else: print(json.dumps(session.automated(),indent=2))
 finally: session.tools.runner.__self__.close()
if __name__=='__main__':main()
