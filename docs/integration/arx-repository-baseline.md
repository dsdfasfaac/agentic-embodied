# ARX deployment repository baseline

The source checkout on aigc31 is `/home/dingxin/Agentic-Embodied`. The large
local experiment assets are stored under
`/mnt/100T/users/dingxin/Agentic-Embodied-data` and linked back into the
checkout at their original relative paths. The Git baseline starts from
`ef6eb940e4a530417b1426ba612ec9a66788af03` on the
`codex/arx-deployment-baseline-20261001` branch.

Shared-storage paths:

- `runs/`: generated campaigns, traces, videos, and compiled scene bundles.
- `scenes/`: source scene assets.
- `pickup_10_episode_arx_physics/`, `pickup_10_episode_scene_bundle/`, and
  `pickup_no_scale_arx_refined_10/`: experiment scene bundles.
- `cosmos-arx-cr-independent-session-20260922/` and the four matching top-level
  zip archives: archived experiment outputs.

These paths and local logs/environments are ignored by Git. Source code, tests,
small reference fixtures, configuration, and documentation are versioned. A new
checkout needs access to the shared storage; create symlinks with the same names
from the repository root if they are absent. Do not commit generated outputs or
private credentials.

Before a deployment commit, inspect `git status --short` and `git diff --check`.
Record the code commit and CandidateBundle/package SHA-256 with each run so an
experiment can be traced to its exact source and candidate inputs.

Migration check on 2026-10-01: `rsync -ain --delete` reported no file,
size, timestamp, or directory differences for the five large data directories.
SHA-256 matched for 135 selected files across those directories, including the
10 largest; the separate archived session and zip files passed full rsync
checksum comparison. The source paths are now symlinks into shared storage.
Git metadata (`.git/`) and the local Python environment (`.venv/`) remain in
the checkout.
