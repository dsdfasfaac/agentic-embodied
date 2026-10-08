import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.arx.deployment.wrist_self_mask import WristSelfMask


def reference(tmp_path):
    value = {"schema_version":"arx.wrist.empty-self.v1", "empty_home_observed":True,
             "camera_serial":"right-serial", "intrinsics_sha256":"intrinsics",
             "wrist_mount_sha256":"mount", "image_shape":[10,10],
             "right_gripper_feedback":-2.46, "gripper_feedback_tolerance":.08,
             "depth_tolerance_mm":4,
             "pixels_xy_depth_mm":[[x,y,100] for y in range(10) for x in range(10)]}
    path = tmp_path / "self.json"
    path.write_text(json.dumps(value))
    return WristSelfMask(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_reference_removes_only_measured_depth_at_calibrated_opening(tmp_path):
    ref = reference(tmp_path)
    rgb = np.zeros((10,10,3),np.uint8)
    depth = np.full((10,10),100,np.uint16)
    target = np.zeros((10,10),bool); target[1,1] = True
    rgb[2,2] = [64,43,0]  # dim yellow rack is protected
    rgb[3,3] = [0,30,90]  # neighbouring blue tube is protected
    depth[4,4] = 0
    depth[5,5] = 106
    h = {"camera_health":{"right_rgb":{"device_id":"right-serial"}},
         "camera_calibration_sha256":{"right_rgb":"intrinsics"},
         "measured_state":[0]*13+[-2.46]}
    p = SimpleNamespace(wrist_mount_sha="mount", wrist_intrinsics={},
                        _deproject_intrinsics=lambda x,y,z,i:np.array([x*.001,y*.001,z]))
    mask = ref.mask(rgb,depth,h,p,target)
    assert mask[0,0]
    assert all(not mask[i,i] for i in range(1,6))
    h["measured_state"][13] = -1.0
    assert not ref.mask(rgb,depth,h,p,target).any()
    p.wrist_mount_sha = "changed"
    with pytest.raises(ValueError,match="camera or mount"):
        ref.mask(rgb,depth,h,p,target)


def test_bad_source_digest_is_not_a_self_mask(tmp_path):
    path = tmp_path / "self.json"; path.write_text("{}")
    with pytest.raises(ValueError,match="SHA"):
        WristSelfMask(path,"0"*64)
