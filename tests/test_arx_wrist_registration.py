"""Camera registration must preserve rigid geometry and handedness."""
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.deployment.calibrate_arx_wrist_from_snapshots import fit_rigid


def test_scoped_registration_recovers_known_camera_frame():
    points=np.random.default_rng(42).uniform([-.05,-.05,.1],[.05,.05,.4],size=(40,3))
    rotation=Rotation.from_rotvec([.1,-.3,.2]).as_matrix()
    translation=np.array([.04,.03,.08])
    target=points@rotation.T+translation
    fitted,spread=fit_rigid(points,target)
    assert np.max(np.abs(fitted[:3,:3]-rotation)) < 1e-10
    assert np.max(np.abs(fitted[:3,3]-translation)) < 1e-10
    assert spread[-1] > .005


def test_mirrored_correspondences_cannot_become_a_reflected_camera_frame():
    points=np.random.default_rng(42).normal(size=(40,3))*.02
    target=points*[-1.,1.,1.]
    fitted,_=fit_rigid(points,target)
    assert np.linalg.det(fitted[:3,:3]) > .999999
    residual=np.linalg.norm(points@fitted[:3,:3].T+fitted[:3,3]-target,axis=1)
    assert np.sqrt(np.mean(residual**2)) > .008
