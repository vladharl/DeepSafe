"""Miru #2/#18: the video apps abstain instead of voting 0.5, and score like upstream (contract C11).

Runs in eval/envs/deepsafe/.venv (setup.sh), which has the upstream code; model and face detector are stubbed.
"""

import base64
import importlib.util
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "eval/envs/deepsafe/repo"
os.environ.setdefault("CEV_MODEL_CODE_DIR", str(ENV / "cev"))
os.environ.setdefault("FAKESTORMER_MODEL_CODE_DIR", str(ENV / "fakestormer"))
VIDEO = Path(__file__).resolve().parents[1] / "models/video"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"miru_{name}", VIDEO / name / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cev, fs = load("cross_efficient_vit"), load("fake_stormer")
logit = lambda p: math.log(p / (1 - p))  # noqa: E731


@pytest.fixture
def clip(tmp_path):
    """16 frames of 128x96 solid blue (BGR 200, 30, 30)."""
    path = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (128, 96))
    for _ in range(16):
        out.write(np.full((96, 128, 3), (200, 30, 30), np.uint8))
    out.release()
    return path


@pytest.fixture
def junk(tmp_path):
    path = tmp_path / "junk.mp4"
    path.write_bytes(b"not a video")
    return str(path)


class Model:
    """Stub classifier: records its input, returns fn(input)."""

    def __init__(self, fn):
        self.fn, self.inputs = fn, []

    def __call__(self, x):
        self.inputs.append(x)
        return self.fn(x)


class Detector:
    """Stub MTCNN: `boxes` per frame in half-size coordinates."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img, landmarks=False):
        if not self.boxes:
            return None, None
        return np.array(self.boxes, float), np.full(len(self.boxes), 0.99)


@pytest.fixture
def cev_stubs(monkeypatch):
    def install(boxes, fn=lambda x: torch.zeros(len(x), 1)):
        model = Model(fn)
        monkeypatch.setattr(cev, "ensure_model_loaded", lambda variant: None)
        monkeypatch.setattr(cev, "model_instance", model)
        monkeypatch.setattr(cev, "face_detector_mtcnn", Detector(boxes))
        monkeypatch.setattr(
            cev, "face_transform_pipeline", cev._create_face_transform(224)
        )
        return model

    return install


@pytest.fixture
def fs_stubs(monkeypatch):
    def install(face_frames, fn=lambda x: [{"cls": torch.zeros(len(x), 1)}]):
        model, calls = Model(fn), iter(range(10**6))
        crop = np.full((256, 256, 3), 120, np.uint8)
        monkeypatch.setattr(fs, "ensure_model_loaded", lambda: None)
        monkeypatch.setattr(fs, "model_instance", model)
        monkeypatch.setattr(
            fs,
            "largest_face",
            lambda frame: (
                (crop, [0, 0, 10, 10], 0.99) if next(calls) < face_frames else None
            ),
        )
        return model

    return install


def assert_abstains(result, reason):
    assert result["class"] == "abstain" and result["abstain_reason"] == reason
    assert result["probability"] is None and result["prediction"] is None


# --- #2 abstentions --------------------------------------------------------------------------------


def test_cev_no_faces_abstains(cev_stubs, clip):
    cev_stubs(boxes=[])
    result = cev.score_video(clip)
    assert_abstains(result, "no_faces")
    assert result["total_frames_sampled"] == 16 and result["total_faces_analyzed"] == 0


@pytest.mark.parametrize("app", ["cev", "fs"])
def test_undecodable_file_abstains_no_frames(app, cev_stubs, fs_stubs, junk):
    cev_stubs(boxes=[[10, 10, 40, 40]]), fs_stubs(face_frames=32)
    assert_abstains({"cev": cev, "fs": fs}[app].score_video(junk), "no_frames")


def test_cev_inference_error_abstains(cev_stubs, clip):
    def boom(x):
        raise RuntimeError("MPS out of memory")

    cev_stubs(boxes=[[10, 10, 40, 40]], fn=boom)
    assert_abstains(cev.score_video(clip), "model_unavailable")


@pytest.mark.parametrize("app", ["cev", "fs"])
def test_model_load_failure_abstains(app, monkeypatch, clip):
    def missing(*args):
        raise FileNotFoundError("weights")

    module = {"cev": cev, "fs": fs}[app]
    monkeypatch.setattr(module, "ensure_model_loaded", missing)
    assert_abstains(module.score_video(clip), "model_unavailable")


def test_fs_no_faces_abstains(fs_stubs, clip):
    fs_stubs(face_frames=0)
    assert_abstains(fs.score_video(clip), "no_faces")


def test_fs_three_face_frames_are_too_few(fs_stubs, clip):
    model = fs_stubs(face_frames=3)
    result = fs.score_video(clip)
    assert_abstains(result, "too_few_faces")
    assert result["total_faces_analyzed"] == 3 and not model.inputs


def test_fs_inference_error_abstains(fs_stubs, clip):
    fs_stubs(
        face_frames=8, fn=lambda x: x.dim()
    )  # the old wrapper's bug: output is a list of dicts
    assert_abstains(fs.score_video(clip), "model_unavailable")


@pytest.mark.parametrize("app", ["cev", "fs"])
def test_predict_returns_200_with_the_abstention(app, cev_stubs, fs_stubs):
    cev_stubs(boxes=[]), fs_stubs(face_frames=0)
    module = {"cev": cev, "fs": fs}[app]
    r = TestClient(module.app).post(
        "/predict", json={"video_data": base64.b64encode(b"junk").decode()}
    )
    assert r.status_code == 200
    body = r.json()
    assert (
        body["class"] == "abstain"
        and body["probability"] is None
        and body["abstain_reason"] == "no_frames"
    )
    assert "model" in body and "inference_time" in body


# --- #18 upstream preprocessing and aggregation -----------------------------------------------------


def test_cev_feeds_raw_bgr_0_255_face_crops(cev_stubs, clip):
    model = cev_stubs(boxes=[[10, 10, 40, 40]], fn=lambda x: torch.ones(len(x), 1))
    result = cev.score_video(clip)
    x = torch.cat(model.inputs)
    assert x.dtype == torch.float32 and tuple(x.shape) == (16, 3, 224, 224)
    assert x.max() > 150  # raw 0-255, no /255 or ImageNet normalisation
    assert (
        x[:, 0].mean() > 150 and x[:, 2].mean() < 80
    )  # BGR: channel 0 is the blue one
    assert (
        result["probability"] == pytest.approx(1 / (1 + math.exp(-1)))
        and result["total_faces_analyzed"] == 16
    )


def test_custom_video_round():
    assert cev.custom_video_round([0.3, 0.6, 0.9]) == 0.6
    assert cev.custom_video_round([0.2, 0.4]) == pytest.approx(0.3)


def test_cev_groups_faces_by_identity(cev_stubs, clip):
    # two faces per frame (largest first): identity 0 scores 0.4, identity 1 scores 0.9
    cev_stubs(
        boxes=[[10, 10, 40, 40], [50, 10, 60, 20]],
        fn=lambda x: torch.tensor([[logit(0.4)], [logit(0.9)]]).repeat(len(x) // 2, 1),
    )
    result = cev.score_video(clip)
    assert result["probability"] == pytest.approx(
        0.9
    )  # custom_video_round([0.4, 0.9]); a plain mean gives 0.65
    assert result["class"] == "fake" and len(result["per_face_scores"]) == 32


def test_fs_clips_of_four_and_sigmoid_cls(fs_stubs, clip):
    model = fs_stubs(
        face_frames=10, fn=lambda x: [{"cls": torch.tensor([[2.0], [0.0]]), "hm": None}]
    )
    result = fs.score_video(clip)
    assert tuple(model.inputs[0].shape) == (
        2,
        3,
        4,
        224,
        224,
    )  # [K, C, T, H, W]; 2 leftover frames dropped
    assert result["probability"] == pytest.approx((1 / (1 + math.exp(-2)) + 0.5) / 2)
    assert result["total_faces_analyzed"] == 8 and len(result["per_face_scores"]) == 8
