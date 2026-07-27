import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable so `utils.*` / `trainers.*` resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import video_utils


# --------------------------- _sample_kwargs -------------------------------

def test_sample_kwargs_prefers_fps_over_num_frames():
    assert video_utils._sample_kwargs(num_frames=8, fps=2.0) == {"fps": 2.0}


def test_sample_kwargs_falls_back_to_num_frames():
    assert video_utils._sample_kwargs(num_frames=8, fps=None) == {"num_frames": 8}


def test_sample_kwargs_empty_when_neither_set():
    assert video_utils._sample_kwargs(num_frames=None, fps=None) == {}


# ------------------------------ fetch_video -------------------------------

def test_fetch_video_caches_raw_file_and_decodes_once(tmp_path, monkeypatch):
    """A URL is downloaded once, cached on disk (md5 path), and re-served from
    the cache on the second call without re-downloading."""
    downloads = {"n": 0}

    def fake_fetch_url(url, timeout, retries):
        downloads["n"] += 1
        return b"FAKE_VIDEO_BYTES"

    decoded_paths = []

    def fake_decode(path, *, num_frames, fps, backend):
        decoded_paths.append(path)
        return f"frames::{num_frames}"

    monkeypatch.setattr(video_utils, "_fetch_url", fake_fetch_url)
    monkeypatch.setattr(video_utils, "_decode", fake_decode)

    url = "https://example.com/clip.mp4"
    cache_dir = str(tmp_path / "vcache")

    out1 = video_utils.fetch_video(url, num_frames=4, cache_dir=cache_dir)
    out2 = video_utils.fetch_video(url, num_frames=4, cache_dir=cache_dir)

    assert out1 == "frames::4" and out2 == "frames::4"
    # Downloaded exactly once; second call hits the disk cache.
    assert downloads["n"] == 1
    # Exactly one cached file was written and both decodes read that path.
    cached = os.listdir(cache_dir)
    assert len(cached) == 1
    assert decoded_paths[0] == decoded_paths[1] == os.path.join(cache_dir, cached[0])


def test_fetch_video_local_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video_utils.fetch_video(str(tmp_path / "nope.mp4"))


def test_fetch_video_dict_dispatches_on_url_key(monkeypatch):
    monkeypatch.setattr(video_utils, "_fetch_url", lambda url, t, r: b"X")
    monkeypatch.setattr(video_utils, "_decode", lambda p, **k: "ok")
    assert video_utils.fetch_video({"url": "https://e/x.mp4"}, num_frames=2) == "ok"


def test_fetch_video_rejects_unknown_type():
    with pytest.raises(TypeError):
        video_utils.fetch_video(12345)


# --------------------------- collator behavior ----------------------------

def test_video_injecting_proxy_only_injects_on_image_calls():
    from trainers.vlm_collator import _VideoInjectingProcessor

    seen = []

    class FakeProc:
        def __call__(self, *args, **kwargs):
            seen.append(kwargs)
            return "out"

    videos = [["frames"]]
    proxy = _VideoInjectingProcessor(FakeProc(), videos)

    # Prompt/LM call carries images -> videos injected.
    proxy(images=None, text=["t"])
    # Completion call carries only text -> videos NOT injected.
    proxy(text=["c"])

    assert seen[0].get("videos") == videos
    assert "videos" not in seen[1]


def test_extract_videos_none_when_empty_else_list():
    from trainers.vlm_collator import VideoAwareVLMCollator

    assert VideoAwareVLMCollator._extract_videos([{"images": []}, {"videos": []}]) is None
    got = VideoAwareVLMCollator._extract_videos([{"videos": ["f"]}, {"videos": []}])
    assert got == [["f"], []]
