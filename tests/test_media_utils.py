import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.media_utils import collect_media_sources


def test_collect_media_sources_skips_missing_scalar_values():
    values = [
        "https://example.com/image.jpg",
        None,
        float("nan"),
        pd.NA,
        "",
        "   ",
    ]

    assert collect_media_sources(values) == ["https://example.com/image.jpg"]


def test_collect_media_sources_flattens_lists_and_preserves_order():
    values = [
        ["s3://bucket/one.jpg", None, float("nan")],
        "  s3://bucket/two.jpg  ",
        ("s3://bucket/three.jpg", ""),
    ]

    assert collect_media_sources(values) == [
        "s3://bucket/one.jpg",
        "s3://bucket/two.jpg",
        "s3://bucket/three.jpg",
    ]
