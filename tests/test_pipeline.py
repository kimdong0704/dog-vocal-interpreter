"""End-to-end check of Pipeline on the bundled test clip (needs the model weights and third_party/)."""

from __future__ import annotations

import numpy as np
import pytest

from woofdecoder.config import REPO_ROOT, load_config

CLIP = REPO_ROOT / "data" / "test" / "media_test.mp4"


def missing_files() -> list[str]:
    config = load_config()
    needed = [
        CLIP,
        REPO_ROOT / "third_party" / "PretrainedSED",
        REPO_ROOT / "third_party" / "unilm" / "beats",
        REPO_ROOT / config["branches"]["audio"]["heads_dir"],
        REPO_ROOT / config["branches"]["body"]["pose"],
        REPO_ROOT / config["branches"]["body"]["head"],
        REPO_ROOT / config["fusion"]["dir"],
    ]
    return [str(path) for path in needed if not path.exists()]


@pytest.fixture(scope="module")
def results():
    missing = missing_files()
    if missing:
        pytest.skip(f"model weights / third_party missing: {missing}")
    from woofdecoder.pipeline import Pipeline

    pipeline = Pipeline()
    return pipeline, pipeline.run(CLIP)


def test_every_segment_has_all_outputs(results):
    from woofdecoder.constants import BRANCH_NAMES

    _, segment_results = results
    assert segment_results, "the test clip should contain at least one dog vocalization"
    for result in segment_results:
        assert set(result.branches) == set(BRANCH_NAMES)
        for output in [*result.branches.values(), result.fused]:
            assert output.probs.shape == (5,)
            assert np.isclose(output.probs.sum(), 1.0)


def test_fused_equals_fusion_model(results):
    pipeline, segment_results = results
    for result in segment_results:
        expected = pipeline.fusion.predict(result.branches)
        assert np.array_equal(result.fused.probs, expected.probs)
        if result.fusion is not None:
            assert np.array_equal(result.fusion["probs"], expected.probs)
            explained = result.fusion["bias"] + sum(result.fusion["contributions"].values())
            assert np.allclose(explained, result.fusion["logits"])
