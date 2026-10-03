"""Registry: the list of every dataset the downloader can fetch

To add a dataset, add a Dataset entry to REGISTRY
**Nothing else needs to change**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote

from woofdecoder.data.download_config import VALID_KINDS


@dataclass(frozen=True)
class UrlFile:
    """One file to fetch over HTTP. expected_bytes and md5 are None when unknown."""

    url: str
    filename: str
    expected_bytes: int | None = None
    md5: str | None = None


@dataclass(frozen=True)
class Dataset:
    """One registry entry. Fill repo_id/revision/allow_patterns for "hf", files for "url"."""

    name: str
    description: str
    license: str
    license_url: str
    approx_size_gb: float
    kind: str
    repo_id: str | None = None
    revision: str | None = None  # None means the latest commit on main
    allow_patterns: list[str] | None = None  # None means every file
    files: list[UrlFile] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Fail early on a broken registry entry instead of mid-download."""
        if self.kind not in VALID_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if self.kind == "hf" and not self.repo_id:
            raise ValueError(f"{self.name}: an 'hf' dataset needs a repo_id")
        if self.kind == "url" and not self.files:
            raise ValueError(f"{self.name}: a 'url' dataset needs at least one file")


def zenodo_file(record_id: str, filename: str) -> UrlFile:
    """Build the download entry for one file of a Zenodo record.

    Args:
        record_id: Zenodo record number.
        filename: File name as shown on Zenodo (may contain spaces).

    Returns:
        A UrlFile whose URL has the name quoted.
    """
    quoted_name = quote(filename)
    url = f"https://zenodo.org/records/{record_id}/files/{quoted_name}?download=1"
    return UrlFile(url=url, filename=filename)


ZENODO_PLAYPANT_RECORD: str = "18972388"
STANFORD_DOGS_BASE: str = "http://vision.stanford.edu/aditya86/ImageNetDogs"

# "02 Raw recordings.zip" from PlayPanT is left out
# Large & Not needed
PLAYPANT_FILES: list[UrlFile] = [
    zenodo_file(ZENODO_PLAYPANT_RECORD, "03 Play session recordings.zip"),
    zenodo_file(ZENODO_PLAYPANT_RECORD, "04 Annotations.zip"),
    zenodo_file(ZENODO_PLAYPANT_RECORD, "05 Metadata.csv"),
    zenodo_file(ZENODO_PLAYPANT_RECORD, "06 Readme.txt"),
]

REGISTRY: list[Dataset] = [
    Dataset(
        name="playpant",
        description="PlayPanT: dog play-session audio recordings with annotations",
        license="CC-BY-NC-4.0",
        license_url=f"https://zenodo.org/records/{ZENODO_PLAYPANT_RECORD}",
        approx_size_gb=0.38,
        kind="url",
        files=PLAYPANT_FILES,
    ),
    Dataset(
        name="act_env",
        description="Barkopedia: dog barks labeled by activity and environment",
        license="MIT",
        license_url="https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env",
        approx_size_gb=0.61,
        kind="hf",
        repo_id="ArlingtonCL2/Barkopedia_Dog_Act_Env",
        revision="60a721a7f9477634971f6b3df29dc2a5e56994d8",
        allow_patterns=["train/*", "validation/*", "README.md"],
    ),
    Dataset(
        name="emotional_canines",
        description="Barkopedia: dog vocalizations labeled by emotion",
        license="CC-BY-NC-SA-4.0",
        license_url=(
            "https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data"
        ),
        approx_size_gb=0.07,
        kind="hf",
        repo_id="ArlingtonCL2/BarkopediaDogEmotionClassification_Data",
        revision="c0617b15562882b9eee4cc1bcb8e6aa80e51bd34",
    ),
    Dataset(
        name="audioset_strong_eval",
        description="AudioSet strong labels (animal sounds), eval split mirror",
        license="labels CC-BY-4.0 / mirror unlicensed",
        license_url="https://huggingface.co/datasets/enyoukai/AudioSet-Strong-Animals",
        approx_size_gb=3.0,
        kind="hf",
        repo_id="enyoukai/AudioSet-Strong-Animals",
        allow_patterns=["data/eval-*.parquet", "README.md"],
    ),
    Dataset(
        name="dog_pose",
        description="Ultralytics Dog-Pose: dog images with keypoints (YOLO format)",
        license="AGPL-3.0 zip / MIT upstream",
        license_url="https://docs.ultralytics.com/datasets/pose/dog-pose/",
        approx_size_gb=0.35,
        kind="url",
        files=[
            UrlFile(
                url="https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip",
                filename="dog-pose.zip",
                expected_bytes=353152125,
                md5="7169e61b3c2d267c6105446f391d16cd",
            ),
        ],
    ),
    Dataset(
        name="dogposecv",
        description="Dog pose classification images (standing, sitting, lying, ...)",
        license="Apache-2.0 labels / ImageNet NC images",
        license_url="https://huggingface.co/datasets/stockeh/dog-pose-cv",
        approx_size_gb=0.82,
        kind="hf",
        repo_id="stockeh/dog-pose-cv",
    ),
    Dataset(
        name="stanford_dogs",
        description="Stanford Dogs: 120 breeds, images and train/test lists",
        license="ImageNet terms, non-commercial",
        license_url="http://vision.stanford.edu/aditya86/ImageNetDogs/",
        approx_size_gb=0.76,
        kind="url",
        files=[
            UrlFile(url=f"{STANFORD_DOGS_BASE}/images.tar", filename="images.tar"),
            UrlFile(url=f"{STANFORD_DOGS_BASE}/lists.tar", filename="lists.tar"),
        ],
    ),
]


def select_datasets(names: list[str] | None) -> list[Dataset]:
    """Pick registry entries by name.

    Args:
        names: Dataset names to keep, or None for the whole registry.

    Returns:
        The matching datasets, in the order given.
    """
    if names is None:
        return list(REGISTRY)

    by_name: dict[str, Dataset] = {}
    for dataset in REGISTRY:
        by_name[dataset.name] = dataset

    selected: list[Dataset] = []
    for name in names:
        if name not in by_name:
            known = ", ".join(by_name)
            raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}")
        selected.append(by_name[name])
    return selected
