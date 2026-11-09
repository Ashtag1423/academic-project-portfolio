"""
Mirror-accelerated dataset downloads (assignment datasets + Python deps).

Notes:
- Tsinghua TUNA (https://mirrors.tuna.tsinghua.edu.cn): PyPI / uv mirrors.
- VOC 2012: Baidu Cloud BOS mirror (dataset.bj.bcebos.com).
- Oxford-IIIT Pet download order:
  1. Local tar extract (if MD5 valid)
  2. OpenDataLab (requires `openxlab login`, China CDN)
  3. HuggingFace hf-mirror (Parquet, no login)
  4. Oxford official HTTP / wget resume
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional


def _extract_archive(archive: str | Path, extract_root: str | Path) -> None:
    archive, extract_root = Path(archive), Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    # --no-same-owner: avoid uid warnings on WSL/containers breaking tar exit code
    if shutil.which("tar"):
        flag = "-xzf" if str(archive).endswith((".tar.gz", ".tgz")) else "-xf"
        subprocess.run(
            ["tar", flag, str(archive), "-C", str(extract_root), "--no-same-owner"],
            check=True,
        )
        return
    if str(archive).endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=extract_root)
    elif str(archive).endswith(".tar"):
        with tarfile.open(archive, "r") as tar:
            tar.extractall(path=extract_root)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def _download_and_extract_archive(
    url: str, download_root: str | Path, filename: str, md5: Optional[str] = None
) -> None:
    from torchvision.datasets.utils import download_and_extract_archive

    download_and_extract_archive(url, str(download_root), filename=filename, md5=md5)


# ----- Mirror endpoints -----
TUNA_PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# Pascal VOC 2012 (MD5 matches torchvision)
VOC2012_FILENAME = "VOCtrainval_11-May-2012.tar"
VOC2012_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
VOC2012_URL = "https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar"

# Oxford-IIIT Pet (MD5 matches torchvision)
OPENXLAB_PET_REPO = "OpenDataLab/Oxford-IIIT_Pets"
HF_PET_DATASET = "timm/oxford-iiit-pet"
HF_PET_PARQUET = {
    "train": f"{HF_MIRROR_ENDPOINT}/datasets/{HF_PET_DATASET}/resolve/main/data/train-00000-of-00001.parquet",
    "test": f"{HF_MIRROR_ENDPOINT}/datasets/{HF_PET_DATASET}/resolve/main/data/test-00000-of-00001.parquet",
}
OXFORD_PET_RESOURCES = (
    (
        "images.tar.gz",
        "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        [
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
        ],
    ),
    (
        "annotations.tar.gz",
        "95a8c909bbe2e81eed6a22bccdf3f68f",
        [
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
        ],
    ),
)


def setup_china_mirrors(use_hf_mirror: bool = True) -> None:
    """Set env vars: Tsinghua PyPI mirror + optional HuggingFace mirror."""
    os.environ.setdefault("PIP_INDEX_URL", TUNA_PYPI_INDEX)
    os.environ.setdefault("UV_DEFAULT_INDEX", TUNA_PYPI_INDEX)
    if use_hf_mirror:
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    print("Configured mirrors:")
    print(f"  PyPI (Tsinghua TUNA): {TUNA_PYPI_INDEX}")
    if use_hf_mirror:
        print(f"  HuggingFace:          {HF_MIRROR_ENDPOINT}")


def _wget_resume(url: str, dest: Path, desc: str = "dataset") -> None:
    """Download a large file with wget resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "--tries=0", "--timeout=30", "-O", str(dest), url]
    print(f"[{desc}] wget resume: {url}")
    subprocess.run(cmd, check=True)


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    try:
        md5 = hashlib.md5(usedforsecurity=False)
    except TypeError:  # Python < 3.9
        md5 = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def _download_from_mirrors(
    urls: Iterable[str],
    dest: Path,
    expected_md5: Optional[str] = None,
    desc: str = "dataset",
) -> None:
    """Try mirror URLs until download succeeds and MD5 matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_md5 and _md5_file(dest) == expected_md5:
        print(f"[{desc}] Already present and verified: {dest}")
        return

    errors = []
    for i, url in enumerate(urls, start=1):
        print(f"[{desc}] Trying mirror {i}/{len(urls)}: {url}")
        try:
            from torchvision.datasets.utils import download_url

            download_url(url, str(dest.parent), filename=dest.name, md5=None)
            if not dest.is_file():
                raise FileNotFoundError(f"File missing after download: {dest}")
            if expected_md5 and _md5_file(dest) != expected_md5:
                raise ValueError(
                    f"MD5 mismatch (expected {expected_md5}, got {_md5_file(dest)})"
                )
            print(f"[{desc}] Download complete: {dest}")
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"  - {url}\n    {type(exc).__name__}: {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)

    raise RuntimeError(f"[{desc}] All mirrors failed:\n" + "\n".join(errors))


def _pet_is_ready(base: Path) -> bool:
    """Minimum directory layout required by torchvision OxfordIIITPet."""
    images = base / "images"
    anns = base / "annotations"
    if not images.is_dir() or not anns.is_dir():
        return False
    if not (anns / "trainval.txt").is_file() or not (anns / "test.txt").is_file():
        return False
    return len(list(images.glob("*.jpg"))) >= 1000


def print_mirror_plan(dataset: str = "all", root: str | Path = ".") -> None:
    """Print dataset download sources and local cache / checkpoint status."""
    root = Path(root)
    bar = "=" * 62
    print(bar)
    print("Dataset download plan (a3_dataset_mirrors.py; not torchvision download=True)")
    print(bar)

    if dataset in ("all", "voc"):
        voc_data_root = root / "data"
        voc_root = voc_data_root / "VOCdevkit" / "VOC2012"
        voc_tar = voc_data_root / VOC2012_FILENAME
        print("\n[Pascal VOC 2012]")
        if voc_root.is_dir():
            print(f"  Status: ready, skip download -> {voc_root}")
        else:
            print(f"  Status: missing {voc_root}, will download")
            if voc_tar.is_file():
                print(f"  Local archive: {voc_tar} (verify MD5, then extract)")
            print(f"  Download URL (Baidu BOS):\n    {VOC2012_URL}")

    if dataset in ("all", "pet"):
        pet_base = root / "oxford-iiit-pet"
        print("\n[Oxford-IIIT Pet]")
        if _pet_is_ready(pet_base):
            n = len(list((pet_base / "images").glob("*.jpg")))
            print(f"  Status: ready ({n} images), skip download -> {pet_base}")
        else:
            print("  Status: not ready, will try sources in order:")
            img_tar = pet_base / "images.tar.gz"
            if img_tar.is_file():
                print(f"  Local images.tar.gz: {img_tar} (verify MD5, then extract)")
            print("  Source order:")
            print("    1. Local tar extract (if MD5 valid)")
            print("    2. OpenDataLab (run: openxlab login)")
            print(f"       Repo: {OPENXLAB_PET_REPO}")
            print("    3. HuggingFace hf-mirror (Parquet)")
            print(f"       train: {HF_PET_PARQUET['train']}")
            print("    4. Oxford official + wget (fallback only)")

    ckpt = root / "checkpoints"
    if ckpt.is_dir() and dataset == "all":
        print("\n[Training checkpoints]")
        for name in (
            "baseline_fcn.pth",
            "fcn_multiscale.pth",
            "vit_full_finetune.pth",
            "vit_partial_finetune.pth",
            "vit_lora_peft.pth",
        ):
            p = ckpt / name
            mark = "found -> notebook will skip training" if p.is_file() else "missing -> will train"
            print(f"  {name}: {mark}")
    print(bar)


def _purge_invalid_pet_archives(base: Path) -> None:
    for filename, md5, _ in OXFORD_PET_RESOURCES:
        archive = base / filename
        if not archive.is_file():
            continue
        if archive.stat().st_size == 0 or _md5_file(archive) != md5:
            print(f"[Oxford Pet] Removing invalid cache {archive}")
            archive.unlink(missing_ok=True)


def _extract_pet_archives(base: Path) -> None:
    for filename, md5, _ in OXFORD_PET_RESOURCES:
        archive = base / filename
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archive: {archive}")
        if _md5_file(archive) != md5:
            archive.unlink(missing_ok=True)
            raise ValueError(f"{archive} MD5 check failed, removed")
        print(f"[Oxford Pet] Extracting: {archive}")
        _extract_archive(archive, base)


def _download_pet_opendatalab(base: Path) -> bool:
    """OpenDataLab China CDN (requires openxlab login)."""
    try:
        from openxlab.dataset import download as odl_download
    except ImportError:
        print("[Oxford Pet] OpenDataLab: openxlab not installed (pip install openxlab)")
        return False

    ok = True
    for source_path, md5 in (
        ("/raw/images.tar.gz", OXFORD_PET_RESOURCES[0][1]),
        ("/raw/annotations.tar.gz", OXFORD_PET_RESOURCES[1][1]),
    ):
        fname = Path(source_path).name
        archive = base / fname
        if archive.is_file() and _md5_file(archive) == md5:
            print(f"[Oxford Pet] OpenDataLab already have: {archive}")
            continue
        print(f"[Oxford Pet] OpenDataLab downloading: {source_path}")
        try:
            odl_download(
                dataset_repo=OPENXLAB_PET_REPO,
                source_path=source_path,
                target_path=str(base),
            )
        except SystemExit:
            print("[Oxford Pet] OpenDataLab: not logged in (run: openxlab login)")
            ok = False
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[Oxford Pet] OpenDataLab failed ({fname}): {exc}")
            print("  Hint: run openxlab login and configure AK/SK")
            print("  Docs: https://openxlab.org.cn/docs/developers/auth.html")
            ok = False
            break
        if not archive.is_file() or _md5_file(archive) != md5:
            print(f"[Oxford Pet] OpenDataLab {fname} MD5 check failed")
            archive.unlink(missing_ok=True)
            ok = False
            break

    if not ok:
        return False
    _extract_pet_archives(base)
    return _pet_is_ready(base)


def _export_hf_split(
    hf_split: str,
    split_txt: str,
    images_dir: Path,
    anns_dir: Path,
    dataset,
    images_only_txt: bool = False,
) -> None:
    """Export HF split to torchvision layout: images/*.jpg and annotations/*.txt."""
    images_dir.mkdir(parents=True, exist_ok=True)
    anns_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    subset = dataset[hf_split]
    n = len(subset)
    print(f"[Oxford Pet] HuggingFace export {hf_split} ({n} rows) -> {split_txt}")

    for i, row in enumerate(subset):
        image_id = row["image_id"]
        if not images_only_txt:
            img_path = images_dir / f"{image_id}.jpg"
            if not img_path.is_file():
                row["image"].save(img_path)
        class_id = int(row["label"]) + 1
        species = 1 if int(row["label_cat_dog"]) == 0 else 2
        lines.append(f"{image_id} {class_id} {species} {class_id}")
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  ... {i + 1}/{n}")

    (anns_dir / split_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_pet_split_txt_from_hf(base: Path) -> bool:
    """Create trainval.txt / test.txt only (images already extracted locally)."""
    setup_china_mirrors(use_hf_mirror=True)
    try:
        from datasets import load_dataset
    except ImportError:
        return False

    images_dir = base / "images"
    anns_dir = base / "annotations"
    if len(list(images_dir.glob("*.jpg"))) < 1000:
        return False

    try:
        ds = load_dataset("parquet", data_files=dict(HF_PET_PARQUET.items()))
    except Exception as exc:  # noqa: BLE001
        print(f"[Oxford Pet] HF metadata load failed: {exc}")
        return False

    _export_hf_split("train", "trainval.txt", images_dir, anns_dir, ds, images_only_txt=True)
    _export_hf_split("test", "test.txt", images_dir, anns_dir, ds, images_only_txt=True)
    return _pet_is_ready(base)


def _download_pet_huggingface(base: Path) -> bool:
    """Fetch Parquet via hf-mirror and convert to torchvision directory layout."""
    setup_china_mirrors(use_hf_mirror=True)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Oxford Pet] HuggingFace: datasets not installed (pip install datasets)")
        return False

    images_dir = base / "images"
    anns_dir = base / "annotations"
    if _pet_is_ready(base):
        return True

    try:
        print("[Oxford Pet] HuggingFace: loading Parquet from hf-mirror ...")
        ds = load_dataset(
            "parquet",
            data_files={
                "train": HF_PET_PARQUET["train"],
                "test": HF_PET_PARQUET["test"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Oxford Pet] HuggingFace Parquet direct URL failed: {exc}")
        try:
            print(f"[Oxford Pet] HuggingFace: fallback load_dataset('{HF_PET_DATASET}') ...")
            ds = load_dataset(HF_PET_DATASET)
        except Exception as exc2:  # noqa: BLE001
            print(f"[Oxford Pet] HuggingFace dataset load failed: {exc2}")
            return False

    _export_hf_split("train", "trainval.txt", images_dir, anns_dir, ds)
    _export_hf_split("test", "test.txt", images_dir, anns_dir, ds)
    return _pet_is_ready(base)


def _download_pet_official(base: Path) -> bool:
    """Oxford official HTTP + wget / torchvision fallback."""
    base.mkdir(parents=True, exist_ok=True)
    _purge_invalid_pet_archives(base)

    for filename, md5, urls in OXFORD_PET_RESOURCES:
        archive = base / filename
        if archive.is_file() and _md5_file(archive) == md5:
            print(f"[Oxford Pet] Using existing: {archive}")
            continue

        last_err: Optional[Exception] = None
        for url in urls:
            try:
                _wget_resume(url, archive, desc=f"Oxford Pet {filename}")
                if _md5_file(archive) == md5:
                    last_err = None
                    break
                archive.unlink(missing_ok=True)
                last_err = ValueError(f"MD5 mismatch: {url}")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if archive.exists():
                    archive.unlink(missing_ok=True)

        if last_err is not None:
            print(f"[Oxford Pet] wget failed, trying torchvision: {filename}")
            try:
                _download_and_extract_archive(
                    urls[0], download_root=str(base), filename=filename, md5=md5
                )
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[Oxford Pet] Official source failed {filename}: {exc}")
                return False

        if not archive.is_file() or _md5_file(archive) != md5:
            return False

    _extract_pet_archives(base)
    return _pet_is_ready(base)


def _finalize_local_pet_tars(base: Path) -> bool:
    """If user placed tar.gz manually, extract and fill missing split annotation files."""
    images_dir = base / "images"
    anns_dir = base / "annotations"
    img_archive = base / "images.tar.gz"
    ann_archive = base / "annotations.tar.gz"

    if img_archive.is_file() and _md5_file(img_archive) == OXFORD_PET_RESOURCES[0][1]:
        if not images_dir.is_dir() or len(list(images_dir.glob("*.jpg"))) < 1000:
            print(f"[Oxford Pet] Extracting local {img_archive.name}")
            _extract_archive(img_archive, base)

    if ann_archive.is_file() and _md5_file(ann_archive) == OXFORD_PET_RESOURCES[1][1]:
        if not (anns_dir / "trainval.txt").is_file():
            print(f"[Oxford Pet] Extracting local {ann_archive.name}")
            _extract_archive(ann_archive, base)

    if _pet_is_ready(base):
        return True

    if images_dir.is_dir() and len(list(images_dir.glob("*.jpg"))) >= 1000:
        if not (anns_dir / "trainval.txt").is_file():
            print("[Oxford Pet] Generating trainval.txt / test.txt from HuggingFace metadata ...")
            if _ensure_pet_split_txt_from_hf(base):
                return True
    return _pet_is_ready(base)


def download_oxford_iiit_pet(root: str | Path = "./") -> Path:
    """
    Download Oxford-IIIT Pet via mirrors; compatible with torchvision OxfordIIITPet.

    Order: local tar -> OpenDataLab -> HuggingFace (hf-mirror) -> Oxford official.
    """
    root = Path(root)
    base = root / "oxford-iiit-pet"
    print_mirror_plan("pet", root=root)

    if _pet_is_ready(base):
        print(f"Oxford-IIIT Pet already exists: {base}")
        return base

    base.mkdir(parents=True, exist_ok=True)
    _purge_invalid_pet_archives(base)

    if _finalize_local_pet_tars(base):
        print(f"Oxford-IIIT Pet ready (local tar): {base}")
        return base

    sources: list[tuple[str, Callable[[Path], bool]]] = [
        ("OpenDataLab", _download_pet_opendatalab),
        ("HuggingFace (hf-mirror)", _download_pet_huggingface),
        ("Oxford official", _download_pet_official),
    ]

    errors: list[str] = []
    for name, fn in sources:
        print(f"\n[Oxford Pet] ===== Trying {name} =====")
        try:
            if fn(base) and _pet_is_ready(base):
                print(f"Oxford-IIIT Pet ready ({name}): {base}")
                return base
            errors.append(f"{name}: download finished but layout check failed")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Oxford-IIIT Pet: all download sources failed.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nSuggestions:\n"
        "  1. OpenDataLab: pip install openxlab && openxlab login\n"
        "     https://opendatalab.com/OpenDataLab/Oxford-IIIT_Pets/download\n"
        "  2. Or place images.tar.gz + annotations.tar.gz under oxford-iiit-pet/\n"
        "  3. Then: tar -xzf images.tar.gz -C oxford-iiit-pet && tar -xzf annotations.tar.gz -C oxford-iiit-pet"
    )


def download_voc2012(root: str | Path = "./data") -> Path:
    """Download and extract Pascal VOC 2012 via Baidu BOS mirror."""
    root = Path(root)
    voc_root = root / "VOCdevkit" / "VOC2012"
    project_root = root.parent if root.name == "data" else root
    print_mirror_plan("voc", root=project_root)
    if voc_root.is_dir():
        print(f"VOC 2012 already exists: {voc_root}")
        return voc_root

    archive = root / VOC2012_FILENAME
    _download_from_mirrors([VOC2012_URL], archive, VOC2012_MD5, desc="Pascal VOC 2012")

    print(f"Extracting {archive} ...")
    with tarfile.open(archive, "r") as tar:
        tar.extractall(path=root)

    if not voc_root.is_dir():
        raise RuntimeError(f"Missing after extract: {voc_root}")
    print(f"VOC 2012 ready: {voc_root}")
    return voc_root


def load_voc_segmentation(root: str, image_set: str, year: str = "2012"):
    """Mirror download, then VOCSegmentation(download=False)."""
    from torchvision.datasets import VOCSegmentation

    download_voc2012(root)
    return VOCSegmentation(root=root, year=year, image_set=image_set, download=False)


def load_oxford_pet(root: str, split: str, transform=None):
    """Mirror download, then OxfordIIITPet(download=False)."""
    from torchvision.datasets import OxfordIIITPet

    download_oxford_iiit_pet(root)
    return OxfordIIITPet(root=root, split=split, download=False, transform=transform)
