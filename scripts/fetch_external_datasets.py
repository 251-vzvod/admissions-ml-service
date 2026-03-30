"""Fetch external open datasets and optional Kaggle datasets for ML/NLP work."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "external"
RAW_ROOT = DATA_ROOT / "raw"


OPEN_SOURCES: list[dict[str, Any]] = [
    {
        "id": "clear_corpus",
        "name": "CLEAR Corpus",
        "license": "CC BY-NC-SA 4.0",
        "usage_notes": "Open readability corpus; use as essay/text robustness support, not as admissions labels.",
        "official_page": "https://github.com/scrosseye/CLEAR-Corpus",
        "acquisition": {
            "type": "http",
            "url": "https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx",
            "filename": "CLEAR_corpus_final.xlsx",
        },
    },
    {
        "id": "persuade_2_train",
        "name": "PERSUADE 2.0 Train",
        "license": "CC BY-NC-SA 4.0",
        "usage_notes": "Student argumentative essays with holistic scores and discourse annotations.",
        "official_page": "https://github.com/scrosseye/persuade_corpus_2.0",
        "acquisition": {
            "type": "gdrive",
            "file_id": "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135",
            "filename": "persuade_2_train.csv",
        },
    },
    {
        "id": "persuade_2_test",
        "name": "PERSUADE 2.0 Test",
        "license": "CC BY-NC-SA 4.0",
        "usage_notes": "Password-protected zip; password from official repo README is persuade_test.",
        "official_page": "https://github.com/scrosseye/persuade_corpus_2.0",
        "acquisition": {
            "type": "gdrive",
            "file_id": "1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1",
            "filename": "persuade_2_test.zip",
        },
    },
    {
        "id": "mediasum",
        "name": "MediaSum",
        "license": "Research-only",
        "usage_notes": "Use for interview/transcript structure support only; restrict usage to research/prototyping.",
        "official_page": "https://github.com/zcgzcgzcg1/MediaSum",
        "acquisition": {
            "type": "gdrive",
            "file_id": "1ZAKZM1cGhEw2A4_n4bGGMYyF8iPjLZni",
            "filename": "mediasum.zip",
        },
    },
    {
        "id": "ru_en_multilingual_support_placeholder",
        "name": "RU/EN support placeholder",
        "license": "N/A",
        "usage_notes": "Reserved slot for a future openly downloadable RU/EN support corpus. Current focus is PERSUADE, CLEAR, MediaSum, and your internal RU/EN candidate data.",
        "official_page": "https://the-learning-agency.com/guides-resources/datasets/",
        "acquisition": {
            "type": "metadata_only",
            "filename": "README.txt",
        },
    },
]


KAGGLE_SOURCES: list[dict[str, Any]] = [
    {
        "id": "learning_agency_asap_2",
        "name": "ASAP 2.0",
        "license": "CC BY 4.0",
        "official_page": "https://the-learning-agency.com/guides-resources/datasets/",
        "competition_slug": "automated-student-assessment-prize-2",
    },
    {
        "id": "learning_agency_aide",
        "name": "AIDE",
        "license": "CC BY 4.0",
        "official_page": "https://the-learning-agency.com/guides-resources/datasets/",
        "competition_slug": "llm-detect-ai-generated-text",
    },
    {
        "id": "learning_agency_piilo",
        "name": "PIILO",
        "license": "CC BY 4.0",
        "official_page": "https://the-learning-agency.com/guides-resources/datasets/",
        "competition_slug": "pii-detection-removal-from-educational-data",
    },
    {
        "id": "learning_agency_klicke",
        "name": "KLICKE",
        "license": "CC BY 4.0",
        "official_page": "https://the-learning-agency.com/guides-resources/datasets/",
        "competition_slug": "linking-writing-processes-to-writing-quality",
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_manifest() -> Path:
    ensure_dir(DATA_ROOT)
    manifest_path = DATA_ROOT / "dataset_manifest.json"
    manifest = {
        "open_sources": OPEN_SOURCES,
        "kaggle_sources": KAGGLE_SOURCES,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def download_http(url: str, destination: Path) -> None:
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)


def download_gdrive(file_id: str, destination: Path) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    download_http(url=url, destination=destination)


def fetch_open_sources(dataset_ids: set[str] | None = None) -> list[dict[str, Any]]:
    ensure_dir(RAW_ROOT)
    downloaded: list[dict[str, Any]] = []
    for source in OPEN_SOURCES:
        if dataset_ids and source["id"] not in dataset_ids:
            continue

        acquisition = source["acquisition"]
        target_dir = RAW_ROOT / source["id"]
        ensure_dir(target_dir)
        target_file = target_dir / acquisition["filename"]

        if target_file.exists() and target_file.stat().st_size > 0:
            pass
        elif acquisition["type"] == "metadata_only":
            target_file.write_text(source["usage_notes"], encoding="utf-8")
        elif acquisition["type"] == "http":
            download_http(acquisition["url"], target_file)
        elif acquisition["type"] == "gdrive":
            download_gdrive(acquisition["file_id"], target_file)
        else:
            raise ValueError(f"unsupported acquisition type: {acquisition['type']}")

        downloaded.append(
            {
                "id": source["id"],
                "path": str(target_file.relative_to(ROOT)),
                "bytes": target_file.stat().st_size,
                "license": source["license"],
                "official_page": source["official_page"],
            }
        )
    return downloaded


def fetch_kaggle_sources(dataset_ids: set[str] | None = None) -> list[dict[str, Any]]:
    ensure_dir(RAW_ROOT)
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        venv_candidate = Path(sys.executable).with_name("kaggle.exe")
        if venv_candidate.exists():
            kaggle_bin = str(venv_candidate)
    if not kaggle_bin:
        venv_candidate = Path(sys.executable).with_name("kaggle")
        if venv_candidate.exists():
            kaggle_bin = str(venv_candidate)
    if not kaggle_bin:
        raise RuntimeError("kaggle_cli_not_found")

    downloaded: list[dict[str, Any]] = []
    for source in KAGGLE_SOURCES:
        if dataset_ids and source["id"] not in dataset_ids:
            continue

        target_dir = RAW_ROOT / source["id"]
        ensure_dir(target_dir)
        if any(target_dir.iterdir()):
            downloaded.append(
                {
                    "id": source["id"],
                    "path": str(target_dir.relative_to(ROOT)),
                    "license": source["license"],
                    "official_page": source["official_page"],
                    "competition_slug": source["competition_slug"],
                }
            )
            continue
        try:
            subprocess.run(
                [
                    kaggle_bin,
                    "competitions",
                    "download",
                    "-c",
                    source["competition_slug"],
                    "-p",
                    str(target_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            continue
        downloaded.append(
            {
                "id": source["id"],
                "path": str(target_dir.relative_to(ROOT)),
                "license": source["license"],
                "official_page": source["official_page"],
                "competition_slug": source["competition_slug"],
            }
        )
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external datasets for ML/NLP work")
    parser.add_argument(
        "--mode",
        choices=["manifest", "open", "kaggle", "all"],
        default="manifest",
        help="What to fetch",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset ids",
    )
    args = parser.parse_args()

    manifest_path = write_manifest()
    selected_ids = set(args.datasets) if args.datasets else None

    report: dict[str, Any] = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "downloaded": [],
        "notes": [
            "Open-source downloads are fetched directly from official public sources when available.",
            "Kaggle datasets require a configured Kaggle CLI and valid credentials.",
            "Check dataset licenses before using any corpus beyond prototype/research scope.",
        ],
    }

    if args.mode in {"open", "all"}:
        report["downloaded"].extend(fetch_open_sources(dataset_ids=selected_ids))

    if args.mode in {"kaggle", "all"}:
        report["downloaded"].extend(fetch_kaggle_sources(dataset_ids=selected_ids))

    report_path = DATA_ROOT / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
