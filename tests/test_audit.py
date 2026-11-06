"""Tests for the dependency-light provenance audit parser."""

from scripts.audit_datasets import extract_dataset_info


def test_extract_dataset_info_selects_config_and_schema() -> None:
    payload = {
        "sha": "abc123",
        "gated": False,
        "lastModified": "2026-01-01T00:00:00Z",
        "cardData": {
            "dataset_info": [
                {
                    "config_name": "ocr",
                    "splits": [{"name": "test", "num_examples": 876}],
                    "features": [
                        {"name": "id"},
                        {"name": "image"},
                        {"name": "text"},
                        {"name": "language"},
                    ],
                },
                {
                    "config_name": "vqa_indic",
                    "splits": [{"name": "test", "num_examples": 1007}],
                    "features": [{"name": "short_q1"}],
                },
            ]
        },
    }
    info = extract_dataset_info(payload, "ocr")
    assert info["found"] is True
    assert info["splits"]["test"] == 876
    assert info["features"] == ["id", "image", "text", "language"]
    assert info["revision"] == "abc123"

    missing = extract_dataset_info(payload, "does_not_exist")
    assert missing["found"] is False


def test_extract_dataset_info_handles_single_info_dict() -> None:
    payload = {
        "sha": "def",
        "cardData": {
            "dataset_info": {
                "config_name": "default",
                "splits": [{"name": "test", "num_examples": 50}],
                "features": [{"name": "words"}],
            }
        },
    }
    info = extract_dataset_info(payload, None)
    assert info["found"] is True
    assert info["splits"]["test"] == 50
