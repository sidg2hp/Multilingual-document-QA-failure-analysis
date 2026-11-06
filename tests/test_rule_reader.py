"""Tests for the deterministic field reader used by the OCR lower bound."""

from src.models.rule_reader import RuleBasedFieldReader


def test_reader_prefers_numeric_value_before_a_fax_label() -> None:
    reader = RuleBasedFieldReader()
    context = "614 -466 -5087 Fax: NOTE: Attorney General Betty D. Montgomery"
    assert reader.answer('What is the value of "Fax"?', context) == "614 -466 -5087"


def test_reader_takes_name_after_label() -> None:
    reader = RuleBasedFieldReader()
    context = "Name: John Doe Date: 01/02/2024"
    assert reader.answer('What is the value of "Name"?', context) == "John Doe"


def test_reader_abstains_when_label_is_absent() -> None:
    reader = RuleBasedFieldReader()
    assert (
        reader.answer('What is the value of "GSTIN"?', "Name: John Doe")
        == "UNANSWERABLE"
    )
