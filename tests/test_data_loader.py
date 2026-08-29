"""Unit tests for the e-commerce data loader module."""

import json
from pathlib import Path
import pytest
from data_loader import (
    DATA_FILE,
    _generate_fallback_dataframe,
    fetch_and_save_hf_data,
    load_cached_customers,
)


def test_load_cached_customers_existing():
    customers = load_cached_customers(DATA_FILE)
    assert len(customers) >= 50
    first = customers[0]
    assert "customer_id" in first
    assert "toi" in first
    assert "churn_probability" in first
    assert "action_payload" in first
    assert first["toi"] in ["high", "medium", "low"]
    assert 0.0 <= first["churn_probability"] <= 1.0


def test_generate_fallback_dataframe():
    df = _generate_fallback_dataframe(limit=10)
    assert len(df) == 10
    assert "PreferedOrderCat" in df.columns
    assert "Churn" in df.columns


def test_fetch_and_save_fallback(tmp_path: Path):
    out_file = tmp_path / "test_customers.json"
    customers = fetch_and_save_hf_data(limit=15, output_path=out_file)
    assert len(customers) == 15
    assert out_file.exists()
    
    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 15
