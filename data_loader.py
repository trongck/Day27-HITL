"""Module tải và tiền xử lý 200 mẫu dữ liệu khách hàng E-Commerce từ Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
from datasets import load_dataset


DATA_FILE = Path(__file__).parent / "ecommerce_customers_200.json"


def fetch_and_save_hf_data(limit: int = 200, output_path: Path = DATA_FILE) -> list[dict[str, Any]]:
    """Tải dữ liệu từ Hugging Face Hub, chuẩn hoá 200 mẫu và lưu vào file JSON."""
    try:
        dataset = load_dataset(
            "RAK05/e-commerce-churn-data",
            data_files="data_ecommerce_customer_churn.csv",
            split="train",
        )
        df = dataset.to_pandas()
    except Exception as exc:
        print(f"Warning: Failed to fetch from HF Hub ({exc}), generating fallback data...")
        df = _generate_fallback_dataframe(limit)

    customers: list[dict[str, Any]] = []
    
    for idx, row in df.head(limit).iterrows():
        cust_id = f"ECOMM_{10001 + idx}"
        tenure = float(row["Tenure"]) if pd.notnull(row.get("Tenure")) else 1.0
        category = str(row.get("PreferedOrderCat", "General"))
        satisfaction = int(row.get("SatisfactionScore", 3)) if pd.notnull(row.get("SatisfactionScore")) else 3
        complain = int(row.get("Complain", 0)) if pd.notnull(row.get("Complain")) else 0
        days_since_last_order = float(row.get("DaySinceLastOrder", 5.0)) if pd.notnull(row.get("DaySinceLastOrder")) else 5.0
        cashback = float(row.get("CashbackAmount", 150.0)) if pd.notnull(row.get("CashbackAmount")) else 150.0
        devices = int(row.get("NumberOfDeviceRegistered", 2)) if pd.notnull(row.get("NumberOfDeviceRegistered")) else 2
        churn_flag = int(row.get("Churn", 0)) if pd.notnull(row.get("Churn")) else 0

        # Tính TOI (Tier of Interest: High / Medium / Low)
        if tenure >= 12 or devices >= 4:
            toi = "high"
        elif tenure >= 4 or devices >= 3:
            toi = "medium"
        else:
            toi = "low"

        # Tính Churn Probability chuẩn hóa cho lab HITL (0.0 -> 1.0)
        if churn_flag == 1 and complain == 1:
            churn_prob = 0.88
        elif churn_flag == 1:
            churn_prob = 0.78
        elif complain == 1:
            churn_prob = 0.55
        elif days_since_last_order >= 12:
            churn_prob = 0.45
        elif satisfaction <= 2:
            churn_prob = 0.38
        else:
            churn_prob = 0.15

        # Thiết lập payload mặc định
        if churn_prob >= 0.75:
            # High risk action: Tăng hạn mức tín dụng mua trước trả sau (BNPL)
            action_payload = {
                "amount": int(min(cashback * 300_000, 100_000_000)) or 50_000_000,
                "currency": "VND",
                "category": category,
            }
        else:
            # Low risk action: Gửi email voucher / chăm sóc khách hàng
            action_payload = {
                "template": "retention_offer" if churn_prob >= 0.40 else "customer_care",
                "category": category,
                "discount_code": f"SAVE_{category.upper().replace(' ', '_')[:6]}",
            }

        customer_record = {
            "customer_id": cust_id,
            "category": category,
            "tenure_months": tenure,
            "satisfaction_score": satisfaction,
            "complain": complain,
            "day_since_last_order": days_since_last_order,
            "cashback_amount": cashback,
            "devices_registered": devices,
            "toi": toi,
            "churn_probability": churn_prob,
            "action_payload": action_payload,
        }
        customers.append(customer_record)

    output_path.write_text(json.dumps(customers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Successfully saved {len(customers)} records to {output_path.name}")
    return customers


def _generate_fallback_dataframe(limit: int) -> pd.DataFrame:
    """Tạo dataframe dự phòng nếu mạng offline."""
    import random
    categories = ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Home & Kitchen"]
    records = []
    for i in range(limit):
        records.append({
            "Tenure": random.choice([1.0, 3.0, 6.0, 12.0, 24.0, 36.0]),
            "PreferedOrderCat": random.choice(categories),
            "SatisfactionScore": random.choice([1, 2, 3, 4, 5]),
            "Complain": random.choice([0, 0, 0, 1]),
            "DaySinceLastOrder": random.choice([1.0, 3.0, 7.0, 15.0, 30.0]),
            "CashbackAmount": random.uniform(100.0, 350.0),
            "NumberOfDeviceRegistered": random.choice([1, 2, 3, 4, 5]),
            "Churn": random.choice([0, 0, 1]),
        })
    return pd.DataFrame(records)


def load_cached_customers(file_path: Path = DATA_FILE) -> list[dict[str, Any]]:
    """Đọc dữ liệu 200 khách hàng đã được lưu trữ sẵn."""
    if not file_path.exists():
        return fetch_and_save_hf_data(limit=200, output_path=file_path)
    
    content = file_path.read_text(encoding="utf-8")
    return json.loads(content)


if __name__ == "__main__":
    fetch_and_save_hf_data(limit=200)
