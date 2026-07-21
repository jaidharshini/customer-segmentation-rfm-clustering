"""
generate_data.py
-----------------
Generates a realistic synthetic e-commerce transaction dataset that mirrors
the schema of the popular 'Online Retail II' (UCI) dataset:

    Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

Why synthetic data?
This makes the project 100% self-contained and reproducible — no manual
Kaggle/UCI download required to run the notebook. The generation logic
deliberately builds in different customer behavior archetypes (loyal,
new, at-risk, one-time, bargain hunters) with different Recency/Frequency/
Monetary patterns, so the clustering step has genuine structure to discover
(exactly like a real retail dataset would).

If you want to use the REAL Online Retail II dataset instead:
1. Download it from https://archive.ics.uci.edu/dataset/502/online+retail+ii
   or Kaggle ("Online Retail II UCI").
2. Save it as data/online_retail_II.xlsx
3. In the notebook, skip this generator and load that file directly with
   pd.read_excel() instead of pd.read_csv() on the synthetic file.
"""

import numpy as np
import pandas as pd
import datetime as dt
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 4000
END_DATE = dt.datetime(2024, 12, 31)
START_DATE = END_DATE - dt.timedelta(days=365)

COUNTRIES = ["United Kingdom", "Germany", "France", "Ireland", "Spain",
             "Netherlands", "Belgium", "Portugal", "Italy", "Australia"]
COUNTRY_WEIGHTS = [0.55, 0.09, 0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02]

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("47566", "PARTY BUNTING", 4.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("21212", "PACK OF 72 RETRO SPOT CAKE CASES", 0.55),
    ("23298", "SPACEBOY LUNCH BOX", 1.95),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("21977", "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22197", "SMALL POPCORN HOLDER", 0.85),
    ("23203", "JUMBO BAG DOILEY PATTERNS", 2.08),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
]

# ---- Customer archetypes: (share of base, freq_range(orders/yr), recency_bias_days, spend_per_order) ----
ARCHETYPES = {
    "loyal_highvalue":  dict(share=0.12, orders=(15, 30), recency_max=20,  spend=(60, 220)),
    "potential_loyal":  dict(share=0.22, orders=(6, 14),  recency_max=60,  spend=(25, 90)),
    "new_customer":     dict(share=0.18, orders=(1, 3),   recency_max=30,  spend=(15, 70)),
    "at_risk":          dict(share=0.20, orders=(5, 15),  recency_max=365, spend=(20, 80), recency_min=120),
    "bargain_hunter":   dict(share=0.16, orders=(4, 10),  recency_max=90,  spend=(5, 25)),
    "one_time_lapsed":  dict(share=0.12, orders=(1, 1),   recency_max=365, spend=(10, 60), recency_min=180),
}


def random_date_within(recency_min_days, recency_max_days):
    """Return a purchase date that is between recency_min and recency_max days before END_DATE."""
    days_ago = random.randint(recency_min_days, recency_max_days)
    return END_DATE - dt.timedelta(days=days_ago)


def generate():
    rows = []
    customer_id = 12000
    invoice_no = 536000

    customers = []
    for archetype, cfg in ARCHETYPES.items():
        n = int(N_CUSTOMERS * cfg["share"])
        customers += [archetype] * n

    random.shuffle(customers)

    for archetype in customers:
        cfg = ARCHETYPES[archetype]
        customer_id += 1
        country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]
        n_orders = random.randint(*cfg["orders"])
        recency_min = cfg.get("recency_min", 0)
        recency_max = cfg["recency_max"]

        for _ in range(n_orders):
            invoice_no += 1
            order_date = random_date_within(recency_min, recency_max)
            # jitter order date across the year for frequency spread (except most-recent order)
            n_items = random.randint(1, 5)
            target_spend = random.uniform(*cfg["spend"])

            for _ in range(n_items):
                code, desc, base_price = random.choice(PRODUCTS)
                price = round(base_price * random.uniform(0.9, 1.1), 2)
                qty = max(1, int(round((target_spend / n_items) / price)))
                rows.append({
                    "Invoice": str(invoice_no),
                    "StockCode": code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": order_date + dt.timedelta(
                        hours=random.randint(8, 19), minutes=random.randint(0, 59)),
                    "Price": price,
                    "Customer ID": customer_id,
                    "Country": country,
                })

    df = pd.DataFrame(rows)

    # Sprinkle in a small number of cancelled orders (Invoice starting with 'C')
    # to mimic real-world data quality issues the notebook must clean.
    n_cancel = int(len(df) * 0.02)
    cancel_idx = df.sample(n_cancel, random_state=42).index
    df.loc[cancel_idx, "Invoice"] = "C" + df.loc[cancel_idx, "Invoice"]
    df.loc[cancel_idx, "Quantity"] = -df.loc[cancel_idx, "Quantity"]

    # Sprinkle a few missing Customer IDs (guest checkouts) - also realistic noise
    n_missing = int(len(df) * 0.015)
    missing_idx = df.sample(n_missing, random_state=1).index
    df.loc[missing_idx, "Customer ID"] = np.nan

    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "online_retail_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} transaction rows for {df['Customer ID'].nunique():,} unique customers")
    print(f"Saved to {out_path}")
