"""
Core analysis logic shared by the notebook.
Run standalone with: python analysis_core.py
Produces: outputs/rfm_clusters.csv, outputs/cluster_summary.csv, outputs/plots/*.png
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "online_retail_synthetic.csv")
OUT_DIR = os.path.join(BASE, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    n_raw = len(df)

    df = df[~df["Invoice"].astype(str).str.startswith("C")]          # drop cancellations
    df = df.dropna(subset=["Customer ID"])                            # drop guest checkouts
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]                 # drop bad rows
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    print(f"Raw rows: {n_raw:,}  ->  Cleaned rows: {len(df):,}  "
          f"({n_raw - len(df):,} removed)")
    return df


# ---------------------------------------------------------------
# 2. RFM FEATURE ENGINEERING
# ---------------------------------------------------------------
def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm


# ---------------------------------------------------------------
# 3. TRANSFORM + SCALE
# ---------------------------------------------------------------
def transform_scale(rfm):
    rfm_log = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_log[col] = np.log1p(rfm_log[col])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    return rfm_scaled


# ---------------------------------------------------------------
# 4. CHOOSE K (elbow + silhouette)
# ---------------------------------------------------------------
def choose_k(rfm_scaled, k_range=range(2, 9)):
    inertia, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(list(k_range), inertia, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="darkorange")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_k_selection.png"))
    plt.close()

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    return best_k, dict(zip(k_range, inertia)), dict(zip(k_range, sil_scores))


# ---------------------------------------------------------------
# 5. FIT KMEANS + COMPARE WITH DBSCAN
# ---------------------------------------------------------------
def fit_models(rfm, rfm_scaled, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["KMeans_Cluster"] = kmeans.fit_predict(rfm_scaled)
    kmeans_sil = silhouette_score(rfm_scaled, rfm["KMeans_Cluster"])

    # DBSCAN comparison (eps tuned roughly for scaled RFM space)
    dbscan = DBSCAN(eps=0.6, min_samples=15)
    db_labels = dbscan.fit_predict(rfm_scaled)
    n_noise = int((db_labels == -1).sum())
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_clusters_db > 1:
        db_sil = silhouette_score(rfm_scaled, db_labels)
    else:
        db_sil = None

    comparison = {
        "kmeans_k": k,
        "kmeans_silhouette": round(kmeans_sil, 3),
        "dbscan_clusters_found": n_clusters_db,
        "dbscan_noise_points": n_noise,
        "dbscan_silhouette": round(db_sil, 3) if db_sil else "N/A (too few clusters)",
    }
    return rfm, comparison


# ---------------------------------------------------------------
# 6. PROFILE + NAME CLUSTERS
# ---------------------------------------------------------------
def name_clusters(rfm):
    summary = rfm.groupby("KMeans_Cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Count=("Recency", "size"),
    ).round(1)

    # Rank-based RFM score (1 = best, n_clusters = worst) avoids duplicate
    # labels that a simple median-split can produce when cluster count > 4.
    r_rank = summary["Recency"].rank(method="first")                      # low recency = good = low rank
    f_rank = summary["Frequency"].rank(ascending=False, method="first")   # high freq = good = low rank
    m_rank = summary["Monetary"].rank(ascending=False, method="first")    # high monetary = good = low rank
    summary["_RFM_Score"] = r_rank + f_rank + m_rank                      # lower total = better customer

    ranked = summary.sort_values("_RFM_Score")
    idx_by_rank = ranked.index.tolist()
    r_med = summary["Recency"].median()
    f_med = summary["Frequency"].median()

    labels = {}
    remaining_labels = [
        "Potential Loyalists",
        "New / Low-Engagement Customers",
        "Frequent Bargain Hunters",
        "At-Risk / Churning",
        "Lapsed / Dormant",
    ]

    labels[idx_by_rank[0]] = "High-Value Loyalists"  # best-ranked cluster

    worst = idx_by_rank[-1]
    worst_label = "Lapsed / Dormant" if summary.loc[worst, "Recency"] > r_med else "Frequent Bargain Hunters"
    labels[worst] = worst_label
    if worst_label in remaining_labels:
        remaining_labels.remove(worst_label)

    for cid in idx_by_rank[1:-1]:
        row = summary.loc[cid]
        if row["Recency"] > r_med and "At-Risk / Churning" in remaining_labels:
            lbl = "At-Risk / Churning"
        elif row["Recency"] <= r_med and row["Frequency"] < f_med and \
                "New / Low-Engagement Customers" in remaining_labels:
            lbl = "New / Low-Engagement Customers"
        elif "Potential Loyalists" in remaining_labels:
            lbl = "Potential Loyalists"
        else:
            lbl = remaining_labels[0]
        labels[cid] = lbl
        if lbl in remaining_labels:
            remaining_labels.remove(lbl)

    summary["Segment"] = summary.index.map(labels)
    summary = summary.drop(columns="_RFM_Score")

    action_map = {
        "High-Value Loyalists": "VIP loyalty perks, early access to new products, referral incentives",
        "At-Risk / Churning": "Win-back email campaign, personalized discount, satisfaction survey",
        "New / Low-Engagement Customers": "Onboarding series, first-purchase follow-up, welcome discount",
        "Frequent Bargain Hunters": "Bundle deals, loyalty points on volume, upsell to premium lines",
        "Lapsed / Dormant": "Reactivation campaign or sunset from active marketing spend",
        "Potential Loyalists": "Loyalty program invite, cross-sell recommendations",
    }
    summary["Recommended_Action"] = summary["Segment"].map(action_map)
    return summary


# ---------------------------------------------------------------
# 7. BUSINESS IMPACT ESTIMATE
# ---------------------------------------------------------------
def business_impact(rfm, summary):
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()

    at_risk_row = summary[summary["Segment"] == "At-Risk / Churning"]
    if at_risk_row.empty:
        return None

    at_risk_count = int(at_risk_row["Count"].iloc[0])
    at_risk_avg_monetary = at_risk_row["Monetary"].iloc[0]
    at_risk_total_value = at_risk_count * at_risk_avg_monetary

    # Assumption: a win-back campaign recovers ~15-20% of at-risk customers'
    # historical annual value (industry-typical win-back conversion range).
    recovery_rate_low, recovery_rate_high = 0.15, 0.20
    recovered_low = at_risk_total_value * recovery_rate_low
    recovered_high = at_risk_total_value * recovery_rate_high

    impact = {
        "total_customers": total_customers,
        "total_historical_revenue": round(total_revenue, 2),
        "at_risk_customer_count": at_risk_count,
        "at_risk_pct_of_base": round(100 * at_risk_count / total_customers, 1),
        "at_risk_historical_value": round(at_risk_total_value, 2),
        "estimated_recoverable_revenue_low": round(recovered_low, 2),
        "estimated_recoverable_revenue_high": round(recovered_high, 2),
        "assumption": "15-20% win-back conversion rate on at-risk segment's historical spend",
    }
    return impact


# ---------------------------------------------------------------
# 8. VISUALIZATIONS
# ---------------------------------------------------------------
def make_visuals(rfm, summary):
    # Cluster sizes
    plt.figure(figsize=(7, 4.5))
    order = summary.sort_values("Count", ascending=False)
    sns.barplot(x=order["Segment"], y=order["Count"], palette="viridis")
    plt.xticks(rotation=30, ha="right")
    plt.title("Customer Count by Segment")
    plt.ylabel("Number of Customers")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_segment_sizes.png"))
    plt.close()

    # Scatter: Recency vs Monetary colored by segment
    seg_map = summary["Segment"].to_dict()
    rfm_plot = rfm.copy()
    rfm_plot["Segment"] = rfm_plot["KMeans_Cluster"].map(seg_map)

    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(data=rfm_plot, x="Recency", y="Monetary", hue="Segment",
                     palette="tab10", alpha=0.6, s=25)
    plt.yscale("log")
    plt.title("Customer Segments: Recency vs Monetary Value")
    plt.xlabel("Recency (days since last purchase)")
    plt.ylabel("Monetary Value (log scale)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_recency_vs_monetary.png"))
    plt.close()

    # Radar chart of average RFM per segment (normalized 0-1)
    from math import pi
    radar_df = summary[["Recency", "Frequency", "Monetary"]].copy()
    # invert recency so "higher = better" for all axes (more intuitive radar)
    radar_df["Recency_inv"] = radar_df["Recency"].max() - radar_df["Recency"]
    radar_cols = ["Recency_inv", "Frequency", "Monetary"]
    radar_norm = (radar_df[radar_cols] - radar_df[radar_cols].min()) / \
                 (radar_df[radar_cols].max() - radar_df[radar_cols].min() + 1e-9)

    labels = ["Recency (recent=high)", "Frequency", "Monetary"]
    n_vars = len(labels)
    angles = [n / float(n_vars) * 2 * pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for idx, seg in zip(radar_norm.index, summary.loc[radar_norm.index, "Segment"]):
        values = radar_norm.loc[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=seg, linewidth=1.8)
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Segment Profiles (normalized RFM)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_radar_segment_profiles.png"))
    plt.close()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("Step 1/8: Loading and cleaning data...")
    df = load_and_clean()

    print("Step 2/8: Building RFM features...")
    rfm = build_rfm(df)

    print("Step 3/8: Transforming and scaling...")
    rfm_scaled = transform_scale(rfm)

    print("Step 4/8: Selecting optimal k...")
    best_k, inertia, sil_scores = choose_k(rfm_scaled)
    print(f"   -> Best k by silhouette score: {best_k}")

    print("Step 5/8: Fitting KMeans and comparing with DBSCAN...")
    rfm, comparison = fit_models(rfm, rfm_scaled, best_k)
    print(f"   -> {comparison}")

    print("Step 6/8: Naming clusters and mapping business actions...")
    summary = name_clusters(rfm)
    print(summary[["Recency", "Frequency", "Monetary", "Count", "Segment"]])

    print("Step 7/8: Estimating business impact...")
    impact = business_impact(rfm, summary)
    print(f"   -> {impact}")

    print("Step 8/8: Generating visualizations...")
    make_visuals(rfm, summary)

    # Save outputs
    rfm_out = rfm.copy()
    rfm_out["Segment"] = rfm_out["KMeans_Cluster"].map(summary["Segment"].to_dict())
    rfm_out.to_csv(os.path.join(OUT_DIR, "rfm_clusters.csv"))
    summary.to_csv(os.path.join(OUT_DIR, "cluster_summary.csv"))

    import json
    with open(os.path.join(OUT_DIR, "business_impact.json"), "w") as f:
        json.dump(impact, f, indent=2)
    with open(os.path.join(OUT_DIR, "algorithm_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nAll outputs saved to /outputs")
    return rfm, summary, impact, comparison


if __name__ == "__main__":
    main()
