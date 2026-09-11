"""
Matplotlib-based Visualizer for Stage 5 Genomic EDA.
Strictly Matplotlib ONLY (no seaborn). Uses Agg backend.
Produces:
1. visualizations/mutation_frequency.png
2. visualizations/mutation_cooccurrence.png
3. visualizations/stage_genomic_coverage.png
4. visualizations/resistance_coverage.png
"""

import json
from pathlib import Path

# Enforce Agg backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"
GENOMIC_EDA_REPORT_JSON = REPORTS_DIR / "genomic_eda_report.json"
BLIND_SPOT_REPORT_JSON = REPORTS_DIR / "blind_spot_report.json"
DATA_ENG_PROCESSED = BASE_DIR.parent / "data_engineering" / "processed"
MUTATION_FREQ_JSON = DATA_ENG_PROCESSED / "mutation_frequency.json"


def load_reports():
    with open(GENOMIC_EDA_REPORT_JSON, "r", encoding="utf-8") as f:
        eda_data = json.load(f)
    with open(BLIND_SPOT_REPORT_JSON, "r", encoding="utf-8") as f:
        blind_spot_data = json.load(f)
    with open(MUTATION_FREQ_JSON, "r", encoding="utf-8") as f:
        mut_freq_data = json.load(f)
    return eda_data, blind_spot_data, mut_freq_data


def generate_mutation_frequency_chart(mut_freq_data):
    """Generate bar chart of gene mutation frequencies in the reference baseline."""
    gene_level = mut_freq_data.get("gene_level_frequencies", [])
    # Sort genes by frequency
    sorted_genes = sorted(gene_level, key=lambda x: x["frequency_percent"], reverse=True)
    
    genes = [item["gene"] for item in sorted_genes]
    freqs = [item["frequency_percent"] for item in sorted_genes]
    counts = [item["observed_count"] for item in sorted_genes]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(genes, freqs, color="#2b5c8f", edgecolor="#1a365d", width=0.6)

    ax.set_title("Gene Mutation Frequency in Reference Baseline Cohort (N=75)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Gene Symbol", fontsize=12, fontweight="medium")
    ax.set_ylabel("Frequency (%)", fontsize=12, fontweight="medium")
    ax.set_ylim(0, max(freqs) + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with counts and percentages
    for bar, count, freq in zip(bars, counts, freqs):
        height = bar.get_height()
        ax.annotate(f"{freq:.1f}%\n(n={count})",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    output_path = VIS_DIR / "mutation_frequency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def generate_mutation_cooccurrence_chart(eda_data, mut_freq_data):
    """Generate heatmap matrix of observed co-occurring driver mutations using pure Matplotlib."""
    top_coocs = eda_data["mutation_cooccurrence"].get("observed_cooccurrences", [])
    gene_level = mut_freq_data.get("gene_level_frequencies", [])
    
    # Top 8 most frequent genes for clean matrix representation
    top_genes = [g["gene"] for g in gene_level[:8]]
    n_genes = len(top_genes)
    gene_idx = {g: i for i, g in enumerate(top_genes)}
    
    # Initialize matrix
    matrix = np.zeros((n_genes, n_genes))
    
    # Populate diagonal with single gene counts
    for g in gene_level[:8]:
        matrix[gene_idx[g["gene"]], gene_idx[g["gene"]]] = g["observed_count"]
    
    # Populate off-diagonal from observed cooccurrences
    for item in top_coocs:
        combo = item["combination"]
        if "+" in combo:
            parts = [p.strip() for p in combo.split("+")]
            if len(parts) == 2:
                g1, g2 = parts[0], parts[1]
                if g1 in gene_idx and g2 in gene_idx:
                    c = item["observed_instances"]
                    i, j = gene_idx[g1], gene_idx[g2]
                    matrix[i, j] = c
                    matrix[j, i] = c

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    
    # Colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Patient Count / Co-occurrence Count", fontsize=10)

    # Set labels
    ax.set_xticks(range(n_genes))
    ax.set_yticks(range(n_genes))
    ax.set_xticklabels(top_genes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(top_genes, fontsize=10)

    # Annotate values in cells
    for i in range(n_genes):
        for j in range(n_genes):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if val > (matrix.max() / 2) else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Driver Mutation Co-occurrence Matrix (Observed Cohort N=75)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    output_path = VIS_DIR / "mutation_cooccurrence.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def generate_stage_genomic_coverage_chart(eda_data):
    """Generate grouped bar chart showing stage distribution across clinical panels (TCGA vs MSK-IMPACT)."""
    crosstab = eda_data["genomic_and_clinical_coverage"]["stage_genomic_coverage"]["stage_by_source_crosstab"]
    tcga_stages = crosstab.get("TCGA-GDC PanCancer Atlas", {})
    msk_stages = crosstab.get("MSK-IMPACT Targeted Sequencing (cBioPortal)", {})
    
    # Aggregate into 4 major stages: Stage I, Stage II, Stage III, Stage IV
    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    tcga_counts = [
        tcga_stages.get("Stage IA", 0) + tcga_stages.get("Stage IB", 0),
        tcga_stages.get("Stage IIA", 0) + tcga_stages.get("Stage IIB", 0),
        tcga_stages.get("Stage IIIA", 0) + tcga_stages.get("Stage IIIB", 0),
        tcga_stages.get("Stage IV", 0)
    ]
    msk_counts = [
        msk_stages.get("Stage IA", 0) + msk_stages.get("Stage IB", 0),
        msk_stages.get("Stage IIA", 0) + msk_stages.get("Stage IIB", 0),
        msk_stages.get("Stage IIIA", 0) + msk_stages.get("Stage IIIB", 0),
        msk_stages.get("Stage IV", 0)
    ]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    rects1 = ax.bar(x - width/2, tcga_counts, width, label="TCGA PanCancer (Primary Resection WES, N=50)", color="#3182bd", edgecolor="#1a365d")
    rects2 = ax.bar(x + width/2, msk_counts, width, label="MSK-IMPACT (Advanced/Metastatic Panel, N=25)", color="#e6550d", edgecolor="#8c2d04")

    ax.set_title("Genomic Baseline Representation by AJCC Stage (Source Stratified)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Clinical Stage", fontsize=11, fontweight="medium")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0, max(tcga_counts + msk_counts) + 5)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path = VIS_DIR / "stage_genomic_coverage.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def generate_resistance_coverage_chart(eda_data):
    """Generate status breakdown of key resistance mechanisms audited in the baseline."""
    res_audits = eda_data["treatment_resistance"]["candidate_resistance_evaluation"]
    
    mechanisms = []
    statuses = []
    status_colors = {
        "well_represented": "#31a354",
        "rare": "#74c476",
        "sparse": "#fd8d3c",
        "not_observed": "#d95f02"
    }

    status_labels = {
        "well_represented": "Well Represented",
        "rare": "Rare (<3% prevalence)",
        "sparse": "Sparse (n=1 observed)",
        "not_observed": "Not Observed in Reference Baseline"
    }

    for item in res_audits:
        short_name = item["mechanism"].split("(")[0].strip()
        mechanisms.append(short_name)
        statuses.append(item["coverage_status"])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    y_pos = np.arange(len(mechanisms))
    colors = [status_colors.get(s, "#969696") for s in statuses]

    bars = ax.barh(y_pos, [1] * len(mechanisms), color=colors, edgecolor="#252525", height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(mechanisms, fontsize=10)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlim(0, 1.6)
    ax.set_xticks([])
    ax.set_title("Targeted Therapy Resistance Coverage & Representation Status", fontsize=13, fontweight="bold", pad=15)

    # Annotate status text on each bar
    for idx, (bar, status) in enumerate(zip(bars, statuses)):
        label = status_labels.get(status, status)
        ax.text(1.02, bar.get_y() + bar.get_height() / 2, label,
                ha="left", va="center", fontsize=9, fontweight="medium", color="#252525")

    # Custom legend
    handles = [
        plt.Rectangle((0,0),1,1, color=status_colors["sparse"], label="Sparse (n=1 observed in cohort)"),
        plt.Rectangle((0,0),1,1, color=status_colors["not_observed"], label="Not Observed in Reference Baseline")
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    output_path = VIS_DIR / "resistance_coverage.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    eda, bs, mut = load_reports()
    generate_mutation_frequency_chart(mut)
    generate_mutation_cooccurrence_chart(eda, mut)
    generate_stage_genomic_coverage_chart(eda)
    generate_resistance_coverage_chart(eda)
    print("All 4 visualizations generated successfully!")
