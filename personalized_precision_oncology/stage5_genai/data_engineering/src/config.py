"""
Configuration and constants for the Data Engineering pipeline.
Defines canonical categories, clinical physiological bounds, paths, and rare variant preservation watchlists.
"""

from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
SCHEMAS_DIR = BASE_DIR / "schemas"
REPORTS_DIR = BASE_DIR / "reports"

# Raw file paths
TCGA_CLINICAL_RAW = RAW_DIR / "tcga_nsclc_clinical_raw.csv"
TCGA_MUTATIONS_RAW = RAW_DIR / "tcga_nsclc_mutations_raw.csv"
MSK_IMPACT_RAW = RAW_DIR / "msk_impact_nsclc_raw.csv"
SEER_EPIDEMIOLOGY_RAW = RAW_DIR / "seer_nsclc_epidemiology_raw.csv"
CLINVAR_CIVIC_RAW = RAW_DIR / "clinvar_civic_resistance_raw.csv"
BIOMARKERS_EVIDENCE_RAW = RAW_DIR / "oncology_biomarkers_evidence_raw.csv"

# Processed file paths
CLEANED_COHORT_CSV = PROCESSED_DIR / "cleaned_cohort.csv"
CANCER_TYPE_DIST_JSON = PROCESSED_DIR / "cancer_type_distribution.json"
STAGE_DIST_JSON = PROCESSED_DIR / "stage_distribution.json"
AGE_DIST_JSON = PROCESSED_DIR / "age_distribution.json"
SEX_DIST_JSON = PROCESSED_DIR / "sex_distribution.json"
MUTATION_FREQ_JSON = PROCESSED_DIR / "mutation_frequency.json"
MUTATION_COOCCURRENCE_JSON = PROCESSED_DIR / "mutation_cooccurrence.json"
BIOMARKER_DIST_JSON = PROCESSED_DIR / "biomarker_distributions.json"
TREATMENT_RESISTANCE_JSON = PROCESSED_DIR / "treatment_resistance_distributions.json"
GENAI_REFERENCE_BASELINE_JSONL = PROCESSED_DIR / "genai_reference_baseline.jsonl"

# Reports
SOURCE_METADATA_CSV = REPORTS_DIR / "source_metadata.csv"
DATA_QUALITY_REPORT_JSON = REPORTS_DIR / "data_quality_report.json"
QUARANTINE_REPORT_JSON = REPORTS_DIR / "quarantine_report.json"

# Missing value sentinels and semantics
# STRICT SEMANTIC DEFINITIONS:
# A. None / np.nan: Genuinely missing patient-level value where assay was performed or expected
# B. not_available_in_source: Original source registry does not contain/assay that variable
# C. not_observed_in_reference: Genomic mutation/pattern was not observed in the reference data (does NOT mean biologically impossible)
# D. insufficient_evidence: Some evidence exists, but insufficient for clinical consensus
# E. source_limitation: The source/data type does not support the requested analysis
UNAVAILABLE_SENTINEL = "not_available_in_source"
NOT_OBSERVED_SENTINEL = "not_observed_in_reference"
INSUFFICIENT_EVIDENCE_SENTINEL = "insufficient_evidence"
SOURCE_LIMITATION_SENTINEL = "source_limitation"

# Source-Aware Taxonomy
SOURCE_TYPES = [
    "patient_cohort",
    "epidemiology",
    "variant_evidence",
    "biomarker_evidence",
    "literature_reference"
]

DATA_DOMAINS = [
    "clinical",
    "genomic",
    "epidemiology",
    "resistance",
    "biomarker",
    "treatment_response"
]

EVIDENCE_STATUSES = [
    "prospective_clinical_cohort",
    "retrospective_surgical_cohort",
    "population_registry",
    "curated_clinical_evidence",
    "published_guideline_benchmark"
]

# Valid physiological and clinical ranges
CLINICAL_BOUNDS = {
    "age": {"min": 18, "max": 115},
    "pack_years": {"min": 0.0, "max": 200.0},
    "pdl1_tps_percent": {"min": 0.0, "max": 100.0},
    "ctdna_maf_percent": {"min": 0.0, "max": 100.0},
    "tmb_mut_per_mb": {"min": 0.0, "max": 500.0},
    "overall_survival_months": {"min": 0.0, "max": 300.0},
    "progression_free_months": {"min": 0.0, "max": 300.0},
}

# Standardized Stage Normalization Dictionary
STAGE_NORMALIZATION_MAP = {
    "stage ia": "Stage IA",
    "stage ib": "Stage IB",
    "stage i": "Stage I",
    "ia": "Stage IA",
    "ib": "Stage IB",
    "stage iia": "Stage IIA",
    "stage iib": "Stage IIB",
    "stage ii": "Stage II",
    "iia": "Stage IIA",
    "iib": "Stage IIB",
    "stage iiia": "Stage IIIA",
    "stage iiib": "Stage IIIB",
    "stage iiic": "Stage IIIC",
    "stage iii": "Stage III",
    "iiia": "Stage IIIA",
    "iiib": "Stage IIIB",
    "stage iva": "Stage IVA",
    "stage ivb": "Stage IVB",
    "stage iv": "Stage IV",
    "iv": "Stage IV",
    "distant": "Stage IV",
    "regional": "Stage III",
    "localized": "Stage I",
    "unstaged": "Unstaged",
    "unknown": "Unstaged"
}

# Standardized Sex Normalization
SEX_NORMALIZATION_MAP = {
    "male": "Male",
    "m": "Male",
    "female": "Female",
    "f": "Female",
    "unknown": "Unknown"
}

# Critical Rare & Resistance Mutations Watchlist (NEVER prune these during outlier rejection)
PROTECTED_RARE_VARIANTS = {
    "EGFR": ["p.C797S", "p.T790M", "p.L718Q", "p.G796S", "p.A763_Y764insFQEA", "p.T790M;p.C797S"],
    "KRAS": ["p.Y99C", "p.G12D", "p.G13D", "p.Q61H", "p.G12A"],
    "ALK": ["p.G1202R", "p.I1171N", "p.L1196M", "EML4-ALK"],
    "ROS1": ["p.G2032R", "CD74-ROS1"],
    "MET": ["p.D1010H", "p.D1010Y", "MET_amplification"],
    "ERBB2": ["p.Y772_A775dup", "HER2_amplification"],
    "RET": ["KIF5B-RET", "p.G810R"],
    "BRAF": ["p.V600E", "p.G469A"],
    "STK11": ["p.Q37*", "p.F354fs", "p.E199*"],
    "KEAP1": ["p.G333C", "p.R320Q", "p.W544*", "p.D236H"],
    "NFE2L2": ["p.E79K", "p.R34Q", "p.E79Q"]
}
