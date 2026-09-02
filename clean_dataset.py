import csv
import math
from collections import defaultdict

input_file = "oncology_synthetic_dataset.csv"
output_file = "oncology_cleaned_dataset.csv"

# Null representations to normalize
NULL_TOKENS = {"", "null", "nan", "n/a", "none"}

def is_null(val):
    if val is None:
        return True
    val_str = str(val).strip().lower()
    return val_str in NULL_TOKENS

def clean_data():
    raw_rows = []
    
    # 1. Read CSV file
    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row: # ignore trailing empty lines
                raw_rows.append(row)
                
    initial_count = len(raw_rows)
    print(f"Initial raw rows: {initial_count}")

    # 2. Normalize NULL values
    normalized_rows = []
    for row in raw_rows:
        norm_row = [None if is_null(cell) else cell.strip() for cell in row]
        normalized_rows.append(norm_row)

    # 3. Deduplicate exact duplicate rows
    seen_rows = set()
    deduped_rows = []
    duplicate_count = 0
    
    for row in normalized_rows:
        row_tuple = tuple(row)
        if row_tuple in seen_rows:
            duplicate_count += 1
        else:
            seen_rows.add(row_tuple)
            deduped_rows.append(row)

    print(f"Duplicates removed: {duplicate_count}")

    # 4. Compute statistics for imputation
    ages = [float(r[1]) for r in deduped_rows if r[1] is not None]
    tumor_sizes = [float(r[6]) for r in deduped_rows if r[6] is not None]
    survivals = [float(r[10]) for r in deduped_rows if r[10] is not None]

    median_age = round(sorted(ages)[len(ages) // 2]) if ages else 60
    median_tumor = round(sorted(tumor_sizes)[len(tumor_sizes) // 2], 1) if tumor_sizes else 4.2
    median_survival = round(sorted(survivals)[len(survivals) // 2], 1) if survivals else 24.0

    # 5. Clean & Impute fields
    cleaned_rows = []
    pat_counter = 1001

    for row in deduped_rows:
        pid, age, gender, ctype, stage, biomarker, tumor_size, treatment, line, response, survival, vital = row

        # Patient ID standardization
        if pid is None or pid == "":
            pid = f"PAT-CLEAN-{pat_counter}"
            pat_counter += 1

        # Age imputation
        if age is None:
            age = str(median_age)
        else:
            try:
                age = str(int(float(age)))
            except ValueError:
                age = str(median_age)

        # Gender imputation
        if gender is None:
            gender = "Unknown"

        # Cancer Type
        if ctype is None:
            ctype = "Unspecified Cancer"

        # Stage imputation
        if stage is None:
            stage = "Unknown Stage"

        # Biomarker imputation
        if biomarker is None:
            biomarker = "Not Tested"

        # Tumor Size imputation
        if tumor_size is None:
            tumor_size = str(median_tumor)
        else:
            try:
                tumor_size = str(round(float(tumor_size), 1))
            except ValueError:
                tumor_size = str(median_tumor)

        # Primary Treatment imputation
        if treatment is None:
            treatment = "Unspecified Treatment"

        # Line of Therapy imputation
        if line is None:
            line = "Unspecified Line"

        # Response Eval imputation
        if response is None:
            response = "Not Evaluated"

        # Overall Survival imputation
        if survival is None:
            survival = str(median_survival)
        else:
            try:
                survival = str(round(float(survival), 1))
            except ValueError:
                survival = str(median_survival)

        # Vital Status imputation
        if vital is None:
            vital = "Unknown"

        cleaned_rows.append([
            pid, age, gender, ctype, stage, biomarker,
            tumor_size, treatment, line, response, survival, vital
        ])

    # 6. Write Cleaned Dataset
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned_rows)

    print(f"Cleaned dataset saved to '{output_file}' with {len(cleaned_rows)} rows.")
    return initial_count, duplicate_count, len(cleaned_rows)

if __name__ == "__main__":
    clean_data()
