# Data Cleaning Report

## Dataset Before Cleaning
- **Rows:** 10015
- **Columns:** 8
- **Total Missing Values:** 0
- **Empty Text Records:** 10
- **Duplicate Record IDs:** 15
- **Duplicate Clinical Text:** 35
- **Invalid Language Flags:** 0
- **PII Flags:** 0

## Cleaning Operations Performed
- Loaded raw data while preserving original file (MD5 checksum verified).
- Null or empty `text` fields were converted to empty strings safely.
- Null categorical fields (e.g., `note_type`, `department`, `source`, `language`) were safely replaced with `"UNKNOWN"`.
- Performed Unicode normalization (NFKC) on all text to fix accidental mojibake/smart quotes.
- Stripped leading and trailing whitespace from all text.
- Collapsed multiple consecutive whitespaces (including tabs and erratic line breaks) into single spaces.
- Evaluated for PII patterns (emails, SSNs, phone numbers).
- Kept all original 8 columns and EXACT row counts (no rows were dropped).
- Deliberately **AVOIDED** lowercasing, stemming, tokenization, or punctuation removal to preserve clinical intent and sentence structure.

## Dataset After Cleaning
- **Rows:** 10015
- **Columns:** 8
- **Total Missing Values:** 10
- **Empty Text Records:** 10
- **Duplicate Record IDs:** 15
- **Duplicate Clinical Text:** 35
- **Minimum Text Length:** 80 characters
- **Maximum Text Length:** 591 characters
- **Average Text Length:** 274.8 characters

## Data Integrity Confirmation
- ✅ Raw dataset file was strictly preserved and unmodified.
- ✅ All 8 required columns are present: `record_id, patient_id, note_type, timestamp, department, text, source, language`.
- ✅ Medical terminology, negative terms ("denies", "no"), dosage expressions, and abbreviations were fully preserved.
- ✅ Clinical meaning was retained for downstream NER and urgency classification.
