# Sasol Loyalty Customer Value & Opportunity Prediction: Dataset & Task Overview

## About the Challenge

The dataset contains historical loyalty transactions recorded across Sasol forecourt and convenience retail locations. Each row represents an individual line item within a transaction, linked to an anonymised loyalty customer.

* **Transaction Structure:** Transactions may contain fuel purchases, convenience-store items, or both. Multiple rows can therefore belong to the same transaction, sharing the same `TransactionId` and `LoyaltyCustomerNumber`, but differing by products or line-item sequences.
* **Level of Granularity:**
  * **Training Data:** Provided at the **transaction level**, detailing historical purchase activity for loyalty customers across the observation window. You must aggregate and transform these transactions into customer-level feature representations.
  * **Test Data:** Provided at the **customer level**, with exactly one row per `LoyaltyCustomerNumber` ($= \text{ID}$) requiring predictions.
* **Temporal Setup:** The split is time-based. Features must be engineered strictly from historical transaction activity preceding the prediction cutoff to predict outcomes for the subsequent 3-month evaluation window (Dec 2025 – Feb 2026).

---

## Objectives & Target Variables

For each customer in the test dataset, your models must predict three outcomes over the future 3-month evaluation window:

1. **`CLV_fuel`**: Total fuel volume purchased (in litres), submitted as log-transformed, normalized target $z$.
2. **`CLV_nonfuel`**: Total non-fuel spend (in rands), submitted as log-transformed, normalized target $z$.
3. **`Opportunity`**: Most likely customer behaviour or opportunity category (e.g., inactivity, new-category adoption, existing-category growth, or stable behaviour).

---

## Files Manifest

| File | Size | Description |
| :--- | :--- | :--- |
| `SampleSubmission.csv` | ~134 KB | Example submission template illustrating required columns and formats. Row order is flexible, but `ID` values must match exactly. |
| `train.csv` | ~71.8 MB | Historical line-item transaction data used to engineer features and extract training labels. |
| `test.csv` | ~75 KB | Test dataset containing target customer IDs (`LoyaltyCustomerNumber`) evaluated over the 3-month period succeeding `train.csv` (Dec 2025 – Feb 2026). |
| `DATA_DICTIONARY.md` | ~1.2 KB | Detailed schema, field definitions, and data types for the provided tables. |
| `LABEL_RULES.md` | ~4.3 KB | Formal specification and business logic used to generate the ground-truth targets and normalization constants. |
| `label_config.json` | ~4.9 KB | Machine-readable configuration parameters associated with target definitions. |
| `label_rules.py` | ~5.4 KB | Executable Python script implementing target generation and normalization rules locally. |