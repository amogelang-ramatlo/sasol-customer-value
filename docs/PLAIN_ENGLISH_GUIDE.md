# Sasol Loyalty Competition: The Plain-English & Real-World Guide

A practical, layman's guide explaining the business context, the till slip data, the three prediction targets, and the competition rulebook.

---

## 1. The Big Picture: What is this competition really about?

Imagine you are the manager of Sasol’s customer loyalty program in South Africa.

Sasol does not just operate petrol stations that pump fuel at the forecourt; they also run convenience stores (like Sasol Delight shops) that sell cold drinks, pies, groceries, bread, coffee, and car products. When customers visit a Sasol station, they swipe their loyalty card at the cash register.

Sasol wants to know **what each customer will do over the next 3 months** based on how they shopped in the past.

Specifically, for each customer, Sasol wants us to answer three practical questions:
1. **How much fuel will they buy?** (Litres over the next 3 months)
2. **How much shop merchandise will they buy?** (Rands spent in the convenience store over the next 3 months)
3. **What is their overall shopping trajectory or biggest sales opportunity?** (Will they churn? Will they stay predictable? Or will they start buying more of something?)

---

## 2. The 3 Targets & Specialist Models (Model F, Model N, Model O)

### Target 1: `CLV_fuel` -> Model F (`FuelModel`)
* **What it means:** The total volume of petrol or diesel (in litres) the customer will buy over the 3-month evaluation window.
* **Why the formula $z = \frac{\ln(1 + \text{litres})}{2.3528}$?**  
  In real life, some drivers buy 20 litres a month, while taxi or fleet drivers might pump 2,000 litres. If our model tried to predict raw litres directly, a single huge customer would heavily distort the model's accuracy. Taking the natural logarithm ($\ln(1 + y)$) compresses extreme numbers, and dividing by the fixed constant $s_{\text{fuel}} = 2.3528$ scales everyone smoothly between roughly $0$ and $3$.

### Target 2: `CLV_nonfuel` -> Model N (`NonfuelModel`)
* **What it means:** The total amount of money in South African Rands (ZAR) the customer spends inside the convenience store over the 3 months (excluding fuel).
* **The scaling formula $z = \frac{\ln(1 + \text{rands})}{2.4201}$:**  
  Just like fuel litres, raw Rands are log-compressed and scaled by $s_{\text{nonfuel}} = 2.4201$ so large spenders do not dominate the scoring metric.

### Target 3: `Opportunity` -> Model O (`OpportunityModel`)
For every customer, we must predict **exactly one label out of 17 possible options**. Think of this as categorizing the customer into the single best bucket describing their next 3 months:

1. **`Inactivity`**: The customer vanished. They did not make a single qualifying purchase at Sasol during the 3 months. (Churn / lapsed loyalty member).
2. **`Stable`**: The customer kept shopping, but their spend remained steady or dropped slightly. They didn't try anything brand new, and none of their categories grew significantly.
3. **`Existing-category growth` (4 categories):** The customer already buys from this category, but their spending jumped significantly in the outcome quarter:
   * `Existing-category growth: Fuel`
   * `Existing-category growth: Beverages`
   * `Existing-category growth: Food & beverage`
   * `Existing-category growth: Other`
4. **`New category adoption` (11 categories):** The customer bought a product from a category they **had never touched in their entire observed history**:
   * `New category adoption: Beverages`
   * `New category adoption: Braai & Ice`
   * `New category adoption: Breads & rolls`
   * `New category adoption: Confectionary`
   * `New category adoption: Food & beverage`
   * `New category adoption: Groceries`
   * `New category adoption: Ice-cream`
   * `New category adoption: Lubricants`
   * `New category adoption: Other`
   * `New category adoption: Snacking`
   * `New category adoption: Tobacco`

---

## 3. The Dataset: The Till Slip Analogy

To understand `train.csv` (~470,000 rows) and `test.csv` (1,280 customer IDs), imagine a customer walking into a Sasol station.

### The Real-World Scenario
A customer drives up to pump 30 litres of petrol, walks into the Delight store to buy a 500ml Coca-Cola and a meat pie, and hands their loyalty card to the cashier.

That single visit creates **one receipt (basket)** with **three printed lines**:
1. Line 1: 30 Litres Petrol (Category: `Fuel`)
2. Line 2: 1x 500ml Coca-Cola (Category: `Beverages`)
3. Line 3: 1x Meat Pie (Category: `Food & beverage`)

In `train.csv`, that single trip generates **three rows**, all sharing the same `TransactionId` and `ID` (customer number).

---

## 4. The Data Dictionary (Field-by-Field in Plain English)

| Column | What It Represents on the Till Slip | Layman's Explanation & Critical Pitfalls |
| :--- | :--- | :--- |
| **`ID`** | The Customer Loyalty Card Number | The unique account string (e.g. `4001000000001`). Must be read as text (string) so leading zeros or large numbers are not altered. |
| **`TransactionId`** | The Till Slip / Receipt Number | The unique code for the whole shopping basket. All items purchased during the same cash register session share this ID. |
| **`LineItemSequence`** | Line Number on the Receipt | Line 1, line 2, line 3 on the printed slip showing the order items were scanned. |
| **`LocationId`** | Station / Forecourt Code | The specific Sasol franchise or retail store where the customer shopped. |
| **`DateTimeZA`** | Date and Time Stamp | Local South African timestamp. Tells us customer purchase cadence (e.g. morning commuter vs. weekend road tripper). |
| **`LineItemType`** | Till Line Classification | Tells the system what kind of transaction this was. Fuel purchases are specifically labeled **`Sale_Fuel`**. |
| **`StockItemId`** | Product Barcode / SKU | Sasol's internal product code for the exact item. |
| **`ItemName`** | Product Description | The text description printed on the slip (e.g. `ULP95`, `COKE 500ML`, or discount line items like `PENSIONER DISCOUNT`). |
| **`ItemCategoryLevel1`** | Broad Department | High-level division (e.g. `Forecourt` vs. `Convenience`). |
| **`ItemCategoryLevel2`** | Department / Sub-Department | **The most critical category column.** This level defines growth and adoption rules (e.g. `Fuel`, `Beverages`, `Snacking`, `NA`). |
| **`ItemCategoryLevel3`** | Detailed Product Group | Finer product classification (e.g. `Petrol`, `Diesel`, `Carbonated Soft Drinks`). |
| **`Quantity`** | Litres or Item Units | **Litres** for fuel (`Sale_Fuel`); unit count (e.g. 1, 2) for store merchandise. Negative values represent product returns/refunds. |
| **`TotalExclAmount`** | Actual Paid Line Amount | **The primary money column.** Pre-tax for shop goods; tax-inclusive for fuel. If an item was returned, this is negative and reduces net spend. |
| **`TransactionTotalInclAmount`** | Receipt Grand Total (Repeated!) | **THE BIG TRAP:** The total bill for the entire basket, repeated on every single row of the receipt. **NEVER SUM THIS COLUMN**, or you will multiply the basket total by the number of items! |
| **`TotalModifierAmount`** | Promotional Discount | Discounts or specials already deducted. **Do not subtract this again** from `TotalExclAmount`, or discounts will be double-counted. |
| **`UnitSellingExclPrice`** | Base Shelf Price per Unit | Regular shelf price before discounts. **Do not do manual math** (`Quantity * UnitSellingExclPrice`); always trust `TotalExclAmount`. |

### The Critical `"NA"` Category Nuance
In data science, the letters "NA" usually mean "Not Available" (a blank cell or missing data). However, in Sasol’s point-of-sale system, **`"NA"` is an actual, legitimate retail category**! 

The competition rules explicitly require that `"NA"` is never converted to missing values (`NaN`). If a customer purchases an item in category `"NA"`, that spend counts toward their total and keeps them from being labeled as `Inactivity`.

---

## 5. The Rulebook: What is `label_config.json` Doing?

The competition organizers provided `docs/competition/label_config.json` as the machine-readable rulebook that generates the official answer key:

1. **The 3-Visit Eligibility Rule (`minimum_historical_transactions: 3`):**  
   A customer must have made at least 3 distinct basket visits strictly before the cutoff date. Anyone with fewer than 3 visits is considered too new to forecast reliably and is excluded from the competition cohort.
2. **Discount Exclusion (`adjustment_items`):**  
   Line items like `TOTAL/ROUNDING DISCO`, `PENSIONER DISCOUNT`, and `WOW_PROMO` are register adjustments, not real product purchases. They are filtered out completely.
3. **What Counts as "Growth"? (`growth_min_rand: 20`, `growth_min_relative: 0.25`):**  
   A customer doesn't qualify for a "Growth" label just by spending R2 more. They must meet both conditions:
   * Spend **at least R20 more** in that category than in the previous quarter.
   * Increase their category spend by **at least 25%** over the previous quarter.
4. **What Counts as "Adoption"? (`adoption_min_rand: 0`, `adoption_min_purchases: 1`):**  
   The customer had **zero prior purchases** in that category across their entire observed history, but bought it at least once with positive spend during the outcome quarter.
5. **How Does a Winning Category Get Picked?**  
   If a customer qualifies for multiple opportunities (e.g. grew fuel by R150, but also bought coffee for the first time for R40):
   * Each candidate category competes on **net Rand increase** (even fuel uses Rands, not litres, for this competition).
   * Whichever category brought in the largest monetary gain wins!
   * In the event of a tie, **Adoption beats Growth**, followed by alphabetical order of the category name.
6. **Exclusions (`included: false`):**  
   Categories like `Carrier Bags`, `Print media` (newspapers), and `Value added services` do not count toward purchase activity or sales opportunities.

---

## 6. How is the Scoreboard Calculated?

The official leaderboard score combines our error across all three targets into a single loss score:

$$\text{Final Score} = 0.30 \times \text{RMSE}(\text{Fuel Litres}) + 0.30 \times \text{RMSE}(\text{Shop Rands}) + 0.40 \times \text{Weighted F1}(\text{Opportunity})$$

* **Fuel Litres (30% weight):** How closely our normalized fuel predictions match actual fuel pumped.
* **Shop Spend (30% weight):** How closely our normalized store spend predictions match actual store spend.
* **Opportunity (40% weight):** How accurately our models classify the customer into the correct 1 of 17 behavioral categories (weighted to account for both common and rare classes).
* **Objective:** A **lower score** represents fewer errors and higher prediction accuracy.

---

## 7. The 3 Golden Rules of this Dataset

1. **For Fuel Litres:** Look at `Quantity` where `ItemCategoryLevel2 == 'Fuel'` and `LineItemType == 'Sale_Fuel'`.
2. **For Shop Rands:** Sum `TotalExclAmount` directly across in-scope shop categories. Never calculate prices manually from unit price.
3. **Never Sum `TransactionTotalInclAmount`:** It is repeated on every row of the same basket.
