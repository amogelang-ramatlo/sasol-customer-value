"""Pre-caching v7 snapshot features for all 15 monthly cutoffs."""

import time
import pandas as pd
from sasol_customer_value.data.loader import load_transactions
from sasol_customer_value.data.snapshots import build_snapshot, cached_paths

cutoffs = [str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")]
print(f"Building/caching v7 features for {len(cutoffs)} cutoffs: {cutoffs[0]} to {cutoffs[-1]}")

t0 = time.time()
df_trans = load_transactions()
print(f"Loaded transactions in {time.time()-t0:.1f}s")

for i, c in enumerate(cutoffs, 1):
    xp, yp = cached_paths(c)
    t_c = time.time()
    print(f"[{i}/{len(cutoffs)}] Building snapshot {c}...")
    X, y = build_snapshot(df_trans, c, use_cache=True)
    print(f"    Snapshot {c} done in {time.time()-t_c:.1f}s -> X shape: {X.shape}")

print(f"\nAll v7 snapshots cached successfully in {time.time()-t0:.1f}s!")
