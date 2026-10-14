"""Production Training & Refitting CLI.

Trains Target Specialists (F, N, O) on all 15 historical monthly snapshots (65k rows)
and generates verified test predictions for the Dec 2025 - Feb 2026 test horizon.
"""

import argparse

from sasol_customer_value.models.pipeline import train_production_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train production models on full historical data"
    )
    parser.add_argument(
        "--config",
        choices=[
            "best_zindi",
            "best_local",
            "champion_v2",
            "champion_v3",
            "champion_v4",
            "champion_v5",
        ],
        default="champion_v5",
        help="Configuration to train",
    )
    parser.add_argument(
        "--target",
        choices=["all", "F", "N", "O"],
        default="all",
        help="Target to train",
    )
    args = parser.parse_args()

    train_production_models(config_name=args.config, target=args.target)
