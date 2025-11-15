
"""
generate_synthetic.py
Generates a synthetic, noisy transaction dataset for training/testing a transaction categoriser.

Usage:
    python generate_synthetic.py --rows-per-cat 1000 --test-size 0.15 --val-size 0.15 --seed 42

Outputs:
    data/synthetic_full.csv
    data/train.csv
    data/val.csv
    data/test.csv
    data/meta.json
"""

import argparse
import csv
import json
import random
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------
# Configuration (editable)
# -------------------------
TAXONOMY = {
    "food": ["Starbucks", "McDonald's", "KFC", "Dominos", "Subway", "Burger King", "McDonalds"],
    "shopping": ["Amazon", "Walmart", "Target", "Best Buy", "eBay"],
    "transport": ["Uber", "Lyft", "Shell", "BP", "Exxon", "Taxi Co"],
    "entertainment": ["Netflix", "Spotify", "Disney+", "YouTube Premium"],
    "technology": ["Apple Store", "Google Play", "Microsoft Store", "Dell", "Samsung"],
    "groceries": ["Whole Foods", "Trader Joe's", "Aldi", "Kroger", "Safeway"],
    "utilities": ["Comcast", "AT&T", "Verizon", "PG&E", "DTE Energy"],
    "healthcare": ["CVS Pharmacy", "Walgreens", "Rite Aid", "Health Clinic"]
}

# realistic description phrases by category (helps diversify "description" field)
DESCRIPTION_TEMPLATES = {
    "food": ["POS PURCHASE", "MOBILE ORDER", "INSTORE", "MEAL", "APP PURCHASE", "COFFEE"],
    "shopping": ["ONLINE ORDER", "WEB PURCHASE", "MKTPLACE", "ORDER", "RETAIL"],
    "transport": ["RIDE", "TRIP", "FUEL CHARGE", "GASOLINE", "TOLL"],
    "entertainment": ["MONTHLY SUBSCRIPTION", "STREAMING", "SUBSCRIPTION", "FEE"],
    "technology": ["APP PURCHASE", "ELECTRONICS", "SOFTWARE", "DEVICE"],
    "groceries": ["GROCERY", "SUPERMARKET", "INSTORE", "DELI", "FRESH"],
    "utilities": ["BILL PAYMENT", "AUTOPAY", "SERVICE CHARGE", "UTILITY BILL"],
    "healthcare": ["PHARMACY", "PRESCRIPTION", "CLINIC", "HEALTHCARE SERVICE"]
}

REGIONS = ["NY", "CA", "TX", "FL", "IL", "WA", "MA", "NJ"]
with open("taxonomy.json", "r") as f:
    categories = json.load(f)
# -------------------------
# Noise utilities
# -------------------------
def inject_prefix(merchant):
    prefixes = ["POS", "DEBIT", "CREDIT", "ONLINE", "ACH", "AMZN MKTP US"]
    if random.random() < 0.25:
        return random.choice(prefixes) + " " + merchant
    return merchant

def inject_suffix(merchant):
    # add store number, city code, or alphanumeric code
    if random.random() < 0.5:
        return merchant + " #" + str(random.randint(10, 9999))
    if random.random() < 0.25:
        return merchant + " " + random.choice(REGIONS)
    if random.random() < 0.15:
        return merchant + " " + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
    return merchant

def mk_ambiguous_amazon(merchant):
    # sometimes transform Amazon to marketplace style
    patterns = ["AMZN MKTP US*", "AMZN.COM/BILL", "AMAZON.COM*"]
    if "Amazon" in merchant and random.random() < 0.6:
        return random.choice(patterns) + str(random.randint(1000,9999))
    return merchant

def char_noise(s, typo_prob=0.08):
    # introduce small typos: swap, delete, replace
    s = list(s)
    i = 0
    while i < len(s):
        if random.random() < typo_prob:
            op = random.choice(["swap", "drop", "replace"])
            if op == "swap" and i < len(s)-1:
                s[i], s[i+1] = s[i+1], s[i]
                i += 2
                continue
            elif op == "drop":
                s.pop(i)
                continue
            elif op == "replace":
                s[i] = random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
        i += 1
    return "".join(s)

def abbreviate(merchant):
    # some merchants appear as abbreviated forms
    abbr_map = {
        "Starbucks": ["STBK", "STARBUCKS"],
        "McDonald's": ["MCD", "MCDONALDS", "MCD'S"],
        "Amazon": ["AMZN", "AMAZON"],
        "Walmart": ["WMART", "WALMART"],
        "Whole Foods": ["WHOLEFOODS", "WHOLE FD"]
    }
    for full, abbrs in abbr_map.items():
        if full in merchant and random.random() < 0.4:
            return random.choice(abbrs)
    return merchant

def random_date(last_n_days=365):
    # pick a date within the past last_n_days
    days_ago = random.randint(0, last_n_days)
    dt = datetime.now() - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%d")

def amount_for_category(cat):
    # plausible amount ranges per category
    ranges = {
        "food": (2.5, 60),
        "shopping": (5, 500),
        "transport": (3, 100),
        "entertainment": (4.99, 20),
        "technology": (0.99, 2000),
        "groceries": (5, 250),
        "utilities": (20, 400),
        "healthcare": (5, 500)
    }
    low, high = ranges.get(cat, (1, 300))
    return round(random.uniform(low, high), 2)

# -------------------------
# Row generator
# -------------------------
def generate_row(idx, category):
    merchant = random.choice(TAXONOMY[category])
    # apply several noisy transforms
    merchant = mk_ambiguous_amazon(merchant)
    if random.random() < 0.5:
        merchant = abbreviate(merchant)
    merchant = inject_prefix(merchant)
    merchant = inject_suffix(merchant)
    # small char noise
    if random.random() < 0.25:
        merchant = char_noise(merchant, typo_prob=0.06)
    # description
    desc = random.choice(DESCRIPTION_TEMPLATES[category])
    if random.random() < 0.25:
        # append some extra noise words
        desc += " " + random.choice(["POS", "DEBIT", "ONLINE", "AUTO-PAY"])
    # sometimes make description similar to merchant to simulate short bank strings
    if random.random() < 0.15:
        desc = merchant.split()[0] + " " + random.choice(["PURCHASE", "CHARGE"])
    amt = amount_for_category(category)
    region = random.choice(REGIONS)
    date = random_date(365)
    return {
        "id": f"EXP{idx:06d}",
        "amount": amt,
        "merchant": merchant,
        "description": desc,
        "category": category,
        "region": region,
        "date": date
    }

# -------------------------
# Main generator
# -------------------------
def generate_dataset(rows_per_category=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    idx = 1
    for cat in TAXONOMY.keys():
        for _ in range(rows_per_category):
            r = generate_row(idx, cat)
            rows.append(r)
            idx += 1
    random.shuffle(rows)
    return rows

# -------------------------
# Save, split, metadata
# -------------------------
def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["id", "amount", "merchant", "description", "category", "region", "date"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def save_meta(meta, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# -------------------------
# CLI and execution
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-cat", type=int, default=1000, help="Rows to generate per category")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    rows = generate_dataset(rows_per_category=args.rows_per_cat, seed=args.seed)
    full_path = os.path.join(args.out_dir, "synthetic_full.csv")
    save_csv(rows, full_path)

    # create train/val/test splits
    labels = [r["category"] for r in rows]
    X_indices = list(range(len(rows)))
    train_idx, test_idx = train_test_split(X_indices, test_size=args.test_size, stratify=labels, random_state=args.seed)
    # recompute labels for train split to stratify val split
    train_labels = [labels[i] for i in train_idx]
    train_idx2, val_idx = train_test_split(train_idx, test_size=args.val_size/(1-args.test_size),
                                          stratify=train_labels, random_state=args.seed)

    train_rows = [rows[i] for i in train_idx2]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    save_csv(train_rows, os.path.join(args.out_dir, "train.csv"))
    save_csv(val_rows, os.path.join(args.out_dir, "val.csv"))
    save_csv(test_rows, os.path.join(args.out_dir, "test.csv"))

    meta = {
        "rows_per_category": args.rows_per_cat,
        "n_categories": len(TAXONOMY),
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "seed": args.seed,
        "generated_at": datetime.now().isoformat()
    }
    save_meta(meta, os.path.join(args.out_dir, "meta.json"))
    print("Done. Files written to:", args.out_dir)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
