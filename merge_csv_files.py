#!/usr/bin/env python3
"""
Script to merge two CSV files by stacking them vertically.
V1 file will be placed on top of the other file.
"""

import pandas as pd

# Define file paths
v1_file = "results/gemini_3_pro/cxr_multi_history_base_multi_versions (V1).csv"
v2_file = "results/gemini_3_pro/cxr_multi_history_base_multi_versions (V2).csv"
output_file = "results/gemini_3_pro/cxr_multi_history_base_multi_versions.csv"

# Read both CSV files
print(f"Reading {v1_file}...")
df_v1 = pd.read_csv(v1_file)
print(f"V1 file shape: {df_v1.shape}")

print(f"Reading {v2_file}...")
df_v2 = pd.read_csv(v2_file)
print(f"V2 file shape: {df_v2.shape}")

# Stack V1 on top of V2 (concatenate vertically)
print("Merging files (V1 on top)...")
df_merged = pd.concat([df_v1, df_v2], ignore_index=True)
print(f"Merged file shape: {df_merged.shape}")

# Save the merged dataframe
print(f"Saving merged file to {output_file}...")
df_merged.to_csv(output_file, index=False)
print("Done!")

# Display summary
print("\n=== Summary ===")
print(f"V1 rows: {len(df_v1)}")
print(f"V2 rows: {len(df_v2)}")
print(f"Total merged rows: {len(df_merged)}")
print(f"Columns: {list(df_merged.columns)}")
