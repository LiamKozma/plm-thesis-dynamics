import pandas as pd
import glob

# 1. Load and combine all 24 CSVs
files = glob.glob("tuning_result_*.csv")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

# 2. Save the master file so you have it
df.to_csv("master_tuning_1M.csv", index=False)

# 3. Filter for the Sweet Spot (Purity 50-70% | Promiscuity 40-60%)
sweet_spot = df[
    (df["Purity_%"] >= 50)
    & (df["Purity_%"] <= 70)
    & (df["Promiscuity_%"] >= 40)
    & (df["Promiscuity_%"] <= 60)
].copy()

if sweet_spot.empty:
    print(
        "\nNo configurations hit BOTH the Purity and Promiscuity targets exactly."
    )
    print("Here are the top 5 closest based on Purity:")
    # Calculate distance from ideal 60% purity
    df["Purity_Diff"] = abs(df["Purity_%"] - 60)
    best_alternatives = df.sort_values("Purity_Diff").head(5)
    print(
        best_alternatives[
            [
                "Oracle_Layers",
                "Base_Sigma",
                "Purity_%",
                "Promiscuity_%",
                "Coverage",
            ]
        ].to_string(index=False)
    )
else:
    # 4. Sort the winners by how close Coverage is to 10
    sweet_spot["Coverage_Diff"] = abs(sweet_spot["Coverage"] - 10)
    sweet_spot = sweet_spot.sort_values("Coverage_Diff")

    print("\n--- SWEET SPOT MATCHES ---")
    print(
        sweet_spot[
            [
                "Oracle_Layers",
                "Base_Sigma",
                "Purity_%",
                "Promiscuity_%",
                "Coverage",
            ]
        ].to_string(index=False)
    )
