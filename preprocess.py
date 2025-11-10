import pandas as pd

def main():
    # Load in file from csv download on GTEx histology
    df = pd.read_csv('gtex.csv')

    # Group by 'Tissue'
    grouped = df.groupby("Tissue")

    # Drop 'Subject ID' (not needed)
    df = df.drop(columns=["Subject ID"])

    # Example: print each group
    for tissue, group in grouped:
        print(f"\n=== {tissue} ===")
        print(group[["Tissue Sample ID", "Subject ID", "Sex", "Age Bracket", "Hardy Scale", "Pathology Notes"]])

    # Add your path prefix as the file path in the new csv
    path_prefix = "/shares/sinha/sinha_common/GTEx/h5-tiles-256/"

    # Replace Tissue Sample ID with full file path with new h5 extension
    df["Tissue Sample ID"] = df["Tissue Sample ID"].apply(lambda x: f"{path_prefix}{x}.h5")

    # Save the updated table for usage in training
    df.to_csv("gtex_features.csv", index=False)
