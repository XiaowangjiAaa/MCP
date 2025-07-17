import os
import pandas as pd

def append_to_csv(csv_path: str, image_name: str, values: dict) -> str:
    """Append results to a CSV file.

    Creates directories automatically, overwrites duplicate image names and
    fills missing columns when necessary.
    """
    row = {"Image": image_name}
    row.update(values)

    df_new = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)

        # add empty columns for new fields
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = None

        # add empty columns for old fields missing in new data
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = None

        # remove duplicates (overwrite existing image)
        df_existing = df_existing[df_existing["Image"] != image_name]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # ensure Image column is first
        columns = ["Image"] + [col for col in df_combined.columns if col != "Image"]
        df_combined = df_combined[columns]
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    return csv_path
