# core/preprocessing.py
import pandas as pd

def parse_and_augment(df):
    # Strip name parsing
    delimiter = '-' if '-' in df['strip name'].iloc[0] else '_'
    split_cols = df['strip name'].str.split(delimiter, expand=True)
    split_cols.columns = [f"strip_part_{i+1}" for i in range(split_cols.shape[1])]

    df = pd.concat([df, split_cols], axis=1)

    # Extract key raw signals
    df['TLH'] = df['line_peak_above_background_1']
    df['CLH'] = df['line_peak_above_background_2']
    df['TLA'] = df['line_area_1']
    df['CLA'] = df['line_area_2']

    # Normalize peak heights
    df['TLH_normalized'] = df['TLH'] / (df['TLH'] + df['CLH'])
    df['CLH_normalized'] = df['CLH'] / (df['TLH'] + df['CLH'])

    # Normalize areas
    df['TLA_normalized'] = df['TLA'] / (df['TLA'] + df['CLA'])
    df['CLA_normalized'] = df['CLA'] / (df['TLA'] + df['CLA'])

    return df, split_cols.columns.tolist()
