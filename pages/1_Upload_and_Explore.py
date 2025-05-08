# pages/1_📁_Upload_and_Explore.py
import streamlit as st
import pandas as pd
from core import preprocessing

st.header("📁 Upload and Explore Raw LFA Data")

uploaded_file = st.file_uploader("Upload raw CSV data", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Original Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    # Parse strip name and compute normalized metrics
    df_processed, strip_cols = preprocessing.parse_and_augment(df_raw)

    st.subheader("Parsed Strip Name Columns")
    st.write(f"Detected {len(strip_cols)} parts in 'strip name':", strip_cols)

    new_names = {}
    st.markdown("#### Rename Strip Columns")
    for col in strip_cols:
        new_name = st.text_input(f"Rename '{col}' to:", value=col)
        new_names[col] = new_name

    df_processed.rename(columns=new_names, inplace=True)

    # Filter columns to show only from strip parts to CLA_normalized
    start_col = new_names.get(strip_cols[0], strip_cols[0])
    end_col = 'CLA_normalized'
    try:
        start_index = df_processed.columns.get_loc(start_col)
        end_index = df_processed.columns.get_loc(end_col)
        display_df = df_processed.iloc[:, start_index:end_index + 1]

        st.subheader("Processed Data with Normalized Metrics")
        st.dataframe(display_df.head(), use_container_width=True)

        if st.button("✅ Finalize and Save Processed Data"):
            st.session_state["processed_df"] = df_processed
            st.session_state["display_df"] = display_df
            st.session_state["ready_for_page2"] = True
            st.success("Processed data saved. You can now move to 4PL analysis.")

            st.switch_page("pages/2_4PL_and_IC50.py")

        # Optionally allow download
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Processed CSV", csv, "processed_lfa_data.csv", "text/csv")

    except Exception as e:
        st.error(f"Error creating display_df: {e}")
