import streamlit as st
import pandas as pd
import yaml
import sys
import os
from pathlib import Path
from io import BytesIO

# Import core logic from existing script
# We need to make sure the directory is in path if running from elsewhere, 
# but usually streamlit run app.py from the same dir works fine.
try:
    from kcp_pipeline import (
        Config, 
        normalize_keyword, 
        assign_intent, 
        cluster_topics_semantic, 
        score_rows
    )
except ImportError:
    st.error("Could not import 'kcp_pipeline.py'. Make sure it's in the same directory.")
    st.stop()

# Page config
st.set_page_config(page_title="Keyword Classifier", layout="wide")

st.title("Keyword Classifier Pipeline")

# -------------------------
# Sidebar: Config
# -------------------------
st.sidebar.header("Configuration")

# Config File Loader
config_files = list(Path("rules").glob("*.yaml"))
config_options = {f.name: f for f in config_files}

st.sidebar.subheader("Select Config")
selected_config_name = st.sidebar.selectbox(
    "Choose a file", 
    list(config_options.keys())
)

st.sidebar.subheader("...or Upload New Config")
uploaded_config = st.sidebar.file_uploader("Upload YAML", type=["yaml", "yml"])

if uploaded_config:
    # Save uploaded config
    save_path = Path("rules") / uploaded_config.name
    with open(save_path, "wb") as f:
        f.write(uploaded_config.getbuffer())
    st.sidebar.success(f"Saved {uploaded_config.name}")
    # Force reload of file list by rerunning? 
    # Streamlit usually handles this on next run, but effectively we want to use this file NOW.
    selected_config_file = save_path
elif selected_config_name:
    selected_config_file = config_options[selected_config_name]
else:
    st.sidebar.warning("No config files found in 'rules/' directory.")
    st.stop()

# Load Config
with open(selected_config_file, "r", encoding="utf-8") as f:
    raw_config = yaml.safe_load(f)

# Allow basic parameter tweaking in sidebar
st.sidebar.subheader("Parameters")
min_volume = st.sidebar.number_input("Min Volume", value=raw_config.get("min_volume", 100))
skiprows = st.sidebar.number_input("Skip Rows (Input CSV)", value=raw_config.get("skiprows", 2))
seed = st.sidebar.number_input("Random Seed", value=raw_config.get("seed", 42))

# -------------------------
# Main: File Upload
# -------------------------
st.header("1. Upload Inputs")
uploaded_files = st.file_uploader(
    "Upload Google Keyword Planner Exports (CSV/TSV)", 
    type=["csv", "txt", "tsv"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one keyword file.")
    st.stop()

# Preview raw data
st.subheader("Raw Data Preview")

@st.cache_data
def load_data(uploaded_files, skip_rows):
    dfs = []
    for uploaded_file in uploaded_files:
        # GKP exports are usually UTF-16 + Tab separated
        # We try to detect or just assume the script's default
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-16", sep="\t", skiprows=skip_rows)
        except Exception:
            # Fallback for standard CSVs
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=skip_rows)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

try:
    raw_df = load_data(uploaded_files, skiprows)
    st.write(f"Loaded {len(raw_df)} rows from {len(uploaded_files)} files.")
    st.dataframe(raw_df.head())
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# -------------------------
# Main: Edit Regex / Rules
# -------------------------
st.header("2. Edit Intent Rules")

# Convert intents list of dicts to DataFrame for editing
intents_list = raw_config.get("intents", [])
# Config intents might be loaded as list of dicts from YAML directly
if intents_list and isinstance(intents_list[0], dict):
    # It's already [{"name":..., "regex":...}, ...]
    pass
else:
    # Handle if it was parsed differently or if we used the Config object struct
    pass

intents_df = pd.DataFrame(intents_list)

edited_intents_df = st.data_editor(
    intents_df, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "name": "Intent Name",
        "regex": "Regex Pattern"
    }
)

# Option to save config back?
if st.button("Save Configuration Changes"):
    # Update the raw_config
    new_intents = edited_intents_df.to_dict(orient="records")
    raw_config["intents"] = new_intents
    raw_config["min_volume"] = min_volume
    raw_config["skiprows"] = skiprows
    raw_config["seed"] = seed
    
    try:
        with open(selected_config_file, "w", encoding="utf-8") as f:
            yaml.dump(raw_config, f, sort_keys=False, allow_unicode=True)
        st.success(f"Saved changes to {selected_config_file}!")
    except Exception as e:
        st.error(f"Failed to save config: {e}")


# -------------------------
# Main: Run Classification
# -------------------------
st.header("3. Run Classification")

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Running classification..."):
        # 1. Prepare Config Object
        # We reconstruct the Config object from current UI state
        # The kcp_pipeline.Config expects specific types
        
        # Helper to reconstruct bands/weights from raw_config as they aren't edited in UI yet
        # (Using defaults from file for things not exposed in UI)
        
        # Re-read intents from the EDITED dataframe
        current_intents = [
            (row["name"], row["regex"]) 
            for _, row in edited_intents_df.iterrows() 
            if row["name"] and row["regex"]
        ]
        
        cfg = Config(
            keep_cols=raw_config.get("keep_cols", []),
            skiprows=int(skiprows),
            min_volume=int(min_volume),
            intents=current_intents,
            bands=raw_config.get("bands", {}),
            bonus_weights=raw_config.get("bonus_weights", {}),
            seed=int(seed),
            k_topics=str(raw_config.get("k_topics", "auto")),
            embedding_model=str(raw_config.get("embedding_model", "all-MiniLM-L6-v2"))
        )
        
        # 2. Process Data
        # Filter Columns
        cols_to_keep = [c for c in cfg.keep_cols if c in raw_df.columns]
        df = raw_df[cols_to_keep].copy()
        
        # Volume Filter
        df["Avg. monthly searches"] = pd.to_numeric(df["Avg. monthly searches"], errors="coerce")
        df = df.dropna(subset=["Avg. monthly searches"])
        df = df[df["Avg. monthly searches"] >= cfg.min_volume].copy()
        
        # Dedupe
        df["keyword_norm"] = df["Keyword"].apply(normalize_keyword)
        df = (
            df.sort_values("Avg. monthly searches", ascending=False)
              .drop_duplicates("keyword_norm")
              .drop(columns=["keyword_norm"])
              .reset_index(drop=True)
        )
        
        # Intent
        df["intent"] = df["Keyword"].apply(lambda x: assign_intent(x, cfg.intents))
        
        # Topic Clustering
        st.info(f"Clustering {len(df)} keywords into topics...")
        keywords = df["Keyword"].fillna("").astype(str).tolist()
        
        # Check if we have enough data
        if not keywords:
            st.error("No keywords remaining after filtering!")
            st.stop()
            
        topics = cluster_topics_semantic(
            keywords,
            seed=cfg.seed,
            k_topics=cfg.k_topics,
            model_name=cfg.embedding_model,
        )
        df["topic"] = topics
        
        # Cluster ID
        df["cluster"] = df["intent"].astype(str) + " | " + df["topic"].astype(str)
        
        # Scoring
        df["score"] = score_rows(df, bands=cfg.bands, bonus_weights=cfg.bonus_weights)
        
        # 3. Display Results
        st.success("Classification Complete!")
        st.dataframe(df)
        
        # 4. Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Output CSV",
            data=csv,
            file_name="classified_keywords.csv",
            mime="text/csv",
        )

