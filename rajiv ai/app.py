
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Load/Prepare Data
# -----------------------------
@st.cache_data
def load_and_prepare_data():
  # <-- replace with your dataset
    df = pd.read_csv(r"C:\Users\rajuy\OneDrive\Documents\FAI MICRO PROJECT\rajiv ai\rajiv ai\customers-100.csv")


    # Copy for displaying later
    df_full = df.copy()
    
    # Select useful features
    feature_cols = ['Company', 'City', 'Country']
    df_train = df[feature_cols].copy()
    
    # Encode categorical data
    le_dict = {}
    for col in feature_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        le_dict[col] = le
    
    # Train Nearest Neighbors model
    model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    model.fit(df_train)
    
    return model, le_dict, feature_cols, df_full, df_train

model, le_dict, feature_cols, df_full, df_train = load_and_prepare_data()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Job Recommendation", page_icon="💼", layout="wide")
st.title("💼 AI Job Recommendation System")
st.markdown("Get job/company recommendations based on your dataset values.")

tab1, tab2 = st.tabs(["Manual Input", "Upload Dataset"])

# -----------------------------
# Tab 1: Manual Input
# -----------------------------
with tab1:
    with st.form("manual_form"):
        company = st.selectbox("Preferred Company", sorted(df_full['Company'].unique()))
        city = st.selectbox("Preferred City", sorted(df_full['City'].unique()))
        country = st.selectbox("Preferred Country", sorted(df_full['Country'].unique()))
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        # Encode input
        input_data = [company, city, country]
        input_encoded = []
        for i, col in enumerate(feature_cols):
            le = le_dict[col]
            value = input_data[i]
            try:
                encoded = le.transform([value])[0]
            except:
                st.error(f"Value '{value}' not recognized for column '{col}'")
                encoded = 0
            input_encoded.append(encoded)
        
        input_array = np.array(input_encoded).reshape(1, -1)
        
        # Find similar jobs
        distances, indices = model.kneighbors(input_array)
        recommended_jobs = df_full.iloc[indices[0]]
        
        st.subheader("📌 Your Profile")
        profile_df = pd.DataFrame({
            "Feature": ["Company", "City", "Country"],
            "Input": input_data
        })
        st.table(profile_df)
        
        st.subheader("💼 Recommended Jobs/Companies")
        st.dataframe(recommended_jobs[['Company', 'City', 'Country']])

# -----------------------------
# Tab 2: Upload Dataset
# -----------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV file with job seeker profiles", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview:")
        st.dataframe(df.head())
        
        # Encode categorical features
        for col in feature_cols:
            if col in df.columns:
                le = le_dict[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        # Predict neighbors
        X_data = df[feature_cols]
        distances, indices = model.kneighbors(X_data)
        
        # Assign top company for each profile
        df['Recommended_Company'] = [
            df_full.iloc[idx]['Company'] for idx in indices[:,0]
        ]
        
        st.subheader("📊 Prediction Results")
        st.dataframe(df.head(20))
        
        st.download_button(
            label="📥 Download Job Recommendations as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="job_recommendations.csv",
            mime="text/csv"
        )
