import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load sample dataset
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# App Title
st.title("📊 Univariate and Multivariate Data Analysis")

# Sidebar options
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Univariate", "Multivariate"])

# ----------------- Univariate Analysis -----------------
if analysis_type == "Univariate":
    st.header("Univariate Analysis")
    col = st.selectbox("Select a numerical column", df.select_dtypes(include="number").columns)

    st.subheader(f"Descriptive Statistics for {col}")
    st.write(df[col].describe())

    # Histogram
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

# ----------------- Multivariate Analysis -----------------
elif analysis_type == "Multivariate":
    st.header("Multivariate Analysis")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot
    st.subheader("Scatter Matrix (Pairplot)")
    st.text("This may take a few seconds...")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

    # PCA
    st.subheader("Principal Component Analysis (PCA)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df[numeric_cols])
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["species"] = df["species"]

    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="species", ax=ax)
    st.pyplot(fig)
