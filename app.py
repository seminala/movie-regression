# Core Packages
import streamlit as st
import joblib, os
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Model
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """🎥 Movie Revenue Prediction Using Linear Regression"""

    st.set_page_config(page_title="Movie Revenue Predictor", layout="wide")
    st.title("🎬 Prediksi Pendapatan Film Berdasarkan Anggaran dan Faktor Lainnya")

    html_temp = """
    <div style="background-color:#2F4F4F;padding:10px;border-radius:10px">
    <h3 style="color:white;text-align:center;">Using Linear Regression Models 🍰</h3>
    </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    menu = ["Predict Revenue", "Visualize Data", "About Regression"]
    choice = st.sidebar.selectbox("Menu", menu)

    # ---- PREDICTION PAGE ----
    if choice == "Predict Revenue":
        st.subheader("🎬 Prediksi Pendapatan Film")

        st.write("Masukkan parameter film untuk memprediksi pendapatan:")
        budget = st.number_input("Anggaran Produksi ($)", min_value=0, step=1_000_000, format="%d")
        popularity = st.number_input("Popularitas", min_value=0.0, step=0.1)
        vote_count = st.number_input("Jumlah Vote", min_value=0, step=10)

        if st.button("Prediksi"):
            try:
                # Load model (ensure it's trained with multiple features)
                regressor = load_prediction_model("models/movie_revenue_regression.pkl")

                # Prepare input
                example_values = np.array([[budget, popularity, vote_count]])
                predicted_revenue = regressor.predict(example_values)[0][0]

                st.success(f"🎬 Estimasi pendapatan untuk film dengan:")
                st.info(f"""
                💰 **Anggaran:** ${budget:,.0f}  
                🌟 **Popularitas:** {popularity}  
                🗳️ **Jumlah Vote:** {vote_count}  
                
                📈 **Estimasi Pendapatan:** ${predicted_revenue:,.2f}
                """)

            except Exception as e:
                st.error("⚠️ Model belum ditemukan atau tidak sesuai. Silakan cek direktori model Anda.")
                st.text(e)

    # ---- VISUALIZATION PAGE ----
    elif choice == "Visualize Data":
        st.subheader("📊 Visualisasi Data Film")
        uploaded_file = st.file_uploader("Upload TMDB Dataset (CSV)", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # Cleaning data
            df = df[['vote_count', 'popularity', 'budget', 'revenue', 'title']].copy()
            df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
            df = df.dropna(subset=['vote_count', 'popularity', 'budget', 'revenue'])
            df.reset_index(drop=True, inplace=True)

            st.write("### Preview Dataset")
            st.dataframe(df.head())

            # Correlation heatmap
            st.write("### 🔥 Heatmap Korelasi")
            corr = df[['vote_count', 'popularity', 'budget', 'revenue']].corr()

            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # Scatter plot
            st.write("### Scatter Plot: Budget vs Revenue")
            fig2 = px.scatter(df, x='budget', y='revenue',
                            size='popularity', color='vote_count',
                            hover_data=['title'],
                            title='Budget vs Revenue')
            st.plotly_chart(fig2, use_container_width=True)

            # Regression visualization
            st.write("### Visualisasi Regresi Linear")
            X = df[['budget', 'popularity', 'vote_count']]
            y = df[['revenue']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**MAE:** {mae:,.2f}")
            st.write(f"**RMSE:** {rmse:,.2f}")

            # Predicted vs Actual Plot
            st.write("### Predicted vs Actual Revenue")
            fig3 = px.scatter(x=y_test['revenue'], y=y_pred.flatten(),
                              labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                              title='Predicted vs Actual Revenue')
            fig3.add_shape(type='line', x0=y_test['revenue'].min(), y0=y_test['revenue'].min(),
                           x1=y_test['revenue'].max(), y1=y_test['revenue'].max(),
                           line=dict(color='red', dash='dash'))
            st.plotly_chart(fig3, use_container_width=True)

    # ---- ABOUT REGRESSION PAGE ----
    elif choice == "About Regression":
        st.subheader("📘 Apa Itu Regresi Linier?")
        st.write("""
        Regresi Linier digunakan untuk memprediksi nilai kontinu (seperti pendapatan film)
        berdasarkan satu atau lebih variabel independen seperti **budget**, **popularitas**, dan **vote_count**.

        - **Regresi Linier Sederhana:** hanya menggunakan satu variabel (misalnya budget → revenue)
        - **Regresi Linier Berganda:** menggunakan lebih dari satu variabel (budget, popularitas, vote_count → revenue)

        Model ini membantu perusahaan produksi memahami bagaimana faktor-faktor tersebut
        memengaruhi potensi pendapatan film.
        """)

        st.info("Developed by **Nabila Putri** using Streamlit and Scikit-Learn\nfor **Aplikasi Web with Bu Akhsin 🌸**\n© 2025 Universitas Negeri Yogyakarta")

# Run App
if __name__ == '__main__':
    main()
