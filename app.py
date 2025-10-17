# Core Packages
import streamlit as st
import joblib, os
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression

# Load Model
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """🎥 Movie Revenue Prediction Using Linear Regression"""

    st.set_page_config(page_title="Movie Revenue Predictor", layout="wide")

    st.title("🎬 Prediksi Pendapatan Film Berdasarkan Anggaran")

    html_temp = """
    <div style="background-color:#2F4F4F;padding:10px;border-radius:10px">
    <h3 style="color:white;text-align:center;">Using Simple Linear Regression 🍰</h3>
    </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    menu = ["Predict Revenue", "Visualize Data", "About Regression"]
    choice = st.sidebar.selectbox("Menu", menu)

    # ---- PREDICTION PAGE ----
    if choice == "Predict Revenue":
        st.subheader("🎬 Prediksi Pendapatan Film")

        st.write("Masukkan jumlah anggaran dalam USD:")
        budget = st.number_input("Anggaran ($)", min_value=0, step=1000000, format="%d")

        if st.button("Prediksi"):
            try:
                regressor = load_prediction_model("models/movie_revenue_regression.pkl")
                budget_reshaped = np.array(budget).reshape(-1, 1)
                predicted_revenue = regressor.predict(budget_reshaped)

                st.success(f"Estimasi pendapatan untuk film dengan anggaran ${budget:,.0f} : ${predicted_revenue[0][0]:,.2f}")

            except Exception as e:
                st.error("⚠️ Model not found or invalid file path. Please check your model directory.")
                st.text(e)

    # ---- VISUALIZATION PAGE ----
    elif choice == "Visualize Data":
        st.subheader("📊 Visualisasi Data Film")
        uploaded_file = st.file_uploader("Upload TMDB Dataset (CSV)", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

            st.write("### Preview Dataset")
            st.dataframe(df.head())

            # Scatter plot: Budget vs Revenue
            st.write("### Scatter Plot: Anggaran vs Pendapatan")
            fig = px.scatter(df, x='budget', y='revenue', 
                             size='popularity' if 'popularity' in df.columns else None,
                             color='vote_average' if 'vote_average' in df.columns else None,
                             hover_data=['title'] if 'title' in df.columns else None,
                             title='Budget vs Revenue')
            st.plotly_chart(fig, use_container_width=True)

            # Regression line visualization
            st.write("### Visualisasi Regresi Linear")
            X = df[['budget']]
            y = df[['revenue']]
            model = LinearRegression()
            model.fit(X, y)
            df['predicted'] = model.predict(X)

            fig2 = px.scatter(df, x='budget', y='revenue', title='Regression: Budget vs Revenue (Red Line = Prediction)')
            fig2.add_scatter(x=df['budget'], y=df['predicted'], mode='lines', name='Regression Line', line=dict(color='red'))
            st.plotly_chart(fig2, use_container_width=True)

    # ---- ABOUT REGRESSION PAGE ----
    elif choice == "About Regression":
        st.subheader("📘 Apa Itu Regresi Linier Sederhana?")
        st.write("""
        Regresi Linier Sederhana adalah teknik statistik yang digunakan untuk memodelkan hubungan antara dua variabel numerik yang bersifat kontinu.
        Dalam proyek ini, regresi digunakan untuk memperkirakan **pendapatan (revenue)** film berdasarkan **anggaran produksi (budget)**. Persamaannya secara umum adalah:

        \n**Revenue = a + b × Budget**
        
        - **a (intercept)**: nilai pendapatan dasar ketika budget = 0  
        - **b (koefisien)**: menunjukkan seberapa besar perubahan pendapatan untuk setiap kenaikan $1 pada budget  
        
        Model ini membantu perusahaan produksi film memahami seberapa besar pengaruh anggaran terhadap potensi keuntungan yang bisa diperoleh.
        """)

        st.info("Developed by Nabila Putri using Streamlit and Scikit-Learn for Aplikasi Web with Bu Akhsin 🌸")

# Run App
if __name__ == '__main__':
    main()
