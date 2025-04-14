import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import zscore
from impyute.imputation.cs import mice

st.set_page_config(page_title="🎓 Student Performance Classifier", layout="wide")
st.title("🎓 Student Performance Classifier")
st.write("Upload your student CSV file and explore predictions for final grades.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('%', 'Perc') \
        .str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.title()

    grade_map = {"A": 95, "B": 85, "C": 75, "D": 65}
    df["Final_Grade_Numeric"] = df["Final_Grade"].map(grade_map)

    st.sidebar.title("🔧 Filtrare")
    if "Gender" in df.columns:
        gender_filter = st.sidebar.selectbox("Filtru după gen:", ["Toate"] + sorted(df["Gender"].dropna().unique()))
        if gender_filter != "Toate":
            df = df[df["Gender"] == gender_filter]

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Date Brute", "📈 Analiză Univariată", "📊 Corelații & Grupări", "🎯 Vizualizări Avansate"])

    with tab1:
        st.subheader("📋 Date brute")
        if st.checkbox("Afișează primele 10 rânduri din dataset"):
            st.dataframe(df.head(10))

    with tab2:
        st.markdown("## 📌 Analiză univariată - variabile categorice")
        cat_cols = ["Gender", "Preferred_Learning_Style", "Participation_In_Discussions", 
                    "Use_Of_Educational_Tech", "Self_Reported_Stress_Level", "Final_Grade"]
        available_cats = [col for col in cat_cols if col in df.columns]
        if available_cats:
            selected_cat = st.selectbox("Alege o variabilă categorică:", available_cats)
            left, center, right = st.columns([1, 4, 1])
            with center:
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, x=selected_cat, order=df[selected_cat].value_counts().index, ax=ax1)
                plt.xticks(rotation=45)
                st.pyplot(fig1)

        st.markdown("## 📌 Analiză univariată - variabile numerice")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Alege o variabilă numerică:", num_cols)
            left, center, right = st.columns([1, 4, 1])
            with center:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.histplot(df[selected_num], kde=True, ax=ax2)
                st.pyplot(fig2)

    with tab3:
        st.markdown("## 🔗 Matrice de corelație")
        if st.checkbox("Afișează matricea de corelație"):
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

        st.markdown("## 📊 Media scorurilor finale pe categorii")
        cat_options = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() <= 10]
        if "Final_Grade" in cat_options:
            cat_options.remove("Final_Grade")
        selected_group_col = st.selectbox("Alege o variabilă categorică pentru grupare:", cat_options)
        if selected_group_col and "Final_Grade_Numeric" in df.columns:
            grouped_means = df.groupby(selected_group_col)["Final_Grade_Numeric"].mean().dropna()
            if not grouped_means.empty:
                st.bar_chart(grouped_means)
            else:
                st.warning("Nu s-au putut calcula mediile.")

    with tab4:
        st.markdown("## 🎯 Analiză vizuală avansată a scorurilor finale")
        if cat_options:
            selected_plot_col = st.selectbox("Alege o variabilă pentru distribuție detaliată:", cat_options, key="dist_col")
            left, center, right = st.columns([1, 4, 1])
            with center:
                st.markdown("### 📦 Boxplot - Distribuția scorurilor finale")
                fig_box = px.box(df, x=selected_plot_col, y="Final_Grade_Numeric", points="all", color=selected_plot_col, width=700, height=400)
                st.plotly_chart(fig_box, use_container_width=False)

                st.markdown("### 📊 Histogramă suprapusă - Scoruri finale")
                fig_hist = px.histogram(df, x="Final_Grade_Numeric", color=selected_plot_col, barmode="overlay", opacity=0.6, width=700, height=400)
                st.plotly_chart(fig_hist, use_container_width=False)

                st.markdown("### 🎻 Violin plot - Formă și variație")
                fig_violin = px.violin(df, y="Final_Grade_Numeric", x=selected_plot_col, box=True, points="all", color=selected_plot_col, width=700, height=400)
                st.plotly_chart(fig_violin, use_container_width=False)

        st.markdown("## 📊 Comparație între stiluri de învățare pe variabile numerice")
        important_vars = [
            "Study_Hours_Per_Week",
            "Exam_Score_Perc",
            "Attendance_Rate_Perc",
            "Self_Reported_Stress_Level"
        ]
        available_vars = [col for col in important_vars if col in df.columns]
        if "Preferred_Learning_Style" in df.columns and available_vars:
            for var in available_vars:
                st.markdown(f"### {var.replace('_', ' ')}")
                left, center, right = st.columns([1, 4, 1])
                with center:
                    fig = px.box(df, x="Preferred_Learning_Style", y=var, color="Preferred_Learning_Style", points="all", width=700, height=400)
                    st.plotly_chart(fig, use_container_width=False)

else:
    st.info("🔼 Te rog încarcă un fișier CSV cu separator `;`.")