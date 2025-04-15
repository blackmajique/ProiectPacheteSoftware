# Toate importurile (la fel)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px

from impyute.imputation.cs import mice
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import f_oneway

st.set_page_config(page_title="📊 Analiză Performanță Studenți", layout="wide")
st.title("📚 Analiză Vizuală a Performanței Studenților")

uploaded_file = st.file_uploader("📁 Încarcă fișierul CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('%', 'Perc') \
        .str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.title()

    if "Total_Score" in df.columns:
        threshold = df["Total_Score"].median()
        df["Target_Binar"] = (df["Total_Score"] >= threshold).astype(int)
        st.markdown(f"🎯 Target numeric: `Total_Score` | Target binar: `Target_Binar` (mediana = {threshold:.2f})")

    # === Sidebar ===
    with st.sidebar.expander("🧩 Tratare valori lipsă"):
        missing_value_method = st.radio("Metodă:", 
            ("fillna - zero or unknown", "fillna - mean or mode", "bfill", "ffill", "interpolate", "mice"))

    with st.sidebar.expander("🔤 Codificare categorice"):
        encode_method = st.radio("Metodă:", ("Fără codificare", "Label Encoding"))

    with st.sidebar.expander("📏 Scalare"):
        scaler_method = st.radio("Scalare:", ("Fără scalare", "Standard", "MinMax", "Robust"))

    with st.sidebar.expander("🚨 Outlieri"):
        outlier_method = st.radio("Metodă:", ("Nimic", "Z-Score < 3", "Quantile 1%-99%"))
        selected_outlier_col = st.selectbox("Coloană numerică:", df.select_dtypes(include=np.number).columns)

    # === Codificare ===
    if encode_method == "Label Encoding":
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]

    if scaler_method != "Fără scalare":
        scaler = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}[scaler_method]
        df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))

    if missing_value_method == "fillna - zero or unknown":
        df = df.fillna(0)
    elif missing_value_method == "fillna - mean or mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean() if df[col].dtype != "object" else df[col].mode()[0])
    elif missing_value_method == "bfill":
        df = df.fillna(method="bfill")
    elif missing_value_method == "ffill":
        df = df.fillna(method="ffill").fillna(method="bfill")
    elif missing_value_method == "interpolate":
        df = df.interpolate(method="linear", limit_direction="both")
    elif missing_value_method == "mice":
        np.float = float
        numeric_df = df.select_dtypes(include=[np.number])
        imputed = mice(numeric_df.values)
        df[numeric_df.columns] = pd.DataFrame(imputed, columns=numeric_df.columns)

    if outlier_method == "Z-Score < 3":
        z = np.abs((df[selected_outlier_col] - df[selected_outlier_col].mean()) / df[selected_outlier_col].std())
        df = df[z < 3]
    elif outlier_method == "Quantile 1%-99%":
        q1 = df[selected_outlier_col].quantile(0.01)
        q99 = df[selected_outlier_col].quantile(0.99)
        df = df[(df[selected_outlier_col] >= q1) & (df[selected_outlier_col] <= q99)]

    df_original = df.copy()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Date brute", "📊 Clasice", "📈 Interactive", "📉 Modele"])

    with tab1:
        st.dataframe(df.head(100))
        st.subheader("📌 Valori lipsă")
        missing_pct = df.isnull().mean() * 100
        if missing_pct.any():
            st.table(missing_pct[missing_pct > 0])
        else:
            st.success("✅ Fără valori lipsă.")

        st.subheader("🌡️ Heatmap lipsuri")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() <= 20]

        viz_type = st.radio("🔍 Vizualizare:", ["Categorică", "Numerică", "Corelații", "Medii"], horizontal=True)
        if viz_type == "Categorică":
            selected_cat = st.selectbox("Coloană:", cat_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x=selected_cat, ax=ax)
            st.pyplot(fig)
        elif viz_type == "Numerică":
            selected_num = st.selectbox("Coloană:", num_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df[selected_num], kde=True, ax=ax)
            st.pyplot(fig)
        elif viz_type == "Corelații":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        elif viz_type == "Medii":
            selected_group = st.selectbox("Grupare după:", cat_cols)
            if selected_group:
                means = df.groupby(selected_group)["Total_Score"].mean()
                st.bar_chart(means)

    with tab3:
        st.subheader("🎯 Vizualizări interactive")

        cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() <= 20]
        plot_cats = [col for col in cat_cols if df[col].nunique() <= 20]
        selected_plot_cat = st.selectbox("🔽 Alege o coloană categorică:", plot_cats)
        selected_plot_num = st.selectbox("📈 Alege o coloană numerică:", df.select_dtypes(include=np.number).columns)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📦 Boxplot + Stripplot")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=df, x=selected_plot_cat, y=selected_plot_num, ax=ax, palette="Set2")
            sns.stripplot(data=df, x=selected_plot_cat, y=selected_plot_num, color="black", alpha=0.5, jitter=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("🎻 Violin plot")
            fig_violin = px.violin(df, x=selected_plot_cat, y=selected_plot_num, color=selected_plot_cat, box=True, points="all")
            fig_violin.update_layout(width=500, height=300)
            st.plotly_chart(fig_violin, use_container_width=False)

        try:
            groups = [df[df[selected_plot_cat] == val][selected_plot_num] for val in df[selected_plot_cat].dropna().unique()]
            stat, p = f_oneway(*groups)
            st.markdown(f"📊 **Test ANOVA:** `F = {stat:.2f}`, `p = {p:.4f}`")
            if p < 0.05:
                st.success("✅ Diferențele între grupuri sunt semnificative statistic.")
            else:
                st.info("ℹ️ Diferențele între grupuri NU sunt semnificative statistic.")
        except Exception as e:
            st.warning(f"⚠️ Eroare ANOVA: {e}")

    with tab4:
        df_model = df_original.copy()

        st.subheader("📈 Regresie liniară (Total_Score)")
        reg_cols = st.multiselect("Predictori:", [col for col in df_model.columns if col != "Total_Score"])
        if reg_cols:
            X = df_model[reg_cols].select_dtypes(include=[np.number])
            y = df_model["Total_Score"]
            X = sm.add_constant(X)
            if X.shape[1] > 1:
                model = sm.OLS(y, X).fit()
                st.text(model.summary())
            else:
                st.warning("Alege minim 1 coloană numerică.")

        st.subheader("🔍 Logistic Regression")
        target = "Target_Binar"
        feat_cols = st.multiselect("Predictori:", [col for col in df_model.columns if col != target], key="logreg")
        if feat_cols:
            X = df_model[feat_cols]
            y = df_model[target]
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = pd.factorize(X[col])[0]

            if st.checkbox("🎯 SelectKBest"):
                selector = SelectKBest(f_regression, k=min(5, X.shape[1]))
                X_new = selector.fit_transform(X, y)
                selected = X.columns[selector.get_support()]
                X = pd.DataFrame(X_new, columns=selected)
                st.success(f"Selectate: {', '.join(selected)}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=100, solver="liblinear")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.markdown("📊 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.markdown("🧾 Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

            st.markdown(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            if len(np.unique(y_test)) == 2:
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                ax.plot([0, 1], [0, 1], "--", label="Random")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)

else:
    st.info("🔼 Încarcă fișierul CSV (cu separator `;`) pentru a începe analiza.")
