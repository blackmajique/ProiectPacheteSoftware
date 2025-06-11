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

st.set_page_config(page_title="Analiza performantei studentilor", layout="wide")
st.title("Analiza vizuala a performantei studentilor")

uploaded_file = st.file_uploader("Incarca fisierul CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")

    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('%', 'Perc') \
        .str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.title()

    ignore_cols = {"student_id", "first_name", "last_name", "email", "unnamed:_0"}
    df = df[[col for col in df.columns if col.lower() not in ignore_cols]]

    df = df.loc[:, df.nunique() > 1]  

    df_raw = df.copy()

    if "Total_Score" in df.columns:
        threshold = df["Total_Score"].median()
        df["Target_Binar"] = (df["Total_Score"] >= threshold).astype(int)
        st.markdown(f"Target numeric: `Total_Score` | Target binar: `Target_Binar` (mediana = {threshold:.2f})")

    # Sidebar
    with st.sidebar.expander("Tratarea valorilor lipsa"):
        missing_value_method = st.radio("Alege metoda:", 
            ("fillna - zero or unknown", "fillna - mean or mode", "bfill", "ffill", "interpolate", "mice"))

    with st.sidebar.expander("Transformarea variabilelor categorice"):
        encode_method = st.radio("Alege metoda:", ("Fara codificare", "Label Encoding"))

    with st.sidebar.expander("Normalizarea/Standardizarea variabile numerice"):
        scaler_method = st.radio("Alege metoda:", ("Fara scalare", "Standard", "MinMax", "Robust"))

    with st.sidebar.expander("Analiza valorilor extreme"):
        outlier_method = st.radio("Alege metoda:", ("Nimic", "Z-Score < 3", "Quantile 1%-99%"))
        selected_outlier_col = st.selectbox("Coloana numerica:", df.select_dtypes(include=np.number).columns)

    # Codificare
    if encode_method == "Label Encoding":
        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.factorize(df[col])[0]

    # Scalare
    if scaler_method != "Fara scalare":
        scaler = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}[scaler_method]
        df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))

    # Tratarea valorilor lipsa
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

    # Outliers
    if outlier_method == "Z-Score < 3":
        z = np.abs((df[selected_outlier_col] - df[selected_outlier_col].mean()) / df[selected_outlier_col].std())
        df = df[z < 3]
    elif outlier_method == "Quantile 1%-99%":
        q1 = df[selected_outlier_col].quantile(0.01)
        q99 = df[selected_outlier_col].quantile(0.99)
        df = df[(df[selected_outlier_col] >= q1) & (df[selected_outlier_col] <= q99)]

    df_original = df.copy()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Previzualizarea datelor si a valorilor lipsa",
        "Analiza descriptiva: categorice, numerice, corelatii",
        "Relatii intre variabile si test ANOVA",
        "Regresie liniara si logistica"
    ])

    with tab1:
        st.dataframe(df.head(100))
        st.subheader("Valori lipsa")
        missing_pct = df_raw.isnull().mean() * 100
        if missing_pct.any():
            st.table(missing_pct[missing_pct > 0])
        else:
            st.success("Fara valori lipsa. :D")

        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    # Tab 2
    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() <= 20]

        viz_type = st.radio("Vizualizare:", ["Categorica", "Numerica", "Corelatii", "Medii"], horizontal=True)

        if viz_type == "Categorica":
            categorical_cols = ["Gender", "Department", "Grade", "Extracurricular_Activities", "Internet_Access_At_Home", "Family_Income_Level"]  # adaptează după fișier

            selected_cat = st.selectbox("Coloană:", categorical_cols)

            fig, ax = plt.subplots(figsize=(4, 3))

            if selected_cat in num_cols:
                x_vals = df[selected_cat].dropna().astype(int).astype(str)
            else:
                x_vals = df[selected_cat].astype(str)

            sns.countplot(x=x_vals, ax=ax)
            ax.set_xlabel(selected_cat.replace("_", " "))
            ax.tick_params(axis='x', rotation=45)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig, use_container_width=False)

        elif viz_type == "Numerica":
            selected_num = st.selectbox("Coloană:", num_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df[selected_num], kde=True, ax=ax)
            ax.set_xlabel(selected_num.replace("_", " "))
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig, use_container_width=False)

        elif viz_type == "Corelatii":
            fig, ax = plt.subplots(figsize=(10, 8)) 

            corr_matrix = df[num_cols].corr().round(2) 
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1, vmax=1,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                cbar_kws={"shrink": 0.75},
                ax=ax
            )
            ax.set_title("Matricea corelatiilor intre variabile numerice", fontsize=12)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig, use_container_width=False)


        elif viz_type == "Medii":
            selected_group = st.selectbox("Grupare dupa:", cat_cols)
            if selected_group:
                means = df.groupby(selected_group)["Total_Score"].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=means.index, y=means.values, palette="Blues_d", ax=ax)
                ax.set_xlabel(selected_group.replace("_", " "))
                ax.set_ylabel("Media Total_Score")
                ax.set_title("Compararea mediilor Total_Score intre categorii")
                plt.xticks(rotation=45, ha="right", fontsize=8)
                st.pyplot(fig)


    with tab3:
        st.subheader("Relatii intre variabile si test ANOVA")

        cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() <= 20]
        plot_cats = [col for col in cat_cols if df[col].nunique() <= 20]
        selected_plot_cat = st.selectbox("Alege o coloana CATEGORICA:", plot_cats)
        selected_plot_num = st.selectbox("Alege o coloana NUMERICA:", df.select_dtypes(include=np.number).columns)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Boxplot**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=df, x=selected_plot_cat, y=selected_plot_num, palette="Set2", ax=ax)
            ax.set_xlabel(selected_plot_cat.replace("_", " "))
            ax.set_ylabel(selected_plot_num.replace("_", " "))
            ax.set_title("Distributia scorurilor per categorie")
            st.pyplot(fig)

        with col2:
            st.markdown("**Violin Plot**")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.violinplot(data=df, x=selected_plot_cat, y=selected_plot_num, palette="Set2", ax=ax)
            ax.set_xlabel(selected_plot_cat.replace("_", " "))
            ax.set_ylabel(selected_plot_num.replace("_", " "))
            ax.set_title("Distributia scorurilor per categorie")
            st.pyplot(fig)

        try:
            groups = [df[df[selected_plot_cat] == val][selected_plot_num] for val in df[selected_plot_cat].dropna().unique()]
            stat, p = f_oneway(*groups)
            st.markdown(f"**Test ANOVA:** `F = {stat:.2f}`, `p = {p:.4f}`")
            if p < 0.05:
                st.success("Diferentele intre grupuri sunt semnificative statistic.")
            else:
                st.info("Diferentele intre grupuri NU sunt semnificative statistic.")
        except Exception as e:
            st.warning(f"⚠️ Eroare ANOVA: {e}")



    with tab4:
        df_model = df_original.copy()

        st.subheader("Regresie liniara (Total_Score)")
        reg_cols = st.multiselect("Predictori:", [col for col in df_model.columns if col != "Total_Score"])
        if reg_cols:
            X = df_model[reg_cols].select_dtypes(include=[np.number])
            y = df_model["Total_Score"]
            X = sm.add_constant(X)
            if X.shape[1] > 1:
                model = sm.OLS(y, X).fit()

                st.markdown("Rezumat regresie")
                summary_df = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
                st.dataframe(summary_df)

            else:
                st.warning("Alege minim 1 coloana numerica.")

        st.subheader("Regresie Logistica")
        target = "Target_Binar"
        feat_cols = st.multiselect("Predictori:", [col for col in df_model.columns if col != target], key="logreg")
        if feat_cols:
            X = df_model[feat_cols]
            y = df_model[target]
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = pd.factorize(X[col])[0]

            if st.checkbox("SelectKBest"):
                selector = SelectKBest(f_regression, k=min(5, X.shape[1]))
                X_new = selector.fit_transform(X, y)
                selected = X.columns[selector.get_support()]
                X = pd.DataFrame(X_new, columns=selected)
                st.success(f"Selectate: {', '.join(selected)}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=100, solver="liblinear")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.markdown("### Matricea de confuzie")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig, use_container_width=False)

            st.markdown("### Raport de clasificare")
            st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

            st.markdown(f"### Acuratete: `{accuracy_score(y_test, y_pred):.2f}`")

            if len(np.unique(y_test)) == 2:
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc = roc_auc_score(y_test, probs)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                ax.plot([0, 1], [0, 1], "--", label="Random")
                ax.set_title("ROC Curve")
                ax.legend()

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.pyplot(fig, use_container_width=False)


else:
    st.info("Incarca fisierul CSV pentru a incepe analiza. :)")
