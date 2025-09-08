import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="📊 Train Your Model App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #00bfff;
        --secondary-color: #FFD43B;
        --success-color: #34A853;
    }
    .main-header {
        font-size: 3rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: rgba(0, 191, 255, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin-bottom: 1.5rem;
    }
    .step-box {
        background-color: rgba(255, 212, 59, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--secondary-color);
    }
    .success-box {
        background-color: rgba(52, 168, 83, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid var(--success-color);
        margin-top: 1rem;
    }
    .result-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin: 1rem 0;
    }
    .emoji {
        font-size: 1.2em;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Train Your Model with Your Way</h1>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌟 About this App</div>', unsafe_allow_html=True)
st.markdown("""
Upload your dataset and experiment with different machine learning models for **Regression** and **Classification**.

- 📁 **Upload CSV Data** easily  
- 🎯 **Choose Target Column** for prediction  
- ⚙️ **Select Features** and preprocessing options  
- 📊 **Train & Evaluate** models with metrics  
- 📈 **Visualize** predictions and results  
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">📋 How to use this tool</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-box"><span class="emoji">1️⃣</span> Upload your dataset (CSV)</div>
<div class="step-box"><span class="emoji">2️⃣</span> Select target column and features</div>
<div class="step-box"><span class="emoji">3️⃣</span> Choose Regression or Classification</div>
<div class="step-box"><span class="emoji">4️⃣</span> Pick a model & tune hyperparameters</div>
<div class="step-box"><span class="emoji">5️⃣</span> View metrics & interactive charts</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sub-header">📤 Upload Your Data</div>', unsafe_allow_html=True)
data = st.file_uploader("Upload Your File Here 📁", type="csv", key="file_uploader")
st.caption("Choose a CSV file from your computer to start analyzing your data.")

if data is not None:
    data = pd.read_csv(data).reset_index(drop=True)

    n_row = st.slider("Enter the number of rows to present 📄",
                      min_value=5, max_value=data.shape[0], key="row_slider")
    choosed_column = st.multiselect("Choose the columns to display 📊",
                                    options=data.columns, default=data.columns, key="multi_1")
    st.write(data.loc[0:n_row, choosed_column])

    hated_columns = st.multiselect("Choose the columns you want to drop ❌",
                                   options=data.columns, key="multi_2")
    data.drop(hated_columns, axis="columns", inplace=True)

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("The data after deleting")
    st.write(data.head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Target 🎯")
    tab1, tab2 = st.tabs(["Regression 📈", "Classification 🤖"])

    with tab1:
        numerical_cols = data.select_dtypes(include="number").columns
        if len(numerical_cols) == 0:
            st.warning("⚠️ No Numerical columns found. Regression cannot be performed.")
        else:
            target = st.selectbox("Choose the target column (numeric) 🎯",
                                  options=numerical_cols, key="target_reg")
            sim_target = SimpleImputer(strategy="mean")
            data[target] = sim_target.fit_transform(data[[target]])[:, 0]
            y = data[target]
            X = data.drop(target, axis="columns")

            numerical_columns = X.select_dtypes(include="number").columns
            categorical_columns = X.select_dtypes(include="object").columns

            numerical_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore"))
            ])

            if len(numerical_columns) > 0 and len(categorical_columns) > 0:
                full_transform = ColumnTransformer([
                    ("num", numerical_pipe, numerical_columns),
                    ("cat", categorical_pipe, categorical_columns)
                ])
                preprocissing_data = full_transform.fit_transform(X)
                if hasattr(preprocissing_data, "toarray"):
                    preprocissing_data = preprocissing_data.toarray()
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=full_transform.get_feature_names_out())
            elif len(numerical_columns) > 0:
                preprocissing_data = numerical_pipe.fit_transform(X[numerical_columns])
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=numerical_columns)
            elif len(categorical_columns) > 0:
                preprocissing_data = categorical_pipe.fit_transform(X[categorical_columns]).toarray()
                preprocissing_data = pd.DataFrame(preprocissing_data, columns=categorical_pipe.get_feature_names_out())
            else:
                st.warning("The columns are not enough to train the data")
                preprocissing_data = None

            if preprocissing_data is not None:
                model_choosed = st.selectbox("Choose regression model 🤖",
                                             ["Linear_Regression", "Random_Forest", "XGBOOST_reg", "Gradient Boost reg"])
                ratio = st.slider("Choose test data percentage 📊", 5, 95, key="split_ratio_reg")
                train_data, test_data, train_target, test_target = train_test_split(
                    preprocissing_data, y, test_size=ratio/100, shuffle=True)

                if model_choosed == "Linear_Regression":
                    model = LinearRegression()
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)

                elif model_choosed == "Gradient Boost reg":
                    n_estimators = st.number_input("n_estimators", 50, 1000, 100, step=50)
                    learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)
                    max_depth = st.selectbox("max_depth", [2, 3, 5, 7, 10])
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42
                    )
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)

                elif model_choosed == "XGBOOST_reg":
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)

                elif model_choosed == "Random_Forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(train_data, train_target)
                    predict_values = model.predict(test_data)

                choosed_accuracy = st.selectbox("Choose regression metric 📈",
                                                ["mean_absolute_error", "mean_squared_error", "r2_score"])
                if choosed_accuracy == "mean_absolute_error":
                    st.write("MAE:", mean_absolute_error(test_target, predict_values))
                elif choosed_accuracy == "mean_squared_error":
                    st.write("MSE:", mean_squared_error(test_target, predict_values))
                else:
                    st.write("R2 Score:", r2_score(test_target, predict_values))

                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=predict_values, mode="lines+markers", name="Predictions"))
                fig.add_trace(go.Scatter(y=test_target, mode="lines+markers", name="Actual"))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        categorical_cols = data.select_dtypes(include="object").columns
        if len(categorical_cols) == 0:
            st.warning("⚠️ No categorical columns found. Classification cannot be performed.")
        else:
            target = st.selectbox("Choose the target column (categorical) 🎯",
                                  options=categorical_cols, key="target_clf")
            sim_target = SimpleImputer(strategy="most_frequent")
            data[target] = sim_target.fit_transform(data[[target]])[:, 0]
            y = data[target]

            if y.nunique() > 20:
                st.error("❌ Target has too many unique values.")
            else:
                X = data.drop(target, axis="columns")
                numerical_columns = X.select_dtypes(include="number").columns
                categorical_columns = X.select_dtypes(include="object").columns

                numerical_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ])
                categorical_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore"))
                ])

                if len(numerical_columns) > 0 and len(categorical_columns) > 0:
                    full_transform = ColumnTransformer([
                        ("num", numerical_pipe, numerical_columns),
                        ("cat", categorical_pipe, categorical_columns)
                    ])
                    preprocissing_data = full_transform.fit_transform(X)
                    if hasattr(preprocissing_data, "toarray"):
                        preprocissing_data = preprocissing_data.toarray()
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=full_transform.get_feature_names_out())
                elif len(numerical_columns) > 0:
                    preprocissing_data = numerical_pipe.fit_transform(X[numerical_columns])
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=numerical_columns)
                elif len(categorical_columns) > 0:
                    preprocissing_data = categorical_pipe.fit_transform(X[categorical_columns]).toarray()
                    preprocissing_data = pd.DataFrame(preprocissing_data, columns=categorical_pipe.get_feature_names_out())
                else:
                    st.warning("The columns are not enough to train the data")
                    preprocissing_data = None

                if preprocissing_data is not None:
                    model_choosed = st.selectbox("Choose classification model 🤖",
                                                 ["Logistic_Regression", "Random_Forest_Classifier", "XGBOOST_Classifier"])
                    ratio = st.slider("Choose test data percentage 📊", 5, 95, key="split_ratio_clf")
                    train_data, test_data, train_target, test_target = train_test_split(
                        preprocissing_data, y, test_size=ratio/100, shuffle=True)

                    if model_choosed == "Logistic_Regression":
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(max_iter=1000)
                        model.fit(train_data, train_target)
                        predict_values = model.predict(test_data)

                    elif model_choosed == "Random_Forest_Classifier":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(train_data, train_target)
                        predict_values = model.predict(test_data)

                    elif model_choosed == "XGBOOST_Classifier":
                        from xgboost import XGBClassifier
                        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
                        model.fit(train_data, train_target)
                        predict_values = model.predict(test_data)

                    from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
                    choosed_accuracy = st.multiselect("Choose classification metrics 📊",
                                                      ["accuracy_score", "f1_score", "precision_score", "confusion_matrix"])
                    if choosed_accuracy:
                        for metric in choosed_accuracy:
                            if metric == "accuracy_score":
                                st.write("Accuracy:", accuracy_score(test_target, predict_values))
                            elif metric == "f1_score":
                                st.write("F1 Score:", f1_score(test_target, predict_values, average="weighted"))
                            elif metric == "precision_score":
                                st.write("Precision:", precision_score(test_target, predict_values, average="weighted"))
                            elif metric == "confusion_matrix":
                                cm = confusion_matrix(test_target, predict_values)
                                fig = px.imshow(
                                    cm,
                                    x=[f"Pred {cls}" for cls in np.unique(test_target)],
                                    y=[f"True {cls}" for cls in np.unique(test_target)],
                                    text_auto=True,
                                    color_continuous_scale="Blues"
                                )
                                st.plotly_chart(fig)

st.markdown('<div class="success-box">', unsafe_allow_html=True)
st.markdown("""
<h3>✅ Ready to Train!</h3>
<p>Upload your data and start training your machine learning models now 🚀</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


