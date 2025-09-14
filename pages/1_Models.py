import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="AI Regression Models", page_icon="📊", layout="wide")

# ---------- CSS Styling ----------
st.markdown("""
<style>
    .main-header {font-size:3rem;color:#4B8BBE;text-align:center;margin-bottom:1rem;text-shadow:0px 2px 4px rgba(0,0,0,0.2);}
    .sub-header {font-size:1.5rem;color:#306998;margin-top:1.5rem;border-bottom:2px solid #FFD43B;padding-bottom:0.5rem;}
    .info-box {background-color:rgba(75,139,190,0.1);padding:1.5rem;border-radius:10px;border-left:5px solid #4B8BBE;margin-bottom:1.5rem;box-shadow:0 4px 6px rgba(0,0,0,0.1);}
    .step-box {background-color:rgba(255,212,59,0.1);padding:1rem;border-radius:8px;margin:0.5rem 0;border-left:3px solid #FFD43B;}
    .uploader-box {background-color:rgba(255,255,255,0.8);padding:2rem;border-radius:12px;text-align:center;border:2px dashed #4B8BBE;margin:1.5rem 0;}
    .success-box {background-color:rgba(52,168,83,0.1);padding:1rem;border-radius:8px;border-left:3px solid #34A853;margin-top:1rem;}
    .emoji {font-size:1.2em;margin-right:0.5rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Function for Training & Predicting ----------
def train_and_predict(model, param_grid):
    if "trained_model" not in st.session_state:
        pipe = Pipeline([
            ("preprocessing", preprocessing),
            ("model", model)
        ])
        with st.spinner("⏳ Training your model, please wait..."):
            trained_model = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=5, cv=5)
            trained_model.fit(data.drop(target, axis=1), data[target])
        st.session_state.trained_model = trained_model
    else:
        trained_model = st.session_state.trained_model

    st.subheader("🔢 Enter input values for prediction")
    input_values = {}
    for col in data.drop(target, axis=1).columns:
        if col in numerical_cols:
            input_values[col] = st.number_input(f"Enter value for {rename_map[col]} ({col})")
        else:
            input_values[col] = st.selectbox(f"Choose value for {rename_map[col]} ({col})", options=data[col].unique())
    test_df = pd.DataFrame([input_values])
    st.write("### ✅ Your Input Data")
    st.write(test_df)

    if st.button("🚀 Predict"):
        prediction = st.session_state.trained_model.predict(test_df)
        st.success(f"✅ Predicted Value: {prediction[0]}")

# ---------- UI ----------
st.title("📌 AI Regression Models Playground")
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="sub-header">🌟 Welcome!</div>', unsafe_allow_html=True)
st.markdown("""
This interactive playground allows you to:
- 🚀 Train powerful regression models
- 📊 Visualize your data with interactive charts
- 🔍 Predict outcomes based on your input
- 🌈 Enjoy a clean, modern interface
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">📋 How to use this tool</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-box"><span class="emoji">1️⃣</span> Upload your dataset (CSV)</div>
<div class="step-box"><span class="emoji">2️⃣</span> Preview your data and select target column</div>
<div class="step-box"><span class="emoji">3️⃣</span> Choose your regression model</div>
<div class="step-box"><span class="emoji">4️⃣</span> Enter input values and get predictions</div>
<div class="step-box"><span class="emoji">5️⃣</span> Explore interactive plots and distributions</div>
""", unsafe_allow_html=True)

st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
st.markdown('<h3>📤 Upload Your CSV Dataset</h3>', unsafe_allow_html=True)
st.markdown('<p>Supported format: CSV</p>', unsafe_allow_html=True)
file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main Logic ----------
if file is not None:
    data = pd.read_csv(file)
    data.reset_index(drop=True, inplace=True)

    # ---------- Rename Columns ----------
    original_cols = data.columns.tolist()
    new_cols = [f"col_{i}" for i in range(len(original_cols))]
    rename_map = dict(zip(new_cols, original_cols))
    data.columns = new_cols
    st.subheader("🛠 Column Mapping (Original → Renamed)")
    st.write(rename_map)

    # ---------- Data Preview ----------
    columns_to_show = st.multiselect("📌 Select columns to display", default=data.columns, options=data.columns)
    num_rows = st.slider("🔍 Select number of rows to show", min_value=5, max_value=1000, step=5)
    st.write(data.loc[:num_rows, columns_to_show])

    # ---------- Target Selection ----------
    st.subheader("🎯 Select target column")
    numerical_target_cols = data.select_dtypes(include="number").columns
    target = st.selectbox("Choose target column", options=numerical_target_cols)

    numerical_cols = data.drop(target, axis=1).select_dtypes(include="number").columns
    categorical_cols = data.drop(target, axis=1).select_dtypes(include="object").columns

    # ---------- Handle missing target values ----------
    if data[target].dtype == "object":
        data[target] = LabelEncoder().fit_transform(SimpleImputer(strategy="most_frequent").fit_transform(data[[target]]))
    else:
        data[target] = SimpleImputer(strategy="mean").fit_transform(data[[target]])

    # ---------- Preprocessing Pipelines ----------
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder())])
    preprocessing = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # ---------- Model Selection ----------
    model_choice = st.selectbox("Select Regression Model", options=[
        "LinearRegression", "ElasticNet", "DecisionTreeRegressor", "RandomForestRegressor",
        "GradientBoostingRegressor", "XGBoost", "SVR", "KNeighborsRegressor"
    ])

    # ---------- Model Training & Prediction ----------
    if model_choice == "LinearRegression":
        if "trained_model" not in st.session_state:
            pipe = Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])
            pipe.fit(data.drop(target, axis=1), data[target])
            st.session_state.trained_model = pipe
        train_and_predict(LinearRegression(), {})
    elif model_choice == "ElasticNet":
        train_and_predict(ElasticNet(), {'model__alpha':[0.01,0.1,1,10],'model__l1_ratio':[0,0.25,0.5,0.75,1],'model__max_iter':[1000,5000]})
    elif model_choice == "DecisionTreeRegressor":
        train_and_predict(DecisionTreeRegressor(), {'model__max_depth':[None,3,5,7],'model__min_samples_split':[2,5,10],'model__min_samples_leaf':[1,2,4]})
    elif model_choice == "RandomForestRegressor":
        train_and_predict(RandomForestRegressor(), {'model__n_estimators':[100,200,500],'model__max_depth':[None,3,5,7],'model__min_samples_split':[2,5,10],'model__min_samples_leaf':[1,2,4]})
    elif model_choice == "GradientBoostingRegressor":
        train_and_predict(GradientBoostingRegressor(), {'model__n_estimators':[100,200,500],'model__learning_rate':[0.01,0.05,0.1],'model__max_depth':[3,4,5]})
    elif model_choice == "XGBoost":
        train_and_predict(XGBRegressor(), {'model__n_estimators':[100,200,300],'model__max_depth':[3,4,5,6],'model__learning_rate':[0.01,0.05,0.1,0.2],'model__subsample':[0.6,0.8,1.0],'model__colsample_bytree':[0.6,0.8,1.0],'model__gamma':[0,0.1,0.3],'model__reg_lambda':[0.1,1,5],'model__reg_alpha':[0,0.1,1]})
    elif model_choice == "SVR":
        train_and_predict(SVR(), {"model__kernel":['linear','rbf','poly'],'model__degree':[2,3,4],"model__C":[0.01,0.1,1,10,100],'model__epsilon':[0.01,0.1,0.5,1.0],'model__gamma':['scale','auto']})
    elif model_choice == "KNeighborsRegressor":
        train_and_predict(KNeighborsRegressor(), {'model__n_neighbors':[3,5,7],'model__weights':['uniform','distance']})

    # ---------- Data Visualization ----------
    st.markdown("## 📊 Data Visualization")
    data_graph = data.copy()
    for col in numerical_cols:
        data_graph[col].fillna(data_graph[col].mean(), inplace=True)
    for col in categorical_cols:
        data_graph[col].fillna(data_graph[col].mode()[0], inplace=True)

    tab1, tab2, tab3 = st.tabs(["📌 Numerical Plots", "📈 Distribution", "📊 Categorical Analysis"])
    with tab1:
        x = st.selectbox("Choose X-axis:", options=numerical_cols, key="tab1_x")
        y = st.selectbox("Choose Y-axis:", options=numerical_cols, key="tab1_y")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.scatter(data_graph, x=x, y=y, title="Scatter Plot", template="plotly_dark"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(data_graph, x=x, y=y, title="Line Plot", template="plotly_dark"), use_container_width=True)

    with tab2:
        col = st.selectbox("Select column for box plot:", options=numerical_cols, key="tab2")
        st.plotly_chart(px.box(data_graph, y=col, title="Box Plot", template="plotly_dark"), use_container_width=True)

    with tab3:
        cat = st.selectbox("Categorical feature:", options=categorical_cols, key="tab3_cat")
        num = st.selectbox("Numerical feature:", options=numerical_cols, key="tab3_num")
        st.plotly_chart(px.histogram(data_graph, x=cat, y=num, title="Categorical vs Numerical", template="plotly_dark"), use_container_width=True)
