import streamlit as st

# -----------------------------
# FORCE UI FIRST (ANTI-SPINNER)
# -----------------------------
st.set_page_config(page_title="Masters Decision Support System", layout="centered")
st.title("🎓 Masters Decision Support System")
st.write("✅ App is starting...")

# -----------------------------
# SAFE IMPORTS
# -----------------------------
try:
    from streamlit_option_menu import option_menu
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
except Exception as e:
    st.error("❌ Import Error")
    st.exception(e)
    st.stop()

# -----------------------------
# SAFE DATA LOAD
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")

try:
    model_df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("❌ Dataset loading failed")
    st.exception(e)
    st.stop()

# -----------------------------
# ENCODING
# -----------------------------
le_gate = LabelEncoder()
model_df["GATE_Score"] = le_gate.fit_transform(model_df["GATE_Score"])

le_master = LabelEncoder()
model_df["Should_Do_Masters"] = le_master.fit_transform(
    model_df["Should_Do_Masters"]
)

if le_master.transform(["Yes"])[0] != 1:
    model_df["Should_Do_Masters"] = 1 - model_df["Should_Do_Masters"]
    le_master.fit(["No", "Yes"])

gate_label_map = dict(
    zip(le_gate.transform(le_gate.classes_), le_gate.classes_)
)
master_label_map = {0: "No", 1: "Yes"}

# -----------------------------
# MODEL (CACHED)
# -----------------------------
X = model_df[["GATE_Score", "Salary"]]
y = model_df["Should_Do_Masters"]

@st.cache_resource
def train_model(X, y):
    x_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_train, model.predict(x_train))
    return model, acc

try:
    model, accuracy = train_model(X, y)
except Exception as e:
    st.error("❌ Model training failed")
    st.exception(e)
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "About Model", "Dataset", "Prediction Model", "Graph"],
        icons=["house", "info-circle", "table", "cpu", "bar-chart"],
        default_index=0
    )

    st.markdown("### 📈 Model Performance")
    st.write(f"Accuracy: **{accuracy*100:.2f}%**")

# -----------------------------
# PAGES
# -----------------------------
if selected == "Home":
    st.subheader("👋 Welcome!")
    st.markdown("""
    This application helps students make **data-driven decisions**
    about pursuing a **Master’s degree** using Machine Learning.
    """)

elif selected == "About Model":
    st.subheader("📘 About the Model")
    st.markdown("""
    **Algorithm:** Random Forest Classifier  
    **Inputs:** GATE Score, Salary  
    **Output:** Should pursue Master’s (Yes / No)
    """)

    # Feature Importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("🔍 Feature Importance")
    st.bar_chart(importance_df.set_index("Feature"))

    st.info(
        "⚠️ This system provides decision support, not absolute advice. "
        "Final decisions should consider personal goals and circumstances."
    )

elif selected == "Dataset":
    st.subheader("📁 Dataset Preview")
    st.dataframe(model_df)

    st.subheader("📊 Dataset Summary")
    st.write(model_df.describe())

elif selected == "Prediction Model":
    st.subheader("🤖 Prediction")

    gate_input = st.selectbox("GATE Score", le_gate.classes_)
    salary_input = st.number_input(
        "Salary (INR)", min_value=0, value=500000, step=50000
    )

    if salary_input < 100000:
        st.warning("⚠️ Salary seems very low. Please verify.")

    if st.button("Predict"):
        gate_encoded = le_gate.transform([gate_input])[0]
        prediction = model.predict([[gate_encoded, salary_input]])
        probability = model.predict_proba([[gate_encoded, salary_input]])[0]

        result = le_master.inverse_transform(prediction)[0]
        confidence = max(probability) * 100

        st.success(
            f"You should {'🧑‍🎓 DO' if result == 'Yes' else '🚫 NOT DO'} a Master's"
        )
        st.info(f"Confidence Level: **{confidence:.2f}%**")

elif selected == "Graph":
    st.subheader("📊 Visualization")

    feature = st.selectbox(
        "Choose feature",
        ["GATE_Score", "Salary", "Should_Do_Masters"]
    )

    plt.clf()

    if feature == "GATE_Score":
        counts = model_df[feature].value_counts().sort_index()
        labels = [gate_label_map[i] for i in counts.index]
        plt.bar(labels, counts.values)
        st.pyplot(plt.gcf())

    elif feature == "Salary":
        plt.hist(model_df[feature], bins=15)
        st.pyplot(plt.gcf())

    elif feature == "Should_Do_Masters":
        counts = model_df[feature].value_counts().sort_index()
        labels = [master_label_map[i] for i in counts.index]
        plt.pie(counts, labels=labels, autopct="%1.1f%%")
        st.pyplot(plt.gcf())

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Developed by Yash Chhatbar | Data-Driven Career Decision System")

