import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.set_page_config(page_title="Brand Switching Analysis", layout="wide")

st.title("Brand Switching Analysis using Markov Chains and ANOVA")
st.write("Upload your CSV file to compute the transition matrix, future predictions, steady state, convergence graph, and ANOVA results.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

EXPECTED_RENAMED = [
    'Gender',
    'Age_Group',
    'Previous_Brand',
    'Current_Brand',
    'Switch_Frequency',
    'Switch_Reason',
    'Purchase_Frequency'
]

def preprocess_data(df):
    df = df.copy()

    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    if len(df.columns) >= 7:
        df = df.iloc[:, :7]
        df.columns = EXPECTED_RENAMED
    else:
        raise ValueError("The uploaded file does not have enough columns after removing Timestamp.")

    for col in EXPECTED_RENAMED:
        df[col] = df[col].astype(str).str.strip()

    return df

def compute_markov_outputs(df):
    transition_counts = pd.crosstab(df['Previous_Brand'], df['Current_Brand'])
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    brands = transition_matrix.index.tolist()
    P = transition_matrix.values

    if P.shape[0] != P.shape[1]:
        raise ValueError(
            "The transition matrix is not square. This usually happens when some brands appear only in "
            "Previous_Brand or only in Current_Brand. Make sure the same set of brands appears in both."
        )

    P2 = np.linalg.matrix_power(P, 2)
    P3 = np.linalg.matrix_power(P, 3)
    P10 = np.linalg.matrix_power(P, 10)

    eigvals, eigvecs = np.linalg.eig(P.T)
    steady_vec = eigvecs[:, np.isclose(eigvals, 1)]
    steady_vec = steady_vec[:, 0]
    steady_vec = np.real(steady_vec / steady_vec.sum())
    steady = pd.Series(steady_vec, index=brands, name="Steady_State")

    initial = np.array([1 / len(P)] * len(P))
    next_step = initial @ P
    future_3 = initial @ np.linalg.matrix_power(P, 3)

    steps = 10
    distributions = []
    current = initial.copy()
    for _ in range(steps):
        current = current @ P
        distributions.append(current.copy())
    distributions = np.array(distributions)

    return {
        "transition_counts": transition_counts,
        "transition_matrix": transition_matrix,
        "brands": brands,
        "P2_df": pd.DataFrame(P2, index=brands, columns=brands),
        "P3_df": pd.DataFrame(P3, index=brands, columns=brands),
        "steady": steady,
        "next_step": pd.Series(next_step, index=brands, name="Next_Step"),
        "future_3": pd.Series(future_3, index=brands, name="Three_Steps_Ahead"),
        "distributions": distributions,
    }

def compute_anova(df):
    df = df.copy()
    df['Transition'] = df['Previous_Brand'] + " → " + df['Current_Brand']
    df['Transition_Code'] = df['Transition'].astype('category').cat.codes

    model = ols(
        'Transition_Code ~ C(Gender) + C(Switch_Reason) + C(Purchase_Frequency)',
        data=df
    ).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, df[['Transition', 'Transition_Code']]

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df = preprocess_data(raw_df)

        st.subheader("Cleaned Data")
        st.dataframe(df, use_container_width=True)

        outputs = compute_markov_outputs(df)

        st.subheader("Transition Count Matrix")
        st.dataframe(outputs["transition_counts"], use_container_width=True)

        st.subheader("Transition Probability Matrix")
        st.dataframe(outputs["transition_matrix"], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2-Step Transition Matrix (P²)")
            st.dataframe(outputs["P2_df"], use_container_width=True)
        with col2:
            st.subheader("3-Step Transition Matrix (P³)")
            st.dataframe(outputs["P3_df"], use_container_width=True)

        st.subheader("Future Prediction")
        pred_df = pd.concat([outputs["next_step"], outputs["future_3"]], axis=1)
        pred_df.columns = ["Next Step", "3 Steps Ahead"]
        st.dataframe(pred_df, use_container_width=True)

        st.subheader("Steady State")
        st.dataframe(outputs["steady"].to_frame(), use_container_width=True)

        st.subheader("Long Run Market Share (Steady State)")
        fig1, ax1 = plt.subplots()
        ax1.bar(outputs["steady"].index, outputs["steady"].values)
        ax1.set_title("Long Run Market Share (Steady State)")
        ax1.set_xlabel("Brand")
        ax1.set_ylabel("Probability")
        st.pyplot(fig1)

        st.subheader("Convergence to Steady State")
        fig2, ax2 = plt.subplots()
        distributions = outputs["distributions"]
        brands = outputs["brands"]
        for i in range(len(brands)):
            ax2.plot(distributions[:, i], label=brands[i])
        ax2.set_title("Convergence to Steady State")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Probability")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("ANOVA on Brand Transition")
        anova_table, transition_preview = compute_anova(df)
        st.dataframe(transition_preview, use_container_width=True)
        st.dataframe(anova_table, use_container_width=True)

        st.info("In the ANOVA table, a p-value less than 0.05 indicates a significant effect.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.markdown("""
    ### Expected file format
    Your CSV should contain these columns before cleaning:
    - Timestamp
    - Gender
    - Age Group
    - Which clothing brand did you purchase from previously?
    - Which clothing brand do you currently purchase/prefer?
    - How often do you switch between clothing brands?
    - What is the main reason you switch brands?
    - How frequently do you purchase clothing?
    """)