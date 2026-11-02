# ============================================================
# STREAMLIT WEB APPLICATION
# LONG-RANGE CONTACT PROCESS BOUNDARY PREDICTION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import keras
from pathlib import Path
import random
import math

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Long-Range Contact Process",
    page_icon="🦠",
    layout="wide"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "contact_process_boundary_neural_network.h5"
)

SCALING_PATH = (
    BASE_DIR
    / "models"
    / "scaling_parameters.csv"
)

COMPARISON_FILE = (
    BASE_DIR
    / "results"
    / "model_comparison"
    / "model_comparison_results.csv"
)

# ============================================================
# LOAD SCALING PARAMETERS
# ============================================================

@st.cache_data
def load_scaling_parameters():
    if not SCALING_PATH.exists():
        raise FileNotFoundError(
            f"Scaling parameters file not found:\n{SCALING_PATH}"
        )

    scaling_df = pd.read_csv(SCALING_PATH)
    required_columns = {"variable", "mean", "sd"}

    if not required_columns.issubset(scaling_df.columns):
        raise ValueError(
            "scaling_parameters.csv must contain the columns: "
            "variable, mean, sd"
        )

    scaling = {}
    for _, row in scaling_df.iterrows():
        scaling[row["variable"]] = {
            "mean": float(row["mean"]),
            "sd": float(row["sd"])
        }

    required_variables = ["lambda", "alpha", "x", "y"]
    for variable in required_variables:
        if variable not in scaling:
            raise ValueError(
                f"Missing scaling parameters for '{variable}'"
            )

    return scaling

# ============================================================
# LOAD NEURAL-NETWORK MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    # compile=False is appropriate because the app only predicts.
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model

# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("Long-Range Contact Process")
st.subheader(
    "Neural-Network Prediction of the Boundary Coordinate y"
)

st.markdown(
    """
    This application demonstrates a neural-network surrogate model
    for the long-range Contact Process.

    The trained neural network predicts the vertical boundary
    coordinate **y** using:

    - the infection parameter **λ (lambda)**,
    - the long-range exponent **α (alpha)**,
    - and the horizontal boundary coordinate **x**.
    """
)

# ============================================================
# LOAD FILES
# ============================================================

model = None
scaling_parameters = None

try:
    scaling_parameters = load_scaling_parameters()
except Exception as error:
    st.error("The scaling parameters could not be loaded.")
    st.code(str(error))

try:
    model = load_model()
except Exception as error:
    st.error("The trained neural-network model could not be loaded.")
    st.code(str(error))

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Neural-Network Model")

if model is not None:
    st.sidebar.success("Neural-network model loaded successfully.")
    st.sidebar.write("**Architecture:** 3 → 64 → 32 → 16 → 1")
else:
    st.sidebar.error("The trained model could not be loaded.")

# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("Model Inputs")

lambda_value = st.sidebar.slider(
    "Infection parameter λ",
    min_value=0.20,
    max_value=0.60,
    value=0.40,
    step=0.05
)

alpha_value = st.sidebar.slider(
    "Long-range exponent α",
    min_value=3.50,
    max_value=4.00,
    value=3.75,
    step=0.25
)

x_value = st.sidebar.slider(
    "Horizontal boundary coordinate x",
    min_value=-200.0,
    max_value=200.0,
    value=0.0,
    step=1.0
)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_boundary_y(
    trained_model,
    scaling,
    lambda_input,
    alpha_input,
    x_input
):
    # STANDARDIZE INPUT VARIABLES
    lambda_scaled = (lambda_input - scaling["lambda"]["mean"]) / scaling["lambda"]["sd"]
    alpha_scaled = (alpha_input - scaling["alpha"]["mean"]) / scaling["alpha"]["sd"]
    x_scaled = (x_input - scaling["x"]["mean"]) / scaling["x"]["sd"]

    # CREATE MODEL INPUT
    input_array = np.array(
        [[lambda_scaled, alpha_scaled, x_scaled]],
        dtype=np.float32
    )

    # NEURAL-NETWORK PREDICTION
    prediction_scaled = trained_model.predict(input_array, verbose=0)
    prediction_scaled = float(np.asarray(prediction_scaled).flatten()[0])

    # TRANSFORM PREDICTION BACK TO ORIGINAL SCALE
    prediction = (
        prediction_scaled
        * scaling["y"]["sd"]
        + scaling["y"]["mean"]
    )

    return prediction

# ============================================================
# BOUNDARY PREDICTION
# ============================================================

st.header("Boundary Prediction")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Parameters")
    st.metric("Infection parameter λ", f"{lambda_value:.2f}")
    st.metric("Long-range exponent α", f"{alpha_value:.2f}")
    st.metric("Horizontal coordinate x", f"{x_value:.2f}")

with col2:
    st.subheader("Neural-Network Prediction")
    if (model is not None and scaling_parameters is not None):
        try:
            predicted_y = predict_boundary_y(
                model,
                scaling_parameters,
                lambda_value,
                alpha_value,
                x_value
            )
            st.metric("Predicted boundary coordinate y", f"{predicted_y:.2f}")
            st.success("Prediction completed successfully.")
        except Exception as error:
            st.error("An error occurred while making the prediction.")
            st.code(str(error))
    else:
        st.warning(
            "Prediction is unavailable because the model or "
            "scaling parameters could not be loaded."
        )

# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("Model Comparison")
st.markdown(
    """
    The research compares four predictive approaches:

    1. Linear Regression
    2. Random Forest
    3. Support Vector Regression
    4. Neural Network
    """
)

if COMPARISON_FILE.exists():
    try:
        comparison_results = pd.read_csv(COMPARISON_FILE)
        st.dataframe(comparison_results, use_container_width=True)
    except Exception as error:
        st.warning("The model comparison file exists but could not be read.")
        st.code(str(error))
else:
    st.info(
        "The model comparison results file has not yet been "
        "found in the repository."
    )

# ============================================================
# FULL CONTACT PROCESS DEMONSTRATION
# ============================================================

st.header("Long-Range Contact Process Simulation")
st.markdown(
    """
    This simulation models the stochastic spreading of the Contact Process. 
    It starts with a single infected site at the origin and uses the long-range 
    exponent **α** to determine how far the infection can jump. 
    
    Adjust the sliders in the sidebar and click the button to run the simulation.
    """
)

# Add a slider for the number of simulation steps
steps = st.slider("Number of simulation steps", min_value=100, max_value=5000, value=1000, step=100)

if st.button("Run Full Simulation"):
    with st.spinner("Simulating infection spread..."):
        # Initialize variables
        random.seed(42)
        infected_sites = [(0, 0)]
        ever_infected = {(0, 0)}
        
        # We use a while loop to represent time steps (generations)
        current_step = 0
        
        while current_step < steps:
            # If the infection dies out completely, stop early
            if len(infected_sites) == 0:
                break
                
            # Pick a random infected site to act as the source
            source = random.choice(infected_sites)
            
            # 1. Infection attempt (Long-range jump)
            u = random.random()
            # The inverse transform sampling for a heavy-tailed distribution
            radius = (1 - u) ** (-1 / (alpha_value - 2))
            
            theta = random.uniform(0, 2 * math.pi)
            
            dx = int(round(radius * math.cos(theta)))
            dy = int(round(radius * math.sin(theta)))
            
            target = (source[0] + dx, source[1] + dy)
            
            # If the target hasn't been infected yet, add it
            if target not in ever_infected:
                infected_sites.append(target)
                ever_infected.add(target)
                
            # 2. Recovery attempt (based on lambda parameter)
            # Higher lambda = lower recovery rate
            if random.random() < (1 - lambda_value) and len(infected_sites) > 1:
                recovered_site = random.choice(infected_sites)
                infected_sites.remove(recovered_site)
                
            current_step += 1

        # Convert results to a DataFrame
        demonstration_data = pd.DataFrame(
            list(ever_infected),
            columns=["x", "y"]
        )

        # Display results
        st.success(
            f"Simulation complete! Generated **{len(ever_infected)}** unique infected sites "
            f"over {current_step} steps."
        )
        
        # Display interactive map
        st.scatter_chart(
            demonstration_data, 
            x="x", 
            y="y",
            use_container_width=True
        )

# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.header("About the Project")
st.markdown(
    """
    This application supports research on the asymptotic growth
    geometry of a **long-range Contact Process**.

    The Contact Process is a stochastic interacting-particle system
    in which occupied sites may recover while infection spreads to
    other sites.

    In the long-range version, infection can spread beyond immediate
    neighbouring sites. The parameter **α** controls the long-range
    behaviour of the infection mechanism.

    The neural network acts as a **surrogate model**, learning the
    relationship between:

    - infection parameter **λ**,
    - long-range exponent **α**,
    - horizontal boundary coordinate **x**,

    and the resulting vertical boundary coordinate **y**.
    """
)

st.markdown(
    """
    ### Research Question

    **Can machine-learning surrogate models efficiently predict the
    boundary geometry generated by a long-range Contact Process?**
    """
)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Long-Range Contact Process | "
    "Simulation and Machine-Learning Surrogate Modelling"
)
