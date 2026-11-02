# ============================================================
# STREAMLIT WEB APPLICATION
# LONG-RANGE CONTACT PROCESS BOUNDARY PREDICTION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow import keras

import random
import math

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Long-Range Contact Process",
    page_icon=":microbe:",
    layout="wide"
)

st.title("Long-Range Contact Process")
st.subheader(
    "Neural-Network Prediction of the Boundary Coordinate y"
)

st.markdown(
    """
    This application demonstrates a neural-network surrogate model
    for the long-range Contact Process.

    The model predicts the vertical boundary coordinate \(y\) from:

    - the infection parameter \(\\lambda\),
    - the long-range exponent \(\\alpha\),
    - and the horizontal boundary coordinate \(x\).
    """
)


# ============================================================
# LOAD TRAINED NEURAL-NETWORK MODEL
# ============================================================

MODEL_PATH = "model/contact_process_boundary_neural_network.keras"

st.sidebar.header("Neural-Network Model")

try:
    model = keras.models.load_model(MODEL_PATH)
    model_loaded = True
    st.sidebar.success("Model loaded successfully.")
except Exception as error:
    model = None
    model_loaded = False
    st.sidebar.error(
        "The trained model could not be loaded. "
        "Please ensure that the model file is included in the repository."
    )
    st.sidebar.code(str(error))


# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("Model Inputs")

lambda_value = st.sidebar.slider(
    "Infection parameter λ",
    min_value=0.2,
    max_value=0.6,
    value=0.4,
    step=0.05
)

alpha_value = st.sidebar.slider(
    "Long-range exponent α",
    min_value=3.5,
    max_value=4.0,
    value=3.75,
    step=0.25
)

x_value = st.sidebar.slider(
    "Horizontal boundary coordinate x",
    min_value=-50,
    max_value=50,
    value=0,
    step=1
)


# ============================================================
# SCALING PARAMETERS
# ============================================================
#
# Replace these values with the actual training means and standard
# deviations saved from your neural-network training script.
#
# They must match the values used when the model was trained.

SCALING_PARAMETERS = {
    "lambda": {"mean": 0.4, "sd": 0.1414},
    "alpha": {"mean": 3.75, "sd": 0.2041},
    "x": {"mean": -2.15, "sd": 30.0},
    "y": {"mean": 37.14, "sd": 30.0}
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_boundary_y(model, lambda_value, alpha_value, x_value):
    """
    Predict the boundary coordinate y using the trained model.
    """

    lambda_scaled = (
        lambda_value - SCALING_PARAMETERS["lambda"]["mean"]
    ) / SCALING_PARAMETERS["lambda"]["sd"]

    alpha_scaled = (
        alpha_value - SCALING_PARAMETERS["alpha"]["mean"]
    ) / SCALING_PARAMETERS["alpha"]["sd"]

    x_scaled = (
        x_value - SCALING_PARAMETERS["x"]["mean"]
    ) / SCALING_PARAMETERS["x"]["sd"]

    input_array = np.array(
        [[lambda_scaled, alpha_scaled, x_scaled]],
        dtype=np.float32
    )

    prediction_scaled = model.predict(input_array, verbose=0)

    prediction = (
        prediction_scaled[0][0] * SCALING_PARAMETERS["y"]["sd"]
        + SCALING_PARAMETERS["y"]["mean"]
    )

    return float(prediction)


# ============================================================
# MAIN PREDICTION SECTION
# ============================================================

st.header("Boundary Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric("Infection parameter λ", f"{lambda_value:.2f}")
    st.metric("Long-range exponent α", f"{alpha_value:.2f}")
    st.metric("Horizontal coordinate x", f"{x_value}")

with col2:
    if model_loaded:
        predicted_y = predict_boundary_y(
            model,
            lambda_value,
            alpha_value,
            x_value
        )

        st.metric(
            "Predicted boundary coordinate y",
            f"{predicted_y:.2f}"
        )

        st.success(
            "The neural-network surrogate has predicted the "
            "upper boundary coordinate for the selected parameters."
        )

    else:
        st.warning(
            "Prediction is unavailable because the trained model "
            "could not be loaded."
        )


# ============================================================
# SIMPLE CONTACT PROCESS DEMONSTRATION
# ============================================================

st.header("Simple Long-Range Contact Process Demonstration")

st.markdown(
    """
    This is a simplified demonstration of the stochastic spreading
    mechanism. It starts with one infected site at the origin and
    generates a small set of sites that become infected over time.
    """
)

if st.button("Run demonstration simulation"):

    random.seed(42)

    infected_sites = [(0, 0)]
    ever_infected = {(0, 0)}

    for _ in range(100):

        source = random.choice(infected_sites)

        u = random.random()
        radius = (1 - u) ** (-1 / (alpha_value - 2))

        theta = random.uniform(0, 2 * math.pi)

        dx = int(round(radius * math.cos(theta)))
        dy = int(round(radius * math.sin(theta)))

        target = (
            source[0] + dx,
            source[1] + dy
        )

        if target not in ever_infected:
            infected_sites.append(target)
            ever_infected.add(target)

        if random.random() < 0.2 and len(infected_sites) > 1:
            recovered_site = random.choice(infected_sites)
            infected_sites.remove(recovered_site)

    demonstration_data = pd.DataFrame(
        list(ever_infected),
        columns=["x", "y"]
    )

    st.success(
        f"The demonstration generated {len(ever_infected)} "
        "sites that were infected at least once."
    )

    st.map(
        demonstration_data,
        latitude="y",
        longitude="x",
        size=1
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.header("About the Project")

st.markdown(
    """
    This application supports research on the asymptotic growth
    geometry of a long-range Contact Process.

    The Contact Process is a stochastic interacting-particle system
    in which occupied sites may recover, while infection may spread
    from occupied sites to vacant sites.

    In the long-range version, infection can spread beyond immediate
    neighbours. The exponent \(\\alpha\) controls how rapidly the
    infection contribution decreases with distance.

    The neural-network surrogate approximates the relationship
    between the model parameters and the boundary coordinate of the
    cumulative occupied region.
    """
)

st.markdown(
    """
    **Research question:**  
    Can a neural-network surrogate efficiently predict the boundary
    geometry generated by a long-range Contact Process?
    """
)