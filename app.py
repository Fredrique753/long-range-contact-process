# ============================================================
# STREAMLIT WEB APPLICATION
# LONG-RANGE CONTACT PROCESS BOUNDARY PREDICTION
# ============================================================
#
# Predicts:
#
#       y = f(lambda, alpha, x)
#
# using the trained Neural Network surrogate model.
#
# Input variables:
#       1. lambda - infection parameter
#       2. alpha  - long-range exponent
#       3. x      - horizontal boundary coordinate
#
# Target variable:
#       y      - vertical boundary coordinate
#
# ============================================================


import os
import random
import math

import streamlit as st
import numpy as np
import pandas as pd
import keras


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Long-Range Contact Process",
    page_icon="🦠",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "contact_process_boundary_neural_network.keras"
)

SCALING_FILE = os.path.join(
    BASE_DIR,
    "model",
    "scaling_parameters.csv"
)

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "results",
    "model_comparison",
    "model_comparison_results.csv"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Long-Range Contact Process")

st.subheader(
    "Neural-Network Prediction of the Boundary Coordinate y"
)

st.markdown(
    """
This application demonstrates a neural-network surrogate model for the
long-range Contact Process.

The model predicts the vertical boundary coordinate \(y\) from:

- the infection parameter \(\lambda\),
- the long-range exponent \(\alpha\),
- and the horizontal boundary coordinate \(x\).

Therefore:

\[
y = f(\lambda,\alpha,x)
\]
"""
)


# ============================================================
# LOAD NEURAL NETWORK MODEL
# ============================================================

@st.cache_resource
def load_neural_network():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    loaded_model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return loaded_model


# ============================================================
# LOAD SCALING PARAMETERS
# ============================================================

@st.cache_data
def load_scaling_parameters():

    if not os.path.exists(SCALING_FILE):
        raise FileNotFoundError(
            f"Scaling file was not found:\n{SCALING_FILE}"
        )

    scaling_data = pd.read_csv(SCALING_FILE)

    required_columns = {
        "variable",
        "mean",
        "sd"
    }

    if not required_columns.issubset(
        scaling_data.columns
    ):
        raise ValueError(
            "scaling_parameters.csv must contain "
            "the columns: variable, mean, sd"
        )

    scaling = {}

    for _, row in scaling_data.iterrows():

        scaling[str(row["variable"])] = {

            "mean": float(row["mean"]),

            "sd": float(row["sd"])

        }

    required_variables = [
        "lambda",
        "alpha",
        "x",
        "y"
    ]

    missing_variables = [

        variable

        for variable in required_variables

        if variable not in scaling

    ]

    if len(missing_variables) > 0:

        raise ValueError(
            "Missing scaling parameters for: "
            + ", ".join(missing_variables)
        )

    return scaling


# ============================================================
# LOAD MODEL AND SCALING PARAMETERS
# ============================================================

st.sidebar.header("Neural-Network Model")

model = None
scaling_parameters = None
model_loaded = False


try:

    model = load_neural_network()

    scaling_parameters = (
        load_scaling_parameters()
    )

    # --------------------------------------------------------
    # VERIFY NUMBER OF INPUT VARIABLES
    # --------------------------------------------------------

    input_shape = model.input_shape

    if isinstance(input_shape, list):

        expected_features = input_shape[0][-1]

    else:

        expected_features = input_shape[-1]


    if expected_features != 3:

        raise ValueError(
            f"The deployed model expects "
            f"{expected_features} input variables, "
            "but this application requires exactly "
            "3 inputs: lambda, alpha, and x."
        )


    model_loaded = True

    st.sidebar.success(
        "Neural-network model loaded successfully."
    )

    st.sidebar.info(
        "Model inputs: λ, α, x"
    )


except Exception as error:

    model_loaded = False

    st.sidebar.error(
        "The trained model could not be loaded correctly."
    )

    st.sidebar.code(
        str(error)
    )


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

    min_value=-126,

    max_value=111,

    value=0,

    step=1

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

    # --------------------------------------------------------
    # SCALE INPUT VARIABLES
    # --------------------------------------------------------

    lambda_scaled = (

        lambda_input
        - scaling["lambda"]["mean"]

    ) / scaling["lambda"]["sd"]


    alpha_scaled = (

        alpha_input
        - scaling["alpha"]["mean"]

    ) / scaling["alpha"]["sd"]


    x_scaled = (

        x_input
        - scaling["x"]["mean"]

    ) / scaling["x"]["sd"]


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # INPUT ORDER MUST MATCH MODEL TRAINING:
    #
    #       lambda, alpha, x
    #
    # --------------------------------------------------------

    input_array = np.array(

        [[

            lambda_scaled,

            alpha_scaled,

            x_scaled

        ]],

        dtype=np.float32

    )


    # --------------------------------------------------------
    # PREDICT SCALED y
    # --------------------------------------------------------

    prediction_scaled = trained_model.predict(

        input_array,

        verbose=0

    )


    prediction_scaled = float(

        np.asarray(
            prediction_scaled
        ).reshape(-1)[0]

    )


    # --------------------------------------------------------
    # CONVERT BACK TO ORIGINAL y SCALE
    # --------------------------------------------------------

    prediction = (

        prediction_scaled
        * scaling["y"]["sd"]

        + scaling["y"]["mean"]

    )


    return float(prediction)


# ============================================================
# MAIN PREDICTION SECTION
# ============================================================

st.header("Boundary Prediction")


col1, col2 = st.columns(2)


with col1:

    st.metric(

        "Infection parameter λ",

        f"{lambda_value:.2f}"

    )


    st.metric(

        "Long-range exponent α",

        f"{alpha_value:.2f}"

    )


    st.metric(

        "Horizontal boundary coordinate x",

        f"{x_value:.0f}"

    )


with col2:

    if model_loaded:

        try:

            predicted_y = predict_boundary_y(

                model,

                scaling_parameters,

                lambda_value,

                alpha_value,

                x_value

            )


            st.metric(

                "Predicted boundary coordinate y",

                f"{predicted_y:.2f}"

            )


            st.success(

                "The neural-network surrogate has "
                "successfully predicted y."

            )


        except Exception as error:

            st.error(

                "Prediction failed."

            )

            st.code(

                str(error)

            )


    else:

        st.warning(

            "Prediction is unavailable because "
            "the trained neural-network model "
            "was not loaded correctly."

        )


# ============================================================
# MODEL COMPARISON RESULTS
# ============================================================

st.header("Model Comparison")


try:

    if not os.path.exists(COMPARISON_FILE):

        raise FileNotFoundError(
            "Model comparison results file was not found."
        )


    comparison_results = pd.read_csv(
        COMPARISON_FILE
    )


    st.dataframe(

        comparison_results,

        use_container_width=True

    )


    st.markdown(
        """
The models are compared using predictive performance measures obtained
from the test dataset.

The four predictive models are:

1. Linear Regression
2. Random Forest
3. Support Vector Regression
4. Neural Network

The Neural Network is used for the interactive surrogate prediction
of the boundary coordinate \(y\).
"""
    )


except Exception as error:

    st.warning(
        "Model comparison results could not be loaded."
    )

    st.caption(
        str(error)
    )


# ============================================================
# SIMPLE CONTACT PROCESS DEMONSTRATION
# ============================================================

st.header(
    "Simple Long-Range Contact Process Demonstration"
)


st.markdown(
    """
This section provides a simplified illustration of the stochastic
long-range spreading mechanism.

It is included only as a visual demonstration and does not replace
the full Contact Process simulation used to generate the research
dataset.
"""
)


if st.button(
    "Run demonstration simulation"
):

    random.seed(42)


    infected_sites = [

        (0, 0)

    ]


    ever_infected = {

        (0, 0)

    }


    for _ in range(100):


        if len(infected_sites) == 0:

            break


        source = random.choice(
            infected_sites
        )


        u = random.random()


        radius = (

            (1 - u)
            ** (-1 / (alpha_value - 2))

        )


        theta = random.uniform(

            0,

            2 * math.pi

        )


        dx = int(
            round(
                radius * math.cos(theta)
            )
        )


        dy = int(
            round(
                radius * math.sin(theta)
            )
        )


        target = (

            source[0] + dx,

            source[1] + dy

        )


        if target not in ever_infected:

            infected_sites.append(
                target
            )

            ever_infected.add(
                target
            )


        if (

            random.random() < 0.20

            and

            len(infected_sites) > 1

        ):

            recovered_site = random.choice(
                infected_sites
            )


            infected_sites.remove(
                recovered_site
            )


    demonstration_data = pd.DataFrame(

        list(ever_infected),

        columns=["x", "y"]

    )


    st.success(

        f"The demonstration generated "
        f"{len(ever_infected)} sites that were "
        "infected at least once."

    )


    st.scatter_chart(

        demonstration_data,

        x="x",

        y="y"

    )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.header("About the Project")


st.markdown(
    """
This application supports research on the asymptotic growth geometry
of a long-range Contact Process.

The Contact Process is a stochastic interacting-particle system in
which occupied sites may recover, while infection spreads from
occupied sites to other sites.

In the long-range version, infection can occur beyond immediate
neighbours. The long-range exponent \(\alpha\) controls the effect of
distance on the infection mechanism.

The neural-network surrogate model approximates the relationship
between:

\[
(\lambda,\alpha,x)
\]

and the boundary coordinate:

\[
y
\]

That is:

\[
\boxed{
y=f(\lambda,\alpha,x)
}
\]
"""
)


st.markdown(
    """
**Research question**

Can a neural-network surrogate efficiently predict the boundary
geometry generated by a long-range Contact Process?
"""
)


# ============================================================
# END OF APPLICATION
# ============================================================
