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
from collections import deque
import altair as alt  # <-- Added Altair for strict axis limits

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
    min_value=-50.0,
    max_value=50.0,
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
        
        # Format the table: hide index and set precision to 4 decimal places
        st.dataframe(
            comparison_results.style.format(precision=4), 
            hide_index=True, 
            use_container_width=True
        )
        
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
steps = st.slider("Number of simulation steps", min_value=100, max_value=10000, value=2000, step=100)

if st.button("Run Full Simulation"):
    with st.spinner("Simulating infection spread..."):
        # Initialize variables
        random.seed(42)
        infected_sites = [(0, 0)]
        ever_infected = {(0, 0)}
        
        current_step = 0
        
        while current_step < steps:
            if len(infected_sites) == 0:
                break
                
            source = random.choice(infected_sites)
            
            # 1. Infection attempt (Long-range jump)
            u = random.random()
            radius = (1 - u) ** (-1 / (alpha_value - 2))
            theta = random.uniform(0, 2 * math.pi)
            
            dx = int(round(radius * math.cos(theta)))
            dy = int(round(radius * math.sin(theta)))
            
            target = (source[0] + dx, source[1] + dy)
            
            if target not in ever_infected:
                infected_sites.append(target)
                ever_infected.add(target)
                
            # 2. Recovery attempt (based on lambda parameter)
            if random.random() < (1 - lambda_value) and len(infected_sites) > 1:
                recovered_site = random.choice(infected_sites)
                infected_sites.remove(recovered_site)
                
            current_step += 1

        demonstration_data = pd.DataFrame(list(ever_infected), columns=["x", "y"])

        st.success(
            f"Simulation complete! Generated **{len(ever_infected)}** unique infected sites "
            f"over {current_step} steps."
        )
        
        # Using Altair to strictly force the x-axis to -50 to 50
        chart = alt.Chart(demonstration_data).mark_circle(size=40, opacity=0.7, color="#4C9AFF").encode(
            x=alt.X('x', scale=alt.Scale(domain=[-50, 50]), title="Horizontal coordinate x (Locked to -50 to 50)"),
            y=alt.Y('y', scale=alt.Scale(zero=False), title="Boundary coordinate y"),
            tooltip=['x', 'y']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

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
# DATA STRUCTURES & OBJECT-ORIENTED PROGRAMMING (OOP)
# ============================================================

st.header("Data Structures & Object-Oriented Programming")
st.markdown(
    """
    This application is built using fundamental computer science principles. 
    Below is a demonstration of how Data Structures and Object-Oriented Programming 
    (OOP) are applied to the simulation and the neural network.
    """
)

col_ds, col_oop = st.columns(2)

with col_ds:
    st.subheader("Data Structures Used")
    st.markdown(
        """
        - **Arrays:** `numpy.array` is used to format the scaled inputs for the Neural Network.
        - **Lists:** Used to store the currently infected sites during the simulation.
        - **Tuples:** Immutable coordinates \((x, y)\) ensure the spatial location of a site cannot be accidentally modified.
        - **Sets:** `ever_infected` is a Set (Hash Table). It prevents duplicate entries and gives \(O(1)\) lookup times to check if a site was already infected.
        - **Dictionaries:** The scaling parameters (`mean`, `sd`) are stored as key-value pairs.
        - **Queues & Stacks:** Used to explore the lattice sites via BFS and DFS.
        - **Trees:** The Random Forest model relies on Decision Trees.
        - **Graphs:** The spatial lattice is represented as a Graph where edges are infection pathways.
        """
    )

with col_oop:
    st.subheader("OOP Principles Used")
    st.markdown(
        """
        - **Encapsulation:** The trained model's weights and layers are hidden. We only interact with it via the `model.predict()` method. 
        - **Abstraction:** The complex TensorFlow matrix multiplications are abstracted into a simple `predict_boundary_y()` function.
        - **Inheritance:** The Neural Network model inherits from the base `keras.Model` class.
        - **Polymorphism:** In the research, `LinearRegression`, `RandomForestRegressor`, and our Neural Network all share the same `predict()` method, allowing us to treat them interchangeably in the comparison table.
        """
    )

# ============================================================
# LIVE DATA STRUCTURE & OOP DEMONSTRATION
# ============================================================

st.subheader("Live Execution of Data Structures and OOP")
st.markdown(
    "Click the button below to run a live Python script that builds a **Linked List**, **Queue**, **Stack**, **Tree**, and **Graph**, and demonstrates **Inheritance & Polymorphism**."
)

if st.button("Run Live Code Demonstration"):
    
    # --- 1. LINKED LIST ---
    st.write("### 1. Linked List (Custom OOP Class)")
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None

    class LinkedList:
        def __init__(self):
            self.head = None

        def append(self, data):
            new_node = Node(data)
            if not self.head:
                self.head = new_node
                return
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

        def display(self):
            elements = []
            current = self.head
            while current:
                elements.append(str(current.data))
                current = current.next
            return " -> ".join(elements)

    history_list = LinkedList()
    history_list.append((0, 0))
    history_list.append((1, 0))
    history_list.append((1, 1))
    st.code(f"Linked List of Infection Coordinates: {history_list.display()}")

    # --- 2. QUEUE (FIFO) ---
    st.write("### 2. Queue (First-In, First-Out) - Used for Breadth-First Search")
    infection_queue = deque()
    infection_queue.append((0, 0))
    infection_queue.append((2, 5))
    infection_queue.append((4, 1))
    st.write(f"Queue of pending sites to process: `{list(infection_queue)}`")
    processed = infection_queue.popleft()
    st.write(f"Processed site (FIFO): `{processed}`. Remaining queue: `{list(infection_queue)}`")

    # --- 3. STACK (LIFO) ---
    st.write("### 3. Stack (Last-In, First-Out) - Used for Depth-First Search / Undo History")
    history_stack = []
    history_stack.append("Start Simulation")
    history_stack.append("Infect (0,0)")
    history_stack.append("Infect (1,1)")
    st.write(f"Stack of simulation events: `{history_stack}`")
    last_event = history_stack.pop()
    st.write(f"Undoing last event (LIFO): `{last_event}`. Remaining stack: `{history_stack}`")

    # --- 4. BINARY SEARCH TREE ---
    st.write("### 4. Trees (Binary Search Tree)")
    class TreeNode:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.left = None
            self.right = None

    def insert(root, key, value):
        if root is None:
            return TreeNode(key, value)
        if key < root.key:
            root.left = insert(root.left, key, value)
        else:
            root.right = insert(root.right, key, value)
        return root

    def inorder(root):
        res = []
        if root:
            res = inorder(root.left)
            res.append((root.key, root.value))
            res = res + inorder(root.right)
        return res

    # Build tree from boundary x coordinates
    tree_root = None
    tree_root = insert(tree_root, -10, 20.5)
    tree_root = insert(tree_root, 5, 35.2)
    tree_root = insert(tree_root, 20, 50.1)
    st.write("Tree nodes (sorted via In-Order Traversal):")
    st.json(inorder(tree_root))

    # --- 5. GRAPH (Adjacency List) ---
    st.write("### 5. Graphs (Adjacency List of Lattice Sites)")
    infection_graph = {
        (0, 0): [(1, 0), (0, 1)],
        (1, 0): [(2, 0)],
        (0, 1): [(1, 1)],
    }
    st.write("Adjacency List representing edges of infection:")
    st.json({str(k): [str(v) for v in val] for k, val in infection_graph.items()})

    # --- 6. OOP INHERITANCE & POLYMORPHISM ---
    st.write("### 6. OOP: Inheritance and Polymorphism")
    st.markdown("We define a base class `BoundaryModel`, and two child classes that inherit from it but override the `predict` method (Polymorphism).")
    
    class BoundaryModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, x):
            return 0.0

    class EmpiricalModel(BoundaryModel):
        def predict(self, x):
            return x * 2.5  # Linear interpolation example

    class MeanModel(BoundaryModel):
        def predict(self, x):
            return 37.2  # Always predicts the mean

    models = [EmpiricalModel("Empirical"), MeanModel("Mean")]
    
    x_test = 10
    results = []
    for m in models:
        # Polymorphism: We call the same .predict() method on different objects
        results.append({"Model": m.name, "Input x": x_test, "Prediction": m.predict(x_test)})
    
    st.dataframe(pd.DataFrame(results), hide_index=True)

    # --- 7. HASH TABLE (DICTIONARY) ---
    st.write("### 7. Hash Tables (Dictionaries)")
    st.write("Storing scaling parameters as Key-Value pairs for \(O(1)\) access:")
    st.json(scaling_parameters)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Long-Range Contact Process | "
    "Simulation and Machine-Learning Surrogate Modelling"
)
