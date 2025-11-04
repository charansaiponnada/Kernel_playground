"""
Kernel Methods Playground - Interactive SVM Visualization with Streamlit
Author: CSP
Run: streamlit run kernel_playground.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from typing import Callable, Tuple, List
import io

# Page configuration
st.set_page_config(
    page_title="Kernel Methods Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Theme toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Custom CSS based on theme
if st.session_state.dark_mode:
    # Dark mode styling
    st.markdown("""
    <style>
        .main {
            background-color: #0a0a0a;
        }
        .stApp {
            background-color: #0a0a0a;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 700;
        }
        .metric-card {
            background: rgba(30, 30, 30, 0.9);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1a1a1a;
        }
        .stTabs [data-baseweb="tab"] {
            color: #ffffff;
            background-color: #1a1a1a;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2a2a2a;
            color: #00ff88 !important;
        }
        .sidebar .sidebar-content {
            background-color: #0f0f0f;
        }
        p, label, .stMarkdown {
            color: #e0e0e0 !important;
        }
        .stSelectbox, .stSlider, .stFileUploader {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light mode styling
    st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1a1a1a !important;
            font-weight: 700;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa;
        }
        .stTabs [data-baseweb="tab"] {
            color: #1a1a1a;
            background-color: #f8f9fa;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            color: #0066cc !important;
            border-bottom: 3px solid #0066cc;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        p, label, .stMarkdown {
            color: #2a2a2a !important;
        }
        .stSelectbox, .stSlider, .stFileUploader {
            color: #1a1a1a !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Title with theme toggle
col_title, col_theme = st.columns([6, 1])
with col_title:
    st.markdown("<h1 style='text-align: center;'>🧠 Kernel Methods Playground</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Interactive SVM & Kernel Tricks Visualization</p>", unsafe_allow_html=True)
with col_theme:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🌓 Toggle Theme", use_container_width=True):
        toggle_theme()
        st.rerun()

# Dataset generators
def generate_linear(n: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable data"""
    np.random.seed(42)
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y

def generate_circles(n: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles"""
    np.random.seed(42)
    r = np.random.rand(n) * 2
    theta = np.random.rand(n) * 2 * np.pi
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y = (r < 1).astype(int)
    y = np.where(y == 0, -1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y

def generate_moons(n: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two moons pattern"""
    np.random.seed(42)
    n_half = n // 2
    theta = np.linspace(0, np.pi, n_half)
    
    # Upper moon
    X1 = np.column_stack([np.cos(theta), np.sin(theta)])
    # Lower moon
    X2 = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_half), -np.ones(n_half)])
    X += np.random.randn(n, 2) * noise
    return X, y

def generate_xor(n: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR pattern"""
    np.random.seed(42)
    X = np.random.randn(n, 2) * 2
    y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(int)
    y = np.where(y == 0, -1, 1)
    X += np.random.randn(n, 2) * noise
    return X, y

def generate_spiral(n: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral pattern"""
    np.random.seed(42)
    n_half = n // 2
    theta = np.linspace(0, 4 * np.pi, n_half)
    r = np.linspace(0, 2, n_half)
    
    # First spiral
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    # Second spiral (offset by pi)
    X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_half), -np.ones(n_half)])
    X += np.random.randn(n, 2) * noise
    return X, y

# Custom kernel functions
def custom_kernel_laplacian(X, Y, gamma=0.5):
    """Laplacian kernel"""
    from scipy.spatial.distance import cdist
    dists = cdist(X, Y, metric='cityblock')
    return np.exp(-gamma * dists)

def custom_kernel_chi2(X, Y):
    """Chi-squared kernel"""
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i, j] = np.sum((X[i] - Y[j])**2 / (X[i] + Y[j] + 1e-10))
    return np.exp(-K)

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Dataset selection
    st.markdown("#### 📊 Dataset Configuration")
    dataset_type = st.selectbox(
        "Select Dataset",
        ["Linear Separable", "Concentric Circles", "Two Moons", "XOR Pattern", "Spiral"],
        help="Choose a dataset optimized for different kernel types"
    )
    
    num_points = st.slider("Number of Points", 50, 500, 150, 10)
    noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.01)
    
    # Custom dataset upload
    st.markdown("#### 📁 Upload Custom Dataset")
    uploaded_file = st.file_uploader("Upload CSV (columns: x, y, label)", type=['csv'])
    
    st.markdown("---")
    
    # Kernel selection
    st.markdown("#### 🔧 Kernel Configuration")
    kernel_type = st.selectbox(
        "Kernel Type",
        ["Linear", "Polynomial", "RBF (Gaussian)", "Sigmoid", "Custom (Laplacian)", "Custom (Chi-squared)"],
        help="Select the kernel function for SVM"
    )
    
    # Kernel parameters
    if kernel_type in ["RBF (Gaussian)", "Sigmoid", "Custom (Laplacian)"]:
        gamma = st.slider("Gamma (γ)", 0.01, 5.0, 0.5, 0.01, 
                         help="Kernel coefficient - higher values = more complex boundaries")
    
    if kernel_type == "Polynomial":
        degree = st.slider("Degree", 2, 5, 3, 1, help="Polynomial degree")
        coef0 = st.slider("Coef0", 0.0, 5.0, 1.0, 0.1, help="Independent term in kernel")
    
    if kernel_type == "Sigmoid":
        coef0 = st.slider("Coef0", -2.0, 2.0, 0.0, 0.1, help="Independent term in kernel")
    
    st.markdown("---")
    
    # SVM parameters
    st.markdown("#### ⚙️ SVM Parameters")
    C = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1,
                  help="Penalty parameter - lower values = smoother boundary")
    
    st.markdown("---")
    
    # Visualization options
    st.markdown("#### 📈 Visualization Options")
    show_support_vectors = st.checkbox("Show Support Vectors", value=True)
    show_decision_boundary = st.checkbox("Show Decision Boundary", value=True)
    show_margins = st.checkbox("Show Margins", value=True)
    resolution = st.slider("Boundary Resolution", 20, 100, 50, 10,
                          help="Higher = smoother but slower")

# Generate or load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df[['x', 'y']].values
    y = df['label'].values
    y = np.where(y > 0, 1, -1)
    dataset_name = "Custom Dataset"
else:
    dataset_map = {
        "Linear Separable": generate_linear,
        "Concentric Circles": generate_circles,
        "Two Moons": generate_moons,
        "XOR Pattern": generate_xor,
        "Spiral": generate_spiral
    }
    X, y = dataset_map[dataset_type](num_points, noise_level)
    dataset_name = dataset_type

# Train SVM
start_time = time.time()

# Configure kernel
if kernel_type == "Linear":
    svm = SVC(kernel='linear', C=C)
elif kernel_type == "Polynomial":
    svm = SVC(kernel='poly', degree=degree, coef0=coef0, C=C, gamma='scale')
elif kernel_type == "RBF (Gaussian)":
    svm = SVC(kernel='rbf', gamma=gamma, C=C)
elif kernel_type == "Sigmoid":
    svm = SVC(kernel='sigmoid', gamma=gamma, coef0=coef0, C=C)
elif kernel_type == "Custom (Laplacian)":
    svm = SVC(kernel=lambda X, Y: custom_kernel_laplacian(X, Y, gamma), C=C)
elif kernel_type == "Custom (Chi-squared)":
    svm = SVC(kernel=custom_kernel_chi2, C=C)

# Fit the model
svm.fit(X, y)
y_pred = svm.predict(X)
accuracy = accuracy_score(y, y_pred)
compute_time = (time.time() - start_time) * 1000

# Get support vectors
support_vectors = X[svm.support_]
n_support = len(support_vectors)

# Main content area - Metrics cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #4ade80; margin: 0;'>Accuracy</h3>
        <h2 style='margin: 10px 0;'>{:.1f}%</h2>
    </div>
    """.format(accuracy * 100), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #fbbf24; margin: 0;'>Support Vectors</h3>
        <h2 style='margin: 10px 0;'>{}</h2>
    </div>
    """.format(n_support), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #a78bfa; margin: 0;'>Compute Time</h3>
        <h2 style='margin: 10px 0;'>{:.1f} ms</h2>
    </div>
    """.format(compute_time), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Set plot template based on theme
plot_template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"
plot_bg_color = "rgba(0,0,0,0)" if st.session_state.dark_mode else "rgba(255,255,255,1)"

# Create visualizations
tab1, tab2, tab3, tab4 = st.tabs(["📍 Decision Boundary", "🔥 Kernel Matrix", "🎯 Feature Space", "📊 Model Info"])

with tab1:
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Decision function for margins
    Z_scores = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_scores = Z_scores.reshape(xx.shape)
    
    # Create figure
    fig = go.Figure()
    
    # Add decision boundary contour
    if show_decision_boundary:
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=Z,
            colorscale=[[0, 'rgba(239, 68, 68, 0.3)'], [1, 'rgba(59, 130, 246, 0.3)']],
            showscale=False,
            contours=dict(
                start=-1,
                end=1,
                size=2,
                showlines=True,
                coloring='heatmap'
            ),
            hoverinfo='skip'
        ))
    
    # Add decision boundary line
    boundary_color = '#00ff88' if st.session_state.dark_mode else '#0066cc'
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z_scores,
        contours=dict(
            start=0,
            end=0,
            size=1,
            showlines=True,
            coloring='lines'
        ),
        line=dict(color=boundary_color, width=4),
        showscale=False,
        hoverinfo='skip',
        name='Decision Boundary'
    ))
    
    # Add margin lines
    if show_margins:
        margin_color = '#fbbf24' if st.session_state.dark_mode else '#ff9800'
        for margin in [-1, 1]:
            fig.add_trace(go.Contour(
                x=np.linspace(x_min, x_max, resolution),
                y=np.linspace(y_min, y_max, resolution),
                z=Z_scores,
                contours=dict(
                    start=margin,
                    end=margin,
                    size=1,
                    showlines=True,
                    coloring='lines'
                ),
                line=dict(color=margin_color, width=2, dash='dash'),
                showscale=False,
                hoverinfo='skip',
                name=f'Margin {margin}'
            ))
    
    # Add data points
    for label, color, name in [(-1, '#ef4444', 'Class -1'), (1, '#3b82f6', 'Class +1')]:
        mask = y == label
        point_line_color = 'white' if st.session_state.dark_mode else 'black'
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                line=dict(color=point_line_color, width=1)
            ),
            name=name
        ))
    
    # Add support vectors
    if show_support_vectors:
        sv_line_color = 'white' if st.session_state.dark_mode else 'black'
        fig.add_trace(go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode='markers',
            marker=dict(
                size=16,
                color='#fbbf24',
                symbol='diamond',
                line=dict(color=sv_line_color, width=2)
            ),
            name='Support Vectors'
        ))
    
    fig.update_layout(
        title=f"Decision Boundary - {kernel_type} Kernel",
        xaxis_title="Feature X",
        yaxis_title="Feature Y",
        template=plot_template,
        height=600,
        showlegend=True,
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.1)" if st.session_state.dark_mode else "rgba(0,0,0,0.05)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Compute kernel matrix for first 30 points
    n_show = min(30, len(X))
    from sklearn.metrics.pairwise import pairwise_kernels
    
    if kernel_type == "Linear":
        K = pairwise_kernels(X[:n_show], metric='linear')
    elif kernel_type == "Polynomial":
        K = pairwise_kernels(X[:n_show], metric='poly', degree=degree, coef0=coef0, gamma=1.0)
    elif kernel_type == "RBF (Gaussian)":
        K = pairwise_kernels(X[:n_show], metric='rbf', gamma=gamma)
    elif kernel_type == "Sigmoid":
        K = pairwise_kernels(X[:n_show], metric='sigmoid', gamma=gamma, coef0=coef0)
    elif kernel_type == "Custom (Laplacian)":
        K = custom_kernel_laplacian(X[:n_show], X[:n_show], gamma)
    elif kernel_type == "Custom (Chi-squared)":
        K = custom_kernel_chi2(X[:n_show], X[:n_show])
    
    colorscale = 'Viridis' if st.session_state.dark_mode else 'Blues'
    fig = go.Figure(data=go.Heatmap(
        z=K,
        colorscale=colorscale,
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title=f"Kernel Matrix ({n_show}×{n_show}) - Higher values = More similar",
        xaxis_title="Sample Index",
        yaxis_title="Sample Index",
        template=plot_template,
        height=600,
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Understanding the Kernel Matrix:**
    - Each cell (i,j) shows the similarity between samples i and j
    - Diagonal elements are always 1 (a point is identical to itself)
    - The pattern reveals how the kernel groups similar points together
    """)

with tab3:
    st.markdown("### Feature Space Transformation")
    
    if kernel_type == "Polynomial" and 'degree' in locals():
        st.markdown(f"""
        **Polynomial Kernel (degree={degree})** implicitly maps data to a higher-dimensional space:
        
        For 2D input (x, y), a degree-2 polynomial creates features:
        - φ(x,y) = [1, x, y, x², xy, y²]
        
        This creates a {((degree + 2) * (degree + 1)) // 2}-dimensional space!
        """)
        
        # Show polynomial features for degree 2
        if degree == 2:
            X_poly = np.column_stack([
                X[:, 0]**2,
                X[:, 0] * X[:, 1],
                X[:, 1]**2
            ])
            
            fig = go.Figure()
            for label, color, name in [(-1, '#ef4444', 'Class -1'), (1, '#3b82f6', 'Class +1')]:
                mask = y == label
                fig.add_trace(go.Scatter3d(
                    x=X_poly[mask, 0],
                    y=X_poly[mask, 1],
                    z=X_poly[mask, 2],
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=name
                ))
            
            fig.update_layout(
                title="3D Feature Space (x², xy, y²)",
                scene=dict(
                    xaxis_title="x²",
                    yaxis_title="xy",
                    zaxis_title="y²"
                ),
                template=plot_template,
                height=600,
                plot_bgcolor=plot_bg_color,
                paper_bgcolor=plot_bg_color
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif kernel_type == "RBF (Gaussian)":
        st.markdown("""
        **RBF Kernel** maps data to an **infinite-dimensional** space!
        
        K(x,y) = exp(-γ||x-y||²)
        
        This is equivalent to an infinite Taylor series expansion:
        - φ(x) lives in infinite dimensions
        - Can't visualize directly, but the kernel trick computes dot products efficiently
        - Creates highly flexible decision boundaries
        """)
        
        # Show RBF influence
        center_idx = np.random.choice(len(X))
        center = X[center_idx]
        
        distances = np.linalg.norm(X - center, axis=1)
        influences = np.exp(-gamma * distances**2)
        
        colorscale_rbf = 'Plasma' if st.session_state.dark_mode else 'YlOrRd'
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=influences,
                colorscale=colorscale_rbf,
                showscale=True,
                colorbar=dict(title="RBF Influence")
            ),
            text=[f"Influence: {inf:.3f}" for inf in influences],
            hoverinfo='text'
        ))
        
        star_color = '#fbbf24' if st.session_state.dark_mode else '#ff9800'
        fig.add_trace(go.Scatter(
            x=[center[0]],
            y=[center[1]],
            mode='markers',
            marker=dict(size=20, color=star_color, symbol='star'),
            name='Center Point'
        ))
        
        fig.update_layout(
            title=f"RBF Kernel Influence (γ={gamma})",
            xaxis_title="Feature X",
            yaxis_title="Feature Y",
            template=plot_template,
            height=600,
            plot_bgcolor=plot_bg_color,
            paper_bgcolor=plot_bg_color
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif kernel_type == "Linear":
        st.markdown("""
        **Linear Kernel** works in the original input space:
        
        K(x,y) = x·y (dot product)
        
        - No feature transformation needed
        - Creates linear decision boundaries
        - φ(x) = x (identity mapping)
        """)
        
        fig = go.Figure()
        for label, color, name in [(-1, '#ef4444', 'Class -1'), (1, '#3b82f6', 'Class +1')]:
            mask = y == label
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(size=8, color=color),
                name=name
            ))
        
        fig.update_layout(
            title="Original Input Space (No Transformation)",
            xaxis_title="Feature X",
            yaxis_title="Feature Y",
            template=plot_template,
            height=600,
            plot_bgcolor=plot_bg_color,
            paper_bgcolor=plot_bg_color
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(f"Feature space visualization for {kernel_type} kernel - Complex transformation!")

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        st.json({
            "Kernel": kernel_type,
            "Regularization (C)": C,
            "Dataset": dataset_name,
            "Number of Samples": len(X),
            "Number of Features": X.shape[1]
        })
    
    with col2:
        st.markdown("### Model Performance")
        st.json({
            "Accuracy": f"{accuracy * 100:.2f}%",
            "Support Vectors": n_support,
            "SV Ratio": f"{n_support / len(X) * 100:.1f}%",
            "Training Time": f"{compute_time:.2f} ms"
        })
    
    st.markdown("### Kernel Information")
    
    kernel_info = {
        "Linear": "K(x,y) = x·y - Simple dot product, works best for linearly separable data",
        "Polynomial": f"K(x,y) = (x·y + {coef0 if 'coef0' in locals() else 1})^{degree if 'degree' in locals() else 3} - Maps to polynomial feature space",
        "RBF (Gaussian)": f"K(x,y) = exp(-{gamma if 'gamma' in locals() else 0.5}·||x-y||²) - Infinite dimensional space, most flexible",
        "Sigmoid": f"K(x,y) = tanh({gamma if 'gamma' in locals() else 0.1}·x·y + {coef0 if 'coef0' in locals() else 0}) - Neural network inspired",
        "Custom (Laplacian)": f"K(x,y) = exp(-{gamma if 'gamma' in locals() else 0.5}·||x-y||₁) - L1 distance based",
        "Custom (Chi-squared)": "K(x,y) = exp(-χ²(x,y)) - Used for histogram comparisons"
    }
    
    st.info(kernel_info.get(kernel_type, "Custom kernel function"))
    
    st.markdown("### Support Vector Details")
    sv_df = pd.DataFrame(support_vectors, columns=['X', 'Y'])
    sv_df['Class'] = y[svm.support_]
    sv_df['Alpha'] = svm.dual_coef_[0]
    st.dataframe(sv_df, use_container_width=True)

# Footer
st.markdown("---")
footer_color = "#888888" if st.session_state.dark_mode else "#666666"
st.markdown(f"""
<div style='text-align: center; color: {footer_color};'>
    <p>💡 <b>Tips:</b> Try different kernels on different datasets to see how they perform!</p>
    <p>🔴 Circles dataset → RBF kernel | 🔵 XOR pattern → Polynomial/RBF | 📏 Linear data → Linear kernel</p>
    <p style='margin-top: 20px;'>Current Mode: <b>{'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}</b></p>
</div>
""", unsafe_allow_html=True)