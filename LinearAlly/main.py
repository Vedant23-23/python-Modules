# main.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from transformations.transform_demo import (
    translation_matrix, rotation_matrix,
    scaling_matrix, shear_matrix, apply_transformation
)

from linear_systems.system_solver import (
    solve_linear_system, get_lu_decomposition, get_qr_decomposition
)
from linear_systems.sensitivity import (
    compute_matrix_norms, compute_condition_number, simulate_sensitivity
)
from fitting.linear_fit import (
    generate_noisy_data, linear_fit, polynomial_fit
)
from mnist.classifier import (
    load_mnist_sample, train_classifier
)

from modules.regularization import (
    generate_regression_data,
    regularized_fit,
    deblur_image
)

from modules.svd_module import (
    compute_svd, low_rank_approximation
)

from modules.power_method_module import (
    power_method
)

from modules.pagerank_module import (
    pagerank
)

st.set_page_config(page_title="LinearAlly – Linear Algebra Explorer", layout="centered")
st.title("🧠 LinearAlly: A Linear Algebra Visual Project")

page = st.sidebar.selectbox("📚 Choose a Module", [
    "📌 2D Transformations",
    "📐 Linear Systems",
    "📊 Sensitivity Analysis",
    "📈 Data Fitting",
    "🔢 MNIST Classification",
    "🧮 Regularization & Deblurring",
    "🧊 SVD & Low Rank",
    "📶 Power Method",
    "🌐 Google PageRank"
])




# --- TAB 1: 2D TRANSFORMATIONS ---
if page == "📌 2D Transformations":
    st.header("🎯 2D Geometric Transformations")

    original_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

    with st.sidebar:
        st.subheader("🔧 Transform Controls")
        tx = st.slider("Translate X", -5.0, 5.0, 0.0)
        ty = st.slider("Translate Y", -5.0, 5.0, 0.0)
        angle = st.slider("Rotation Angle (deg)", -180, 180, 0)
        sx = st.slider("Scale X", 0.1, 3.0, 1.0)
        sy = st.slider("Scale Y", 0.1, 3.0, 1.0)
        shx = st.slider("Shear X", -2.0, 2.0, 0.0)
        shy = st.slider("Shear Y", -2.0, 2.0, 0.0)

    T = translation_matrix(tx, ty)
    R = rotation_matrix(angle)
    S = scaling_matrix(sx, sy)
    H = shear_matrix(shx, shy)

    combined = T @ R @ S @ H
    transformed_points = apply_transformation(original_points, combined)

    fig, ax = plt.subplots()
    ax.plot(original_points[:, 0], original_points[:, 1], 'bo-', label="Original")
    ax.plot(transformed_points[:, 0], transformed_points[:, 1], 'r--o', label="Transformed")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# --- TAB 2: LINEAR SYSTEMS ---
if page == "📐 Linear Systems":

    st.header("📐 Linear Systems & Matrix Decompositions")
    st.write("Solve a system of linear equations **Ax = b** and explore its LU and QR decompositions.")

    size = st.slider("Matrix size (n x n)", min_value=2, max_value=5, value=3, help="Sets the dimension of square matrix A.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        A_input = st.text_area("🔢 Enter Matrix A", "2,1,-1; -3,-1,2; -2,1,2", height=100,
                               help="Rows separated by `;`, values by `,` (e.g. 1,2; 3,4)")
    with col2:
        b_input = st.text_input("🎯 Enter Vector b", "8,-11,-3", help="Comma-separated values (e.g. 1,2,3)")

    # Attempt parsing
    try:
        A = np.array([[float(num) for num in row.split(',')] for row in A_input.strip().split(';')])
        b = np.array([float(x) for x in b_input.strip().split(',')])
    except:
        st.error("❌ Invalid input format. Please check matrix A and vector b.")
        A = np.eye(size)
        b = np.ones(size)

    # LaTeX equation preview
    st.markdown("### 🧮 Equation Preview:")
    st.latex(r"\mathbf{A} \cdot \mathbf{x} = \mathbf{b}")

    # Solve and Decompose
    if st.button("🔎 Solve Ax = b"):
        x, res = solve_linear_system(A, b)

        if x is not None:
            st.success("✅ Solution Vector x:")
            st.write(np.round(x, 4))
            st.info(f"Residual ‖Ax − b‖ = `{res:.4e}`")
        else:
            st.error(f"❌ Could not solve system: {res}")

        # LU Decomposition
        st.subheader("🔍 LU Decomposition")
        try:
            P, L, U = get_lu_decomposition(A)
            st.write("**Permutation Matrix P:**")
            st.write(np.round(P, 4))
            st.write("**Lower Triangular Matrix L:**")
            st.write(np.round(L, 4))
            st.write("**Upper Triangular Matrix U:**")
            st.write(np.round(U, 4))
        except Exception as e:
            st.error(f"LU decomposition failed: {e}")

        # QR Decomposition
        st.subheader("🔍 QR Decomposition")
        try:
            Q, R = get_qr_decomposition(A)
            st.write("**Orthogonal Matrix Q:**")
            st.write(np.round(Q, 4))
            st.write("**Upper Triangular Matrix R:**")
            st.write(np.round(R, 4))
        except Exception as e:
            st.error(f"QR decomposition failed: {e}")


# --- TAB 3: SENSITIVITY ---
if page == "📊 Sensitivity Analysis":
    st.header("📊 Sensitivity Analysis of Linear Systems")

    st.write("Enter Matrix A (rows separated by `;`, values by `,`):")
    A_input = st.text_area("Matrix A", "4,2;1,3")
    st.write("Enter vector b (comma-separated):")
    b_input = st.text_input("Vector b", "10,5")

    try:
        A = np.array([[float(num) for num in row.split(',')] for row in A_input.strip().split(';')])
        b = np.array([float(x) for x in b_input.strip().split(',')])
    except:
        st.error("Invalid matrix or vector format.")
        A = np.eye(2)
        b = np.ones(2)

    if st.button("🧪 Analyze Sensitivity"):
        st.subheader("Matrix Norms")
        norms = compute_matrix_norms(A)
        for name, val in norms.items():
            st.write(f"{name}: {val:.4f}")

        st.subheader("Condition Number")
        cond_num = compute_condition_number(A)
        st.write(f"cond(A) = {cond_num:.2e}")

        st.subheader("Perturbation Test")
        x, x_perturbed, diff, b_perturbed = simulate_sensitivity(A, b)

        if x is not None:
            st.write("Original solution x:")
            st.write(x)
            st.write("Perturbed b:")
            st.write(b_perturbed)
            st.write("Perturbed solution x':")
            st.write(x_perturbed)
            st.success(f"Difference between solutions: {diff:.2e}")
        else:
            st.error("Simulation failed due to singular matrix or size mismatch.")

# --- TAB 4: DATA FITTING ---
if page == "📈 Data Fitting":
    st.header("📈 Geometric Data Fitting & Polynomial Models")

    noise = st.slider("Noise Level", 0.0, 5.0, 1.0)
    degree = st.slider("Polynomial Degree", 1, 5, 1)
    n_samples = st.slider("Number of Data Points", 10, 100, 50)

    x, y = generate_noisy_data(n_samples, noise)
    y_pred_linear, linear_model = linear_fit(x, y)
    y_pred_poly, poly_model = polynomial_fit(x, y, degree)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data", color='gray')
    ax.plot(x, y_pred_linear, label="Linear Fit", color='blue')
    ax.plot(x, y_pred_poly, label=f"Poly Degree {degree}", color='red')
    ax.set_title("Data Fitting")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Model Coefficients")
    st.write(f"Linear: {linear_model.coef_}, Intercept: {linear_model.intercept_}")
    st.write(f"Polynomial Coefs: {poly_model.coef_}, Intercept: {poly_model.intercept_}")

# --- TAB 5: MNIST CLASSIFICATION ---
if page == "🔢 MNIST Classification":
    st.header("🔢 MNIST Classification with Logistic Regression")

    mode = st.radio("Classification Type", ["Binary (0 vs 1)", "Multi-class"])
    poly_degree = st.slider("Polynomial Feature Degree", 1, 3, 1)

    if mode == "Binary (0 vs 1)":
        X, y = load_mnist_sample(binary=True, classes=(0, 1))
    else:
        X, y = load_mnist_sample(binary=False)

    if st.button("Train Classifier"):
        acc, clf, X_test, y_test, y_pred = train_classifier(X, y, poly_degree)
        st.success(f"Accuracy: {acc*100:.2f}%")
        st.subheader("Sample Predictions")
        st.write("True:", y_test[:10].tolist())
        st.write("Pred:", y_pred[:10].tolist())

# --- TAB 6: REGULARIZATION & DEBLURRING ---
if page == "🧮 Regularization & Deblurring":
    st.header("🧮 Regularization & Image Deblurring")

    subtask = st.radio("Choose Task", ["Regularized Regression", "Image Deblurring"])

    if subtask == "Regularized Regression":
        st.subheader("📉 Regularized Regression (Ridge / Lasso)")

        model_type = st.selectbox("Select Model", ["Ridge", "Lasso"])
        alpha = st.slider("Regularization Strength (alpha)", 0.01, 10.0, 1.0)

        X, y = generate_regression_data()
        X_train, y_train, X_plot, y_pred = regularized_fit(X, y, model_type=model_type, alpha=alpha)

        fig, ax = plt.subplots()
        ax.scatter(X_train, y_train, label="Data", color="blue", alpha=0.5)
        ax.plot(X_plot, y_pred, label=f"{model_type} Fit", color="red")
        ax.legend()
        ax.set_title(f"{model_type} Regression with α={alpha}")
        st.pyplot(fig)

    elif subtask == "Image Deblurring":
        st.subheader("🖼 Image Deblurring with Wiener Filter")
        uploaded_image = st.file_uploader("Upload any image (JPG, PNG, BMP, TIFF)", type=["jpg", "jpeg", "png", "bmp", "tiff"])

        if uploaded_image is not None:
            original, blurred, deblurred = deblur_image(uploaded_image)

            col1, col2, col3 = st.columns(3)
            col1.image(np.clip(original, 0.0, 1.0), caption="Original", use_container_width=True)
            col2.image(np.clip(blurred, 0.0, 1.0), caption="Blurred", use_container_width=True)
            col3.image(np.clip(deblurred, 0.0, 1.0), caption="Deblurred", use_container_width=True)

# --- TAB 7: SVD ---
if page == "🧊 SVD & Low Rank":
    st.header("📚 Singular Value Decomposition (SVD)")

    st.write("Enter matrix A (rows separated by `;`, values by `,`):")
    A_input = st.text_area("Matrix A", "3,1,1; -1,3,1")

    try:
        A = np.array([[float(num) for num in row.split(',')] for row in A_input.strip().split(';')])
        m, n = A.shape
        st.success(f"Matrix A ({m}×{n}) successfully loaded.")
    except Exception as e:
        st.error(f"Invalid input: {e}")
        A = None

    if A is not None:
        if st.button("🔍 Compute SVD"):
            U, S, VT = compute_svd(A)

            st.subheader("🧮 SVD Components")
            st.write("U (Left singular vectors):")
            st.write(U)
            st.write("Σ (Singular values):")
            st.write(S)
            st.write("Vᵀ (Right singular vectors):")
            st.write(VT)

            k = st.slider("Select rank k for approximation", 1, min(m, n), 1)
            A_k = low_rank_approximation(U, S, VT, k)

            st.subheader(f"🔁 Rank-{k} Approximation of A")
            st.write(A_k)

            error = np.linalg.norm(A - A_k)
            st.info(f"Reconstruction Error (Frobenius norm): {error:.4e}")

            fig, ax = plt.subplots()
            ax.plot(np.arange(1, len(S)+1), S, 'bo-')
            ax.set_title("Singular Values")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
            st.pyplot(fig)


# --- TAB 8: Power Method ---
if page == "📶 Power Method":
    st.header("⚡ Power Method for Dominant Eigenvalue")

    st.write("Enter a square matrix A (rows separated by `;`, values by `,`):")
    A_input = st.text_area("Matrix A", "4,1; 2,3")

    try:
        A = np.array([[float(num) for num in row.split(',')] for row in A_input.strip().split(';')])
        if A.shape[0] != A.shape[1]:
            st.error("Matrix A must be square.")
            A = None
        else:
            st.success(f"Matrix A ({A.shape[0]}×{A.shape[1]}) successfully loaded.")
    except Exception as e:
        st.error(f"Invalid input: {e}")
        A = None

    if A is not None:
        if st.button("🚀 Run Power Method"):
            eigenvalue, eigenvector = power_method(A)
            st.subheader("✅ Dominant Eigenvalue")
            st.write(eigenvalue)
            st.subheader("🔁 Corresponding Eigenvector")
            st.write(eigenvector)


# --- TAB 9: Google PageRank ---
if page == "🌐 Google PageRank":
    st.header("🌐 Google PageRank via Power Method")

    st.write("Enter a link matrix (rows separated by `;`, values by `,`):")
    link_input = st.text_area("Link Matrix", "0,1,1; 1,0,0; 1,1,0")

    try:
        M = np.array([[float(num) for num in row.split(',')] for row in link_input.strip().split(';')])
        if M.shape[0] != M.shape[1]:
            st.error("Matrix must be square.")
            M = None
        else:
            st.success(f"Link matrix ({M.shape[0]}×{M.shape[1]}) loaded.")
    except Exception as e:
        st.error(f"Invalid input: {e}")
        M = None

    alpha = st.slider("Damping Factor (α)", 0.0, 1.0, 0.85, 0.01)

    if M is not None:
        if st.button("📊 Compute PageRank"):
            ranks = pagerank(M, alpha=alpha)
            st.subheader("🏆 PageRank Scores")
            for i, rank in enumerate(ranks):
                st.write(f"Page {i+1}: {rank:.4f}")
