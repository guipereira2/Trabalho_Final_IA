# pages/svm.py
import os
import random
import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

@st.cache_resource(show_spinner=False)
def load_data(path: str, size=(64, 64)):
    """Carrega imagens (grayscale), redimensiona, normaliza e faz split"""
    imgs, labels = [], []
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = os.path.join(path, split, cls)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                img = Image.open(os.path.join(folder, f)).convert("L").resize(size)
                imgs.append(np.array(img) / 255.0)
                labels.append(0 if cls == "NORMAL" else 1)
    X = np.array(imgs, dtype="float32")
    y = np.array(labels, dtype="int8")
    return train_test_split(X, y, test_size=.30, random_state=42)

@st.cache_resource(show_spinner=False)
def extract_hog(images):
    """Extrai descritores HOG de cada imagem."""
    return np.array(
        [hog(im, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), feature_vector=True) for im in images],
        dtype="float32"
    )


def render() -> None:
    st.title("📈 Máquinas de Vetores de Suporte (SVM)")

    # Definição ----------------------------------------------------------------
    st.header("Definição")
    st.latex(
        r"""
        f(\mathbf{x}) =
        \operatorname{sgn}\!\Bigl(
        \sum_{i=1}^{n} \alpha_i y_i K(\mathbf{x}_i,\mathbf{x}) + b
        \Bigr)
        """
    )
    st.markdown(
        """
        **Componentes principais**

        - $\\alpha_i$: coeficiente do **vetor de suporte** $\\mathbf{x}_i$  
        - $y_i \\in \\{-1, +1\\}$: rótulo da amostra  
        - $K(\\mathbf{x}_i, \\mathbf{x})$: **kernel** (linear, RBF etc.)  
        - $b$: viés (bias)
        """
    )

    # Kernels ------------------------------------------------------------------
    st.header("Kernels mais usados")
    with st.expander("Linear"):
        st.latex(r"K(\mathbf{x}_i,\mathbf{x}) = \mathbf{x}_i^{\top}\mathbf{x}")
    with st.expander("RBF – Radial Basis Function"):
        st.latex(r"K(\mathbf{x}_i,\mathbf{x}) = \exp\!\bigl(-\gamma \|\mathbf{x}_i-\mathbf{x}\|^{2}\bigr)")
    with st.expander("Polinomial"):
        st.latex(r"K(\mathbf{x}_i,\mathbf{x}) = (\mathbf{x}_i^{\top}\mathbf{x}+c)^{d}")

    # Regularização ------------------------------------------------------------
    st.header("Regularização e função de custo")
    st.latex(
        r"""
        J(\mathbf{w},\boldsymbol{\xi}) =
        \tfrac12\|\mathbf{w}\|^{2}
        + C \sum_{i=1}^{n} \xi_i
        \quad
        \text{sujeito a } y_i(\mathbf{w}^{\top}\phi(\mathbf{x}_i)+b) \ge 1-\xi_i
        """
    )
    st.markdown(
        """
        - **$C$** controla o equilíbrio **margem × erros**  
        - **$\\xi_i$** mede a violação da margem
        """
    )

    # Trade-off ----------------------------------------------------------------
    st.header("Trade-off crítico")
    st.latex(
        r"""
        \text{Maximizar margem} \;(\Downarrow \|\mathbf{w}\|)
        \quad\longleftrightarrow\quad
        \text{Minimizar erros} \;(\Uparrow \sum\xi_i)
        """
    )
    st.markdown(
        """
        - $C > 1$: prioriza **ajuste** (margem menor)  
        - $C < 1$: prioriza **generalização** (margem maior)
        """
    )

    # Área prática -------------------------------------------------------------
    st.divider()
    st.header("Configurações de treino")

    with st.form("svm_form"):
        data_dir = st.text_input("📂 Pasta do dataset `chest_xray/`",
                                 value="dashboard/data/pneumonia")

        col1, col2, col3 = st.columns(3)
        kernel   = col1.selectbox("Kernel", ["linear", "rbf"], index=0)
        C_param  = col2.slider("C (regularização)", 0.1, 5.0, 1.0, 0.1)
        gamma    = col3.select_slider(
            "Gamma (RBF)", options=["scale", "auto"], value="scale") if kernel == "rbf" else None

        submitted = st.form_submit_button("🚂 Treinar modelo")

    if submitted:
        if not os.path.isdir(data_dir):
            st.error("Diretório inválido.")
            return

        X_train_img, X_test_img, y_train, y_test = load_data(data_dir)
        X_train, X_test = extract_hog(X_train_img), extract_hog(X_test_img)

        svm = SVC(kernel=kernel, C=C_param, gamma=gamma if kernel == "rbf" else "auto")
        with st.spinner("Treinando…"):
            svm.fit(X_train, y_train)

        acc = accuracy_score(y_test, svm.predict(X_test))
        st.success(f"Acurácia em teste: **{acc:.2%}**")

        # Previsão aleatória ---------------------------------------------------
        idx  = random.randrange(len(X_test_img))
        img  = X_test_img[idx]
        pred = svm.predict(X_test[idx].reshape(1, -1))[0]
        true = y_test[idx]
        colA, colB = st.columns(2)
        with colA:
            st.image(img.squeeze(), width=250, caption="Previsão aleatória selecionada")
        with colB:
            st.metric("Verdadeiro", "Pneumonia" if true else "Normal")
            st.metric("Predito",  "Pneumonia" if pred else "Normal")
            st.metric("Acurácia (global)", f"{acc:.2%}")

# Execução isolada -------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="SVM", page_icon="📈", layout="wide")
    render()
