# pages/random_forest.py
import os
import random
import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_resource(show_spinner=False)
def load_data(path: str, size=(64, 64)):
    """Carrega radiografias NORMAL / PNEUMONIA, redimensiona e normaliza."""
    imgs, y = [], []
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = os.path.join(path, split, cls)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                img = Image.open(os.path.join(folder, f)).convert("L").resize(size)
                imgs.append(np.array(img) / 255.0)
                y.append(0 if cls == "NORMAL" else 1)
    X = np.array(imgs, dtype="float32")      # (N, 64, 64)
    y = np.array(y,  dtype="int8")
    return train_test_split(X, y, test_size=.30, random_state=42)


@st.cache_resource(show_spinner=False)
def extract_hog(images, pixels_per_cell=(8, 8)):
    """Extrai descritor HOG para cada imagem."""
    feats = [
        hog(img, pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2), feature_vector=True)
        for img in images
    ]
    return np.asarray(feats, dtype="float32")

# ‑---------------------- Página -------------------------------------------------
def render() -> None:
    st.title("🌳 Floresta Aleatória (Random Forest)")

    # Definição e componentes --------------------------------------------
    st.header("Definição e principais componentes")
    st.markdown(
        """
        **Floresta Aleatória** é um algoritmo de **aprendizado *ensemble*** que combina diversas árvores de decisão para aumentar **robustez** e **acurácia**. Baseia-se em dois princípios:

        1. **Bootstrapping** – usa subconjuntos de amostras (com reposição).  
        2. **Feature Subsampling** – avalia apenas parte dos atributos em cada nó.
        """
    )

    # Mecanismo de funcionamento -----------------------------------------
    st.header("Mecanismo de funcionamento")

    st.subheader("1 ▸ Construção de árvores")
    st.latex(
        r"""
        \begin{aligned}
        &\text{Para cada árvore } t=1,\dots,T:\\
        1.\;&D_b \subset D\quad (\text{bootstrap})\\
        2.\;&F_b \subset F\quad (\text{subamostra de atributos})\\
        3.\;&\text{Treina-se a árvore em }(D_b, F_b)\\
        4.\;&\text{Predição }f_t(\mathbf{x})
        \end{aligned}
        """
    )

    st.subheader("2 ▸ Combinação de previsões")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Classificação")
        st.latex(
            r"""
            \hat{y}(\mathbf{x}) =
            \operatorname*{arg\,max}_{k}\sum_{t=1}^{T}
            \mathbb{I}\{f_t(\mathbf{x}) = k\}
            """
        )
    with col2:
        st.caption("Regressão")
        st.latex(
            r"""
            \hat{y}(\mathbf{x}) = \frac{1}{T}\sum_{t=1}^{T} f_t(\mathbf{x})
            """
        )

    # Parâmetros principais ----------------------------------------------
    st.header("Parâmetros principais")
    st.table(
        {
            "Parâmetro": ["`n_estimators`", "`max_depth`",
                          "`min_samples_split`", "`max_features`"],
            "Função": [
                "Número de árvores",
                "Profundidade máxima",
                "Mínimo de amostras p/ dividir",
                "Atributos avaliados por nó",
            ],
            "Valores típicos": ["100 · 500", "5 · 15", "2 · 10", "'sqrt', 'log2'"],
        }
    )

    # Regularização -------------------------------------------------------
    st.header("Mecanismos de regularização")
    st.table(
        {
            "Parâmetro": ["`max_depth`", "`min_samples_split`", "`max_features`"],
            "Efeito": [
                "Limita complexidade da árvore",
                "Evita divisões em grupos minúsculos",
                "Aumenta diversidade entre árvores",
            ],
        }
    )

    # Importância de atributos -------------------------------------------
    st.header("Equação de importância de atributos")
    st.latex(
        r"""
        I(f)=\frac{1}{N}\sum_{i=1}^{N}
        \frac{\Delta \text{Impureza}_i(f)}{\text{Árvore } i}
        """
    )
    st.markdown(
        "Onde **$\\Delta \\text{Impureza}_i(f)$** é o ganho total atribuído ao "
        "atributo **$f$** em todas as divisões da árvore *i*."
    )

    # Treino interativo ---------------------------------------------------
    st.divider()
    st.header("Configurações de treino")

    with st.form("rf_params"):
        data_dir = st.text_input("📂 Pasta do dataset `chest_xray/`",
                                 value="dashboard/data/pneumonia")

        col1, col2, col3 = st.columns(3)
        n_estimators = col1.slider("Árvores (`n_estimators`)", 50, 500, 200, 50)
        max_depth    = col2.slider("Profundidade máx.", 2, 20, 10)
        max_features = col3.selectbox("`max_features`",
                                      ["sqrt", "log2", None], index=0)

        submitted = st.form_submit_button("🚂 Treinar modelo")

    if submitted:
        if not os.path.isdir(data_dir):
            st.error("Diretório inválido.")
            return

        # Dados --------------------------------------------------------------
        X_tr_img, X_te_img, y_tr, y_te = load_data(data_dir)
        X_tr, X_te = extract_hog(X_tr_img), extract_hog(X_te_img)

        # Modelo -------------------------------------------------------------
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=-1,
            random_state=42,
        )

        with st.spinner("Treinando…"):
            rf.fit(X_tr, y_tr)

        acc = accuracy_score(y_te, rf.predict(X_te))
        st.success(f"Acurácia em teste: **{acc:.2%}**")

        # Amostra aleatória --------------------------------------------------
        idx = random.randrange(len(X_te_img))
        img = X_te_img[idx]
        pred = rf.predict([X_te[idx]])[0]
        truth = y_te[idx]

        colA, colB = st.columns(2)
        with colA:
            st.image(img, width=260,
                     caption="Previsão aleatória selecionada")
        with colB:
            st.metric("Verdadeiro",
                      "Pneumonia" if truth else "Normal")
            st.metric("Predito",
                      "Pneumonia" if pred else "Normal")
            st.metric("Acurácia (global)", f"{acc:.2%}")
# Execução isolada -------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Random Forest", page_icon="🌳", layout="wide")
    render()
