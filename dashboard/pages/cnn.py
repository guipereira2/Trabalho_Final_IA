# pages/cnn.py
import os
import random
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam

@st.cache_resource(show_spinner=False)
def load_data(path: str, size: tuple[int, int] = (64, 64)):
    """Lê imagens (NORMAL / PNEUMONIA), redimensiona e normaliza."""
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
    X = np.expand_dims(np.array(imgs, dtype="float32"), -1)
    y = np.array(labels, dtype="int8")
    return train_test_split(X, y, test_size=0.30, random_state=42)


def build_cnn(filters: int, kernel: tuple[int, int],
              dropout: float, lr: float) -> tf.keras.Model:
    """Cria e compila a CNN de acordo com os hiperparâmetros."""
    model = Sequential([
        Conv2D(filters, kernel, activation="relu", input_shape=(64, 64, 1)),
        MaxPooling2D(),
        Dropout(dropout),
        Flatten(),
        Dense(filters, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ───────────────────────── Página principal ───────────────────────── #
def render() -> None:
    st.title("🧠 Redes Convolucionais (CNN)")

    # Parte teórica ------------------------------------------------------ #
    st.header("Definição")
    st.markdown("""
    As **redes convolucionais (CNN)** são arquiteturas especializadas em
    processar dados com estrutura espacial (imagens, séries temporais etc.).
    Utilizam filtros convolucionais para extrair **características locais**,
    reduzindo parâmetros e melhorando a generalização.
    """)

    st.header("Campos Receptivos Locais")
    st.markdown("""
    - Células no cérebro respondem a regiões restritas do campo visual\n
    - CNNs replicam essa ideia: cada neurônio “enxerga” apenas uma parte
      da entrada.
    """)

    st.header("Comparação com Redes MLP")
    st.markdown("""
    - **MLPs:** conexões globais → muitos parâmetros\n
    - **CNNs:** pesos **compartilhados** em filtros locais → modelo menor.
    """)

    st.header("Operação de Convolução")
    st.latex(r"f_{\text{out}}(x)=\sum_{k=-M}^{M} f(x-k)\,g(k)")
    st.markdown("Em 2-D aplicamos o mesmo princípio a todos os canais da imagem.")

    st.header("Camada de *Pooling*")
    st.markdown("""
    - Reduz o tamanho do *feature map*\n
    - Diminui custo computacional e adiciona **invariância à translação**.
    """)

    st.header("Stride e *Padding*")
    st.markdown("""
    - **Stride:** passo da janela convolucional (ex.: 1 ou 2)\n
    - **Padding:** preenchimento das bordas para controlar a saída.
    """)

    st.divider()

    # Formulário de hiperparâmetros ------------------------------------- #
    with st.form("params"):
        st.subheader("Configurações de treino")

        data_dir = st.text_input("📂 Pasta do dataset `chest_xray/`",
                                 value="data/chest_xray")

        col1, col2, col3 = st.columns(3)
        epochs   = col1.slider("Épocas", 1, 20, 10)
        batch    = col2.selectbox("Batch size", [16, 32, 64], index=1)
        lr       = col3.select_slider(
            "Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3, format_func=lambda x: f"{x:.0e}"
        )

        col4, col5, col6 = st.columns(3)
        filters  = col4.number_input("Nº filtros", 16, 128, 32, 16)
        kernel   = col5.selectbox("Kernel", [(3, 3), (5, 5)], index=0)
        dropout  = col6.slider("Dropout", 0.0, 0.5, 0.2, 0.05)

        submitted = st.form_submit_button("🚂 Treinar modelo")

    # 3 Execução do treino ------------------------------------------------- #
    if submitted:
        if not os.path.isdir(data_dir):
            st.error("Diretório inválido. Verifique o caminho.")
            return

        X_train, X_test, y_train, y_test = load_data(data_dir)
        model = build_cnn(filters, kernel, dropout, lr)

        with st.spinner("Treinando…"):
            history = model.fit(
                X_train, y_train,
                validation_split=0.20,
                epochs=epochs,
                batch_size=batch,
                verbose=0
            )

        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        st.success(f"Acurácia em teste: **{acc:.2%}**")
        #st.line_chart(history.history["val_accuracy"])

        # Previsão aleatória -------------------------------------------- #
        idx  = random.randrange(len(X_test))
        img  = X_test[idx]
        prob = float(model.predict(img[None, ...], verbose=0)[0, 0])
        pred = "Pneumonia" if prob > 0.5 else "Normal"
        true = "Pneumonia" if y_test[idx] else "Normal"

        colA, colB = st.columns(2)
        with colA:
            st.image(img.squeeze(), width=250, caption="Previsão aleatória selecionada")
        with colB:
            st.metric("Verdadeiro", true)
            st.metric("Predito",  pred)
            st.metric("Acurácia (global)", f"{acc:.2%}")

if __name__ == "__main__":
    st.set_page_config(page_title="CNN", page_icon="🧠", layout="wide")
    render()
