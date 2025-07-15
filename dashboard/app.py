import streamlit as st
import importlib

st.set_page_config(page_title="Trabalho final IA", page_icon="🔍", layout="wide")

tabs = {
    "🏠 Home": "home",
    "🧠 CNN": "pages.cnn",
    "🌳 Random Forest": "pages.random_forest",
    "📈 SVM": "pages.svm",
}

# Criar barra de seleção
choice = st.sidebar.radio("Navegação", list(tabs.keys()))

# Importar módulo dinamicamente e renderizar
module = importlib.import_module(tabs[choice])
module.render()
