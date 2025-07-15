# home.py
import streamlit as st

# ───────────────────────── Cabeçalho ───────────────────────── #
TITLE      = "Classificação de Imagens de Raio-X"
ODS        = "ODS 3 – Saúde e Bem-Estar"
SUBTITLE   = "Trabalho Final – Inteligência Artificial"

PROJECT_DESC = """
O projeto desenvolve **sistemas de detecção automatizada em radiografias**
para apoiar diagnósticos precoces, especialmente em locais com escassez de
especialistas. Utilizamos três abordagens de aprendizado de máquina:

- **CNN** (Redes Convolucionais)  
- **Random Forest**  
- **SVM** (Máquinas de Vetores de Suporte)

Em cada uma, variamos hiperparâmetros e avaliamos como isso impacta a
**acurácia** alcançada.
"""

NAVIGATION_HELP = """
### Como navegar no dashboard

1. **Sidebar à esquerda**  
   Use o menu lateral para escolher a página de interesse:
   - 🧠 CNN  
   - 🌳 Random Forest  
   - 📈 SVM  

2. **Ajuste de parâmetros**  
   Dentro de cada página de modelo há um *formulário* com sliders,
   caixas de seleção e campos numéricos.  
   • Selecione valores de **épocas**, *learning rate*, número de filtros,
   árvores, parâmetro *C* etc.  
   • Clique em **“Treinar modelo”** para executar o experimento.

3. **Resultados em tempo real**  
   Após o treino, o app exibe:  
   - **Acurácia em teste**  
   - Predição em uma imagem aleatória (rótulo real × predito)

4. **Itere e compare**  
   Modifique os hiperparâmetros quantas vezes quiser e observe como as
   métricas mudam. Isso facilita a comparação entre técnicas e a
   compreensão do efeito de cada configuração.

5. **Obtenha o Dataset** 
    Baixe o conjunto **Chest X-Ray Pneumonia** no Kaggle:  
   `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`
    Extraia o arquivo e mantenha a estrutura de treino/teste.
    No campo “📂 Pasta do dataset” de qualquer página de modelo, aponte para
   esse diretório (ex.: `data/chest_xray`) e inicie o treino.
"""

# ───────────────────────── Renderização ───────────────────────── #
def render() -> None:
    st.title(TITLE)
    st.caption(ODS)
    st.subheader(SUBTITLE)

    st.markdown(PROJECT_DESC)

    st.divider()
    st.markdown(NAVIGATION_HELP)

# Execução isolada (debug)
if __name__ == "__main__":
    st.set_page_config(page_title="Home", page_icon="🏠")
    render()
