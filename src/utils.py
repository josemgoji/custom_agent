from langchain.prompts import ChatPromptTemplate
from pathlib import Path
import streamlit as st
import re
import pandas as pd
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- seleccionar preguntas ---
df_path = Path('./data/preguntas_estadistica_niveles.csv')
df = pd.read_csv(df_path)

def seleccionar_preguntas(nivel: str):
    df_filtrado = df[df["nivel"] == nivel.lower()]
    temas = df_filtrado["tema"].unique()
    seleccionadas = []

    for tema in temas:
        preguntas_tema = df_filtrado[df_filtrado["tema"] == tema]
        seleccionadas.append(preguntas_tema.sample(1).iloc[0].to_dict())

    return seleccionadas

#--- cargar promts ---
def crear_prompt(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_prompt = f.read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", texto_prompt)
    ])
    return prompt

#--- mostrar feedback ---
def mostrar_feedback():
    fortalezas_md = "\n".join(f"- {item}" for item in st.session_state["fortalezas"])        
    st.markdown("### Fortalezas")
    st.markdown(fortalezas_md)

    debilidades_md = "\n".join(f"- {item}" for item in st.session_state["debilidades"])
    st.markdown("### Debilidades")
    st.markdown(debilidades_md)
    
    st.markdown(f"### Detalle por pregunta nivel: {st.session_state['nivel']}")
    
    for d in st.session_state["detalle"]:
        st.markdown(f"**Pregunta:** {d['pregunta']}")
        st.markdown(f"**Respuesta:** {d['respuesta']}")
        st.markdown(f"**Tema:** {d['tema']}")
        st.markdown(f"**Puntaje:** {d['puntaje']}")
        st.markdown(f"**Feedback:** {d['feedback']}")
        st.markdown("---")

#--- corregir markdown ----
def corregir_latex_llm(texto):
    # Reemplaza [ ... ] por ecuaciones en bloque ($$...$$)
    texto = re.sub(r'\[\s*(.*?)\s*\]', r'$$\1$$', texto)

    # Corrige posibles duplicaciones por listas numeradas (opcional, si ves esto)
    texto = re.sub(r'\n\d+\.\s*', r'\n\n### ', texto)

    return texto