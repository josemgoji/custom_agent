# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from graphs import (
    build_graphs,
    seleccionar_preguntas,
    quiz_preguntas
)

# --- Cargar variables de entorno y API Key ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Inicialización de session_state ---
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "modo" not in st.session_state:
    st.session_state["modo"] = ""
if "pregunta_idx" not in st.session_state:
    st.session_state["pregunta_idx"] = 0
if "respuestas" not in st.session_state:
    st.session_state["respuestas"] = []
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "nivel" not in st.session_state:
    st.session_state["nivel"] = "basico"
if "fortalezas" not in st.session_state:
    st.session_state["fortalezas"] = []
if "debilidades" not in st.session_state:
    st.session_state["debilidades"] = []
if "puntaje_promedio" not in st.session_state:
    st.session_state["puntaje_promedio"] = 0
if "detalle" not in st.session_state:
    st.session_state["detalle"] = []
if "preguntas_seleccionadas" not in st.session_state:
    st.session_state["preguntas_seleccionadas"] = []
if "rerun_counter" not in st.session_state:
    st.session_state["rerun_counter"] = 0
if "estado_quiz" not in st.session_state:
    st.session_state["estado_quiz"] = "en_curso"
if "modo_detectado" not in st.session_state:
    st.session_state["modo_detectado"] = False

st.title("Agente asitente de estudio de estadística")

graph_modo, graph_feedback = build_graphs()

# --- Paso 1: Entrada de texto libre ---
if not st.session_state["user_input"]:
    st.markdown("### Bienvenido")
    st.markdown("Por favor, escribe cómo quieres interactuar con el agente.")
    user_input = st.text_input(
        "¿Cómo quieres interactuar con el agente? (Ejemplo: 'Quiero que me guíes' o 'Solo respóndeme')"
    )
    if st.button("Enviar"):
        st.session_state["user_input"] = user_input
        st.rerun()

# --- Detectar el modo solo una vez ---
if st.session_state["user_input"] and not st.session_state.get("modo_detectado", False):
    result = graph_modo.invoke({
        "user_input": st.session_state["user_input"],
        "modo": "",
        "pregunta_idx": 0,
        "respuestas": [],
        "feedback": {},
        "nivel": "",
        "fortalezas": [],
        "debilidades": [],
        "puntaje_promedio": 0,
        "detalle": [],
        "preguntas_seleccionadas": []
    })
    if "modo" in result:
        st.session_state["modo"] = result["modo"]
        st.session_state["modo_detectado"] = True

if st.session_state.get("modo_detectado", False):
    st.success(f"Modo detectado: {st.session_state['modo']}")

# --- Lógica para el modo guiado ---
if st.session_state["modo"] == "guiado":
    if not st.session_state["preguntas_seleccionadas"]:
        st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas(st.session_state["nivel"])

    idx = st.session_state.get("pregunta_idx", 0)
    preguntas_seleccionadas = st.session_state["preguntas_seleccionadas"]

    if idx < len(preguntas_seleccionadas):
        st.subheader(f"Nivel actual: {st.session_state['nivel'].capitalize()}")
        st.markdown("Primero te haremos un quiz para ver en qué nivel estás.")
        pregunta = preguntas_seleccionadas[idx]["pregunta"]
        st.markdown(f"**Pregunta {idx+1}:** {pregunta}")
        respuesta = st.text_input("Tu respuesta:", key=f"respuesta_input_{idx}_{st.session_state['rerun_counter']}")
        if st.button("Enviar respuesta", key=f"btn_respuesta_{idx}_{st.session_state['rerun_counter']}"):
            respuestas = st.session_state.get("respuestas", [])
            respuestas.append(respuesta)
            st.session_state["respuestas"] = respuestas
            st.session_state["pregunta_idx"] = idx + 1
            st.session_state["rerun_counter"] += 1
            st.rerun()

    if idx >= len(preguntas_seleccionadas):
        if not st.session_state.get("feedback", {}):
            result = graph_feedback.invoke({
                "user_input": st.session_state["user_input"],
                "modo": st.session_state["modo"],
                "pregunta_idx": len(preguntas_seleccionadas),
                "respuestas": st.session_state["respuestas"],
                "feedback": {},
                "nivel": st.session_state["nivel"],
                "fortalezas": [],
                "debilidades": [],
                "puntaje_promedio": 0,
                "detalle": [],
                "preguntas_seleccionadas": st.session_state["preguntas_seleccionadas"]
            })
            st.session_state["feedback"] = result.get("feedback", {})
            st.session_state["detalle"] = result.get("detalle", [])
            total_puntajes = len(st.session_state["detalle"])
            if total_puntajes > 0:
                st.session_state["puntaje_promedio"] = round(
                    sum(d["puntaje"] for d in st.session_state["detalle"]) / total_puntajes, 2
                )
            st.session_state["fortalezas"] = list(set(st.session_state.get("fortalezas", []) + result.get("fortalezas", [])))
            st.session_state["debilidades"] = list(set(st.session_state.get("debilidades", []) + result.get("debilidades", [])))
            st.rerun()

        promedio = st.session_state["puntaje_promedio"]
        if promedio >= 3:
            if st.session_state["nivel"] == "basico":
                st.success(f"¡Felicidades! Has aprobado el nivel básico con un promedio de {promedio}.")
                st.session_state["nivel"] = "intermedio"
                st.session_state["pregunta_idx"] = 0
                st.session_state["respuestas"] = []
                st.session_state["feedback"] = {}
                st.session_state["detalle"] = []
                st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas("intermedio")
                st.rerun()
            elif st.session_state["nivel"] == "intermedio":
                st.success(f"¡Felicidades! Has aprobado el nivel intermedio con un promedio de {promedio}.")
                st.session_state["nivel"] = "avanzado"
                st.session_state["pregunta_idx"] = 0
                st.session_state["respuestas"] = []
                st.session_state["feedback"] = {}
                st.session_state["detalle"] = []
                st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas("avanzado")
                st.rerun()
            else:
                st.success(f"¡Felicidades! Has aprobado todos los niveles con un promedio de {promedio}!")
                st.session_state["estado_quiz"] = "finalizado"
        else:
            st.error(f"No has aprobado el nivel {st.session_state['nivel']}. Tu promedio fue {promedio}.")
            st.session_state["estado_quiz"] = "finalizado"

        if st.session_state["modo"] == "guiado" and st.session_state["estado_quiz"] == "finalizado":
            st.markdown("### ¿Qué deseas hacer ahora?")
            opcion = st.radio(
                "Selecciona una opción:",
                ("Repetir el examen", "Comenzar a estudiar"),
                key="opcion_post_feedback"
            )

            if st.button("Continuar", key="btn_continuar_post_feedback"):
                if opcion == "Repetir el examen":
                    st.session_state["nivel"] = "basico"
                    st.session_state["pregunta_idx"] = 0
                    st.session_state["respuestas"] = []
                    st.session_state["feedback"] = {}
                    st.session_state["fortalezas"] = []
                    st.session_state["debilidades"] = []
                    st.session_state["detalle"] = []
                    st.session_state["preguntas_seleccionadas"] = []
                    st.session_state["estado_quiz"] = "en_curso"
                    st.session_state["modo"] = "guiado"
                    st.session_state["modo_detectado"] = True
                    st.session_state["user_input"] = st.session_state["user_input"]
                    st.rerun()
                elif opcion == "Comenzar a estudiar":
                    st.session_state["modo"] = "estudio"
                    st.session_state["estado_quiz"] = "en_curso"
                    st.rerun()
            st.markdown("### Feedback del examen")
            st.info(f"**Puntaje promedio del nivel:** {st.session_state['puntaje_promedio']}")
            st.markdown("### Fortalezas")
            st.write(st.session_state["fortalezas"])
            st.markdown("### Debilidades")
            st.write(st.session_state["debilidades"])
            st.markdown(f"### Detalle por pregunta nivel: {st.session_state['nivel']}")
            
            for d in st.session_state["detalle"]:
                st.markdown(f"**Pregunta:** {d['pregunta']}")
                st.markdown(f"**Respuesta:** {d['respuesta']}")
                st.markdown(f"**Tema:** {d['tema']}")
                st.markdown(f"**Puntaje:** {d['puntaje']}")
                st.markdown(f"**Feedback:** {d['feedback']}")
                st.markdown("---")

elif st.session_state["modo"] == "estudio":
    st.success("¡Bienvenido al estudio!")

elif st.session_state["modo"] == "libre":
    st.write("Estás en modo libre. Aquí puedes hacer preguntas directamente.")