# app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from graphs import build_graphs
from utils import seleccionar_preguntas, mostrar_feedback, corregir_latex_llm

load_dotenv()

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
if "temas" not in st.session_state:
    st.session_state["temas"] = []
if "subtemas" not in st.session_state:
    st.session_state["subtemas"] = {}
if "tema_actual" not in st.session_state:
    st.session_state["tema_actual"] = 0
if "contexto_recuperado" not in st.session_state:
    st.session_state["contexto_recuperado"] = {}
if "explicacion" not in st.session_state:
    st.session_state["explicacion"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if "modo_anterior" not in st.session_state:
    st.session_state["modo_anterior"] = None
    
# --- Botón flotante para cambiar de modo ---
with st.sidebar:
    if st.session_state.get("modo_detectado", False):
        st.success(f"Modo activo: {st.session_state['modo'].capitalize()}")
    
    st.markdown("## Cambiar de Modo")
    if st.session_state["modo"] == "libre":
        if st.button("Modo Guiado"):
            st.session_state["modo_anterior"] = st.session_state["modo"]
            st.session_state["modo"] = "guiado"
            st.session_state["modo_detectado"] = True
            st.rerun()
    else:    
        if st.button("Modo Libre"):
            st.session_state["modo_anterior"] = st.session_state["modo"]
            st.session_state["modo"] = "libre"
            st.session_state["modo_detectado"] = True
            st.rerun()

    if st.button("Modo Anterior"):
        if st.session_state["modo_anterior"] is not None:
            st.session_state["modo"] = st.session_state["modo_anterior"]
            st.session_state["modo_detectado"] = True
            st.rerun()
        else:
            st.warning("No hay un modo anterior para volver.")
    
st.title("Agente asitente de estudio de estadística custom")

graph_feedback, graph_plan, graph_explicacion, graph_libre = build_graphs()

# --- Paso 1: Elegir modo de interacción ---
if not st.session_state["modo_detectado"]:
    st.markdown("## Bienvenido ✨")
    st.markdown("""
    Elige cómo quieres interactuar con el agente de estudio:
    
    - **Modo Guiado**: Responde un quiz para evaluar tu nivel, y luego te guiare paso a paso en el estudio.
    - **Modo Libre**: Pregunta cualquier cosa de estadística sin seguir una ruta fija.
    """)
    
    modo = st.radio(
        "Selecciona el modo que prefieras:",
        ("guiado", "libre"),
        key="seleccion_modo"
    )
    
    if st.button("Confirmar modo"):
        st.session_state["modo"] = modo
        st.session_state["modo_detectado"] = True
        st.success(f"Modo seleccionado: {modo.capitalize()}")
        st.rerun()

# --- modo guiado ---
if st.session_state["modo"] == "guiado":
    # --- realizar el quiz ---
    # crear las preguntas
    if not st.session_state["preguntas_seleccionadas"]:
        st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas(st.session_state["nivel"])

    idx = st.session_state.get("pregunta_idx", 0)
    preguntas_seleccionadas = st.session_state["preguntas_seleccionadas"]

    # mostrar las preguntas una a una
    if idx < len(preguntas_seleccionadas):
        st.subheader(f"Nivel actual: {st.session_state['nivel'].capitalize()}")
        st.markdown("Primero te haremos un quiz para ver en qué nivel estás.")
        pregunta = preguntas_seleccionadas[idx]["pregunta"]
        st.markdown(f"**Pregunta {idx+1}:** {pregunta}")
        respuesta = st.text_area("Tu respuesta:", key=f"respuesta_input_{idx}_{st.session_state['rerun_counter']}")
        if st.button("Enviar respuesta", key=f"btn_respuesta_{idx}_{st.session_state['rerun_counter']}") or respuesta:
            respuestas = st.session_state.get("respuestas", [])
            respuestas.append(respuesta)
            st.session_state["respuestas"] = respuestas
            st.session_state["pregunta_idx"] = idx + 1
            st.session_state["rerun_counter"] += 1
            st.rerun()
    
    # --- invocar el agente de feedback para calificar ---
    # cuando termina de responder las preguntas
    if idx >= len(preguntas_seleccionadas):
        if not st.session_state.get("feedback", {}):
            with st.spinner("🕒 Calificando el quiz"):
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
        # Clasificar el resultado y actualizar el estado del quiz
        if promedio >= 3:
            
            mostrar_feedback()
            
            if st.session_state["nivel"] == "basico":
                st.success(f"¡Felicidades! Has aprobado el nivel básico con un promedio de {promedio}.")
                st.session_state["nivel"] = "intermedio"
                st.session_state["pregunta_idx"] = 0
                st.session_state["respuestas"] = []
                st.session_state["feedback"] = {}
                st.session_state["detalle"] = []
                st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas("intermedio")
                
                if st.button("Continuar", key="btn_next_lv"):
                    st.rerun()
                
            elif st.session_state["nivel"] == "intermedio":
                st.success(f"¡Felicidades! Has aprobado el nivel intermedio con un promedio de {promedio}.")
                st.session_state["nivel"] = "avanzado"
                st.session_state["pregunta_idx"] = 0
                st.session_state["respuestas"] = []
                st.session_state["feedback"] = {}
                st.session_state["detalle"] = []
                st.session_state["preguntas_seleccionadas"] = seleccionar_preguntas("avanzado")
                
                if st.button("Continuar", key="btn_next_lv"):
                    st.rerun()
                
            else:
                st.success(f"¡Felicidades! Has aprobado todos los niveles con un promedio de {promedio}!")
                st.session_state["estado_quiz"] = "finalizado"
        else:
            st.error(f"No has aprobado el nivel {st.session_state['nivel']}. Tu promedio fue {promedio}.")
            st.session_state["estado_quiz"] = "finalizado"
            
            
        # --- Mostrar el feedback y las fortalezas/debilidades ---
        # cuando termina el examen si aprueba todo o reprueba un nivel
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
                    
            mostrar_feedback()        
            
# -- modo de explicacion --
elif st.session_state["modo"] == "estudio":
    st.title("Plan de Estudio")

    if "plan_actual" not in st.session_state:
        st.session_state["plan_actual"] = None

    # Si no hay un plan generado, generarlo
    if not st.session_state["plan_actual"]:
        with st.spinner("🕒 Cargando el plan de estudio..."):
            plan = graph_plan.invoke({
                'modo': st.session_state["modo"],
                "nivel": st.session_state["nivel"],
                "debilidades": st.session_state["debilidades"],
                "explicacion": st.session_state["explicacion"],
                
            })
            st.session_state["temas"] = plan.get("temas", [])
            st.session_state["subtemas"] = plan.get("subtemas", {})
            st.session_state["tema_actual"] = plan.get("tema_actual", 0)
            st.session_state["plan_actual"] = plan
            
        st.rerun()            

    # Mostrar el plan actual
    subtemas = st.session_state["plan_actual"].get("subtemas", {})
    for tema, lista_subtemas in subtemas.items():
        st.markdown(f"### {tema}")  
        for subtema in lista_subtemas:
            st.markdown(f"- {subtema}")  
    
    if st.button("Continuar", key="btn_next"):
        st.session_state["modo"] = "explicacion"
        st.rerun()  

# -- modo de explicación (nueva sección) --
elif st.session_state["modo"] == "explicacion":
    tema = st.session_state["tema_actual"]

    if "explicaciones" not in st.session_state:
        st.session_state["explicaciones"] = {}

    if tema < len(st.session_state["temas"]):
        st.title("Explicación del Plan de Estudio")

        if tema not in st.session_state["explicaciones"]:
            # SOLO muestra spinner si de verdad vas a generar la explicación
            with st.spinner(f"🕒 Cargando explicación del tema {st.session_state['temas'][tema]}"):
                explicacion = graph_explicacion.invoke({
                    "tema_actual": tema,
                    "nivel": st.session_state["nivel"],
                    "subtemas": st.session_state["subtemas"],
                    "temas": st.session_state["temas"],
                    "contexto_recuperado": st.session_state["contexto_recuperado"],
                    "modo": st.session_state["modo"],
                })
                st.session_state["explicaciones"][tema] = explicacion["explicacion"]

            st.toast("Explicación lista ✅", icon="📚")

        # Mostrar la explicación ya guardada
        explicacion_corregida = corregir_latex_llm(st.session_state["explicaciones"][tema])
        st.markdown(explicacion_corregida, unsafe_allow_html=True)

        st.write("---")
        st.write("¿Pasamos al siguiente tema?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Siguiente tema", key=f"btn_siguiente_tema_{tema}"):
                st.session_state["tema_actual"] += 1
                st.session_state["contexto_recuperado"] = {}
                st.rerun()
    else:
        st.write("---")
        st.write("felicidades has terminado el plan de estudio, hagamos nuevamente el examen para reforzar los ocnocimientos")
        
        if st.button("Continuar", key="btn_continuar"):
            st.session_state['modo'] = "guiado"
            st.session_state['estado_quiz'] = "en_curso"
            st.session_state["pregunta_idx"] = 0
            st.session_state["plan_aprobado"] = None
            st.session_state["debilidades"] = []
            st.session_state["fortalezas"] = []
            st.session_state["respuestas"] = []
            st.session_state["tema_actual"] = 0
            st.session_state["plan_actual"] = None
            st.session_state["temas"] = []
            st.session_state["subtemas"] = {}
            st.session_state["explicacion"] = ""
            st.session_state["contexto_recuperado"] = {}
            st.session_state["explicaciones"] = {}
            st.rerun()

elif st.session_state["modo"] == "libre":
    st.write("Estás en modo libre. Aquí puedes hacer preguntas directamente.")
    user_input = st.chat_input("Escriba su pregunta aquí:")

    if user_input:
        # Mostrar inmediatamente el mensaje del usuario
        st.session_state.message_history.append({'content': user_input, 'type': 'user'})

        # Mostrar el historial hasta ahora (incluye el mensaje del usuario)
        for message in st.session_state.message_history:
            with st.chat_message(message['type']):
                st.markdown(message['content'])

        # Crear un contenedor vacío para la respuesta del agente
        response_placeholder = st.empty()

        # Mostrar spinner mientras se genera la respuesta
        with st.spinner("El agente está pensando..."):
            respuesta = graph_libre.invoke({
                "messages": [HumanMessage(content=user_input)],
            })
            contenido_respuesta = respuesta["messages"][-1].content

        # Añadir la respuesta al historial
        st.session_state.message_history.append({'content': contenido_respuesta, 'type': 'assistant'})

        # Actualizar el contenedor con la respuesta del agente
        with response_placeholder:
            with st.chat_message("assistant"):
                st.markdown(contenido_respuesta)
    