from pyexpat.errors import messages
from langgraph.graph import StateGraph, END , START
from openai import chat
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain.tools import Tool
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
import numpy as np
import random
from dotenv import load_dotenv


load_dotenv()

quiz_preguntas = {
    "basico": [
        {"tema": "Variables aleatorias", "pregunta": "¿Qué es una variable aleatoria?"},
        {"tema": "Variables aleatorias", "pregunta": "¿Cuál es la diferencia entre una variable aleatoria discreta y continua?"},
        {"tema": "Probabilidad", "pregunta": "¿Qué es la probabilidad clásica?"},
        {"tema": "Probabilidad", "pregunta": "¿Qué es la probabilidad frecuentista?"},
        {"tema": "Distribuciones simples", "pregunta": "¿Qué es una distribución uniforme?"},
        {"tema": "Distribuciones simples", "pregunta": "¿Qué caracteriza a una distribución de probabilidad discreta?"},
        {"tema": "Eventos", "pregunta": "¿Qué es un evento en probabilidad?"},
        {"tema": "Eventos", "pregunta": "¿Qué significa que dos eventos sean mutuamente excluyentes?"}
    ],
    "intermedio": [
        {"tema": "Desviación estándar", "pregunta": "¿Qué es la desviación estándar?"},
        {"tema": "Desviación estándar", "pregunta": "¿Cómo se interpreta una desviación estándar alta o baja?"},
        {"tema": "Medidas de dispersión", "pregunta": "¿Qué es la varianza?"},
        {"tema": "Medidas de dispersión", "pregunta": "¿Cómo se relacionan la varianza y la desviación estándar?"},
        {"tema": "Estadística descriptiva", "pregunta": "¿Qué es la media aritmética?"},
        {"tema": "Estadística descriptiva", "pregunta": "¿Qué diferencia hay entre la media y la mediana?"},
        {"tema": "Distribuciones", "pregunta": "¿Qué es una distribución normal?"},
        {"tema": "Distribuciones", "pregunta": "¿Qué es una distribución sesgada y cómo se identifica?"}
    ],
    "avanzado": [
        {"tema": "Probabilidad conjunta", "pregunta": "¿Cómo se calcula la probabilidad conjunta de dos eventos independientes?"},
        {"tema": "Probabilidad conjunta", "pregunta": "¿Qué diferencia hay entre probabilidad conjunta y probabilidad condicional?"},
        {"tema": "Teorema de Bayes", "pregunta": "Explica el teorema de Bayes con un ejemplo."},
        {"tema": "Teorema de Bayes", "pregunta": "¿Cómo se aplica el teorema de Bayes en problemas médicos?"},
        {"tema": "Distribuciones avanzadas", "pregunta": "¿Qué es una distribución binomial?"},
        {"tema": "Distribuciones avanzadas", "pregunta": "¿Qué es una distribución de Poisson y en qué casos se utiliza?"},
        {"tema": "Inferencia", "pregunta": "¿Qué es una estimación puntual en inferencia estadística?"},
        {"tema": "Inferencia", "pregunta": "¿Qué diferencia hay entre una estimación puntual y un intervalo de confianza?"}
    ]
}
# --- Definición del estado ---
class State(TypedDict):
    user_input: str
    modo: str
    respuestas: List[str]
    pregunta_idx: int
    feedback: Dict[str, Any]
    nivel: str
    fortalezas: List[str]
    debilidades: List[str]
    puntaje_promedio: float
    detalle: List[Dict[str, Any]]
    preguntas_seleccionadas: List[Dict[str, Any]]
    temas: List[str]
    subtemas : Dict[str, Any]
    tema_actual : int
    contexto_recuperado: str
    explicacion: str
    messages: Annotated[list[AnyMessage], add_messages]
    
# --- vector store ---
persist_path = Path('./chroma')
vectorstore = Chroma(persist_directory=str(persist_path), embedding_function=OpenAIEmbeddings())


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
llm = ChatOpenAI(model="o4-mini")

# --- difinir tools ----

@tool
def informe_estadistico_tool(numbers: str) -> str:
    """Retorna un informe estadistico de una lista de numeros sparados por coma. El informe estadistico obtiene la siguiente información de la lista: count, mean, standard deviation, min, max, quartile 1, quartile 2 and quartile 3"""
    try:
        nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
        if not nums:
            return "No valid numbers provided."

        nums_array = np.array(nums)

        count = len(nums_array)
        mean = np.mean(nums_array)
        std = np.std(nums_array)
        min_val = np.min(nums_array)
        max_val = np.max(nums_array)
        q1 = np.percentile(nums_array, 25)
        q2 = np.percentile(nums_array, 50)
        q3 = np.percentile(nums_array, 75)

        result = (
            f"Informe estadistico de la lista:\n"
            f"- Count: {count}\n"
            f"- Mean: {mean:.4f}\n"
            f"- Standard Deviation: {std:.4f}\n"
            f"- Min: {min_val}\n"
            f"- Max: {max_val}\n"
            f"- Q1 (25th percentile): {q1}\n"
            f"- Q2 (median): {q2}\n"
            f"- Q3 (75th percentile): {q3}"
        )

        return result

    except Exception as e:
        return f"Error processing numbers: {str(e)}"
    
    
search_tool = DuckDuckGoSearchRun()


tools = [search_tool, informe_estadistico_tool]

# --- funciones ---
def seleccionar_preguntas(nivel: str):
    preguntas = quiz_preguntas[nivel]
    temas = list(set([p["tema"] for p in preguntas]))
    seleccionadas = []
    for tema in temas:
        preguntas_tema = [p for p in preguntas if p["tema"] == tema]
        seleccionadas.append(random.choice(preguntas_tema))
    return seleccionadas

# --- prompts ---
prompt_quiz = ChatPromptTemplate.from_messages([
    ("system",
     """Eres un experto en educación. Evalúa las siguientes respuestas del usuario a preguntas de probabilidad y estadística.
Para cada respuesta, califica de 0 a 5 (donde 0 es incorrecta y 5 es perfecta), explica brevemente la calificación.

Devuelve la respuesta SOLO en formato JSON con la siguiente estructura:
{{
  "resultados": [puntaje1, puntaje2, ...],
  "detalle": [
    {{
      "pregunta": "...",
      "respuesta": "...",
      "tema": "...",
      "puntaje": 0-5,
      "feedback": "..."
    }},
    ...
  ]
}}

Respuestas del usuario:
{respuestas_usuario}
""")
])

prompt_plan = ChatPromptTemplate.from_template(
    """
Eres un tutor experto en estadística y probabilidad.
El estudiante tiene el nivel: {nivel}.
Sus debilidades principales son: {debilidades}.

Crea un plan de estudio personalizado y devuélvelo SOLO en formato JSON con EXACTAMENTE esta estructura:
{{
    "plan_estudio": {{
        "tema1": {{
            "nombre": "Nombre del Tema 1",
            "subtemas": [
                "Subtema 1.1",
                "Subtema 1.2",
                "Subtema 1.3",
                "Subtema 1.4"
            ]
        }},
        "tema2": {{
            "nombre": "Nombre del Tema 2",
            "subtemas": [
                "Subtema 2.1",
                "Subtema 2.2",
                "Subtema 2.3",
                "Subtema 2.4"
            ]
        }},
        "tema3": {{
            "nombre": "Nombre del Tema 3",
            "subtemas": [
                "Subtema 3.1",
                "Subtema 3.2",
                "Subtema 3.3",
                "Subtema 3.4"
            ]
        }}
    }}
}}

Los temas deben enfocarse en las debilidades mencionadas.
Cada tema DEBE tener exactamente 4 subtemas.
IMPORTANTE: Devuelve SOLO el JSON, sin texto adicional.
"""
)

PROMPT_EXPLICACION = ChatPromptTemplate.from_messages([
    ("system", """Eres un tutor experto en estadística y probabilidad.
    Tu tarea es explicar conceptos de manera clara y estructurada usando formato markdown.

    Instrucciones importantes:
    - Usa el contexto proporcionado como referencia y validación
    - Complementa la explicación con tu conocimiento general del tema
    - Asegúrate que la información sea precisa y actualizada
    - Adapta el nivel de complejidad según el estudiante
    - Incluye definiciones, fórmulas y ejemplos prácticos"""),

    ("user", """Para el tema "{tema_actual}", revisa el siguiente contexto como referencia:
    {contexto}

    Genera una explicación completa que:
    1. Valide y use la información relevante del contexto
    2. Complemente con conocimiento adicional importante
    3. Cubra los siguientes subtemas:
    {subtemas_str}

    La explicación debe incluir:
    - Definiciones claras y completas
    - Fórmulas relevantes con explicación
    - Ejemplos prácticos del mundo real
    - Formato markdown bien estructurado
    - Analogías para facilitar comprensión
    """)
])

# --- definir llm ---
llm = ChatOpenAI(model="o4-mini")

# --- nodos ---
## --- nodo calificar quiz y feedback ---
def nodo_generar_feedback(state: State):
    parser = JsonOutputParser()
    chain_quiz = prompt_quiz | llm | parser
    
    respuestas = state.get("respuestas", [])
    preguntas_seleccionadas = state.get("preguntas_seleccionadas", [])
    respuestas_usuario = []
    for idx, pregunta in enumerate(preguntas_seleccionadas):
        if idx < len(respuestas):
            respuestas_usuario.append({
                "pregunta": pregunta["pregunta"],
                "respuesta": respuestas[idx],
                "tema": pregunta["tema"]
            })
    prompt_str = prompt_quiz.format(respuestas_usuario=str(respuestas_usuario))
    data = chain_quiz.invoke(prompt_str)

    resultados = data.get("resultados", [])
    detalle = data.get("detalle", [])
    promedio = sum(resultados) / len(resultados) if resultados else 0
    promedio = round(promedio, 2)
    fortalezas = [d["tema"] for d in detalle if d["puntaje"] >= 4]
    debilidades = [d["tema"] for d in detalle if d["puntaje"] < 4]
    state["feedback"] = data
    state["fortalezas"] = list(set(state.get("fortalezas", []) + fortalezas))
    state["debilidades"] = list(set(state.get("debilidades", []) + debilidades))
    state["puntaje_promedio"] = promedio
    state["detalle"] = detalle
    return state

## --- nodo generar plan de estudio --- 

def nodo_plan_estudio(state: State):
    # Corregido: usar [] en lugar de .
    debilidades = state["debilidades"]
    debilidades_str = ", ".join(debilidades)

    parser = JsonOutputParser()
    chain = prompt_plan | llm | parser

    # Corregido: usar [] para acceder a los elementos
    respuesta = chain.invoke({
        "nivel": state["nivel"],
        "debilidades": debilidades_str
    })

    plan = respuesta["plan_estudio"]
    state_actualizado = state.copy()

    # Actualizar el estado con la información del plan
    temas = []
    subtemas = {}

    for tema_key, tema_data in plan.items():
        nombre_tema = tema_data["nombre"]
        temas.append(nombre_tema)
        subtemas[nombre_tema] = tema_data["subtemas"]

    state_actualizado["temas"] = temas
    state_actualizado["subtemas"] = subtemas
    state_actualizado["tema_actual"] = 0

    return state_actualizado

def recuperar_contexto(state: State) -> State:
    """Recupera información relevante del tema actual usando RAG"""
    tema_idx = state["tema_actual"]
    tema_actual = state["temas"][tema_idx]
    subtemas = state["subtemas"][tema_actual]
    
    # Construir query para RAG
    subtemas_str = "\n".join([f"- {s}" for s in subtemas])
    query = f"""
    Información detallada sobre {tema_actual}:
    {subtemas_str}
    Incluir definiciones, fórmulas y ejemplos.
    """

    # Obtener documentos relevantes
    documentos = retriever.get_relevant_documents(query)
    contexto = "\n\n".join([doc.page_content for doc in documentos])

    return {**state, "contexto_recuperado": contexto}

def generar_explicacion(state: State) -> State:
    """Genera la explicación en formato markdown"""
    tema_idx = state["tema_actual"]
    tema_actual = state["temas"][tema_idx]
    subtemas = state["subtemas"][tema_actual]
    contexto = state["contexto_recuperado"]

    # Preparar subtemas para el prompt
    subtemas_str = "\n".join([f"- {s}" for s in subtemas])

    # Generar explicación usando el LLM
    mensages = PROMPT_EXPLICACION.format_messages(
        contexto=contexto,
        tema_actual=tema_actual,
        subtemas_str=subtemas_str
    )

    explicacion = llm.invoke(mensages).content

    return {**state, "explicacion": explicacion}


## --- nodo preguntas libres ---

def assistant(state: State):
    last_message = state["messages"][-1]
    chat_with_tools = llm.bind_tools(tools)
    query = last_message.content
    print(f"Query recibida: {query}")
    

    docs = retriever.invoke(query)
    print(f"Documentos recuperados: {len(docs)}")

    if docs:
        context = "\n\n".join(doc.page_content for doc in docs)
        print(f"Contexto recuperado (primeros 300 chars): {context[:300]}")
        system_message = SystemMessage(content=f"""
Eres un asistente inteligente.

Utiliza la siguiente información como fuente principal para elaborar tu respuesta.
Si no es suficiente, usa tu conocimiento general.

Información recuperada:
---
{context}
---
""")
        messages = [system_message] +  state["messages"]
    else:
        messages = state["messages"]

    respuesta = chat_with_tools.invoke(messages)
    print(f"Respuesta generada: {respuesta.content}")

    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

    

def build_graphs():
    graph_builder2 = StateGraph(State)
    graph_builder2.add_node("generar_feedback", nodo_generar_feedback)
    graph_builder2.set_entry_point("generar_feedback")
    graph_builder2.add_edge("generar_feedback", END)
    graph_feedback = graph_builder2.compile()
    
    graph_builder3 = StateGraph(State)
    graph_builder3.add_node("plan_estudio", nodo_plan_estudio)
    graph_builder3.set_entry_point("plan_estudio")
    graph_builder3.add_edge("plan_estudio", END)
    graph_plan = graph_builder3.compile()
    
    graph_builder4 = StateGraph(State)
    graph_builder4.add_node("recuperar_contexto", recuperar_contexto)
    graph_builder4.add_node("generar_explicacion", generar_explicacion)
    graph_builder4.set_entry_point("recuperar_contexto")
    graph_builder4.add_edge("recuperar_contexto", "generar_explicacion")
    graph_builder4.add_edge("generar_explicacion", END)
    graph_explicacion = graph_builder4.compile()
    
    graph_builder5 = StateGraph(State)
    graph_builder5.add_node("assistant", assistant)
    graph_builder5.add_node("tools", ToolNode(tools))

    graph_builder5.add_edge(START, "assistant")
    graph_builder5.add_conditional_edges("assistant", tools_condition)
    graph_builder5.add_edge("tools", "assistant")
    graph_libre = graph_builder5.compile()
    

    return graph_feedback , graph_plan , graph_explicacion, graph_libre