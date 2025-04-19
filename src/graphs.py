# grafo_quiz.py

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.chains import LLMChain
import random

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

def seleccionar_preguntas(nivel: str):
    preguntas = quiz_preguntas[nivel]
    temas = list(set([p["tema"] for p in preguntas]))
    seleccionadas = []
    for tema in temas:
        preguntas_tema = [p for p in preguntas if p["tema"] == tema]
        seleccionadas.append(random.choice(preguntas_tema))
    return seleccionadas

def nodo_clasificacion_modo(state: State):
    user_input = state["user_input"]
    if not user_input:
        return {"modo": ""}
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = (
        "Clasifica la siguiente intención del usuario SOLO como 'guiado' o 'libre'. "
        "Si el usuario quiere que lo guíes paso a paso, responde 'guiado'. "
        "Si solo quiere una respuesta directa, responde 'libre'. "
        f"Intención del usuario: {user_input}\n"
        "Respuesta:"
    )
    respuesta = llm.invoke(prompt).content.strip().lower()
    if "guiado" in respuesta:
        modo = "guiado"
    elif "libre" in respuesta:
        modo = "libre"
    else:
        modo = "libre"
    return {"modo": modo}

llm = ChatOpenAI(model="gpt-3.5-turbo")
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
parser = JsonOutputParser()
chain_quiz = LLMChain(
    llm=llm,
    prompt=prompt_quiz,
    output_parser=parser
)

def nodo_generar_feedback(state: State):
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
    raw_result = chain_quiz.llm.invoke(prompt_str).content
    try:
        data = parser.parse(raw_result)
    except Exception:
        data = {}
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

def build_graphs():
    graph_builder = StateGraph(State)
    graph_builder.add_node("clasificacion_modo", nodo_clasificacion_modo)
    graph_builder.set_entry_point("clasificacion_modo")
    graph_builder.add_edge("clasificacion_modo", END)
    graph_modo = graph_builder.compile()

    graph_builder2 = StateGraph(State)
    graph_builder2.add_node("generar_feedback", nodo_generar_feedback)
    graph_builder2.set_entry_point("generar_feedback")
    graph_builder2.add_edge("generar_feedback", END)
    graph_feedback = graph_builder2.compile()

    return graph_modo, graph_feedback