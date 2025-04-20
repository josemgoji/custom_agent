# grafo_quiz.py

from langgraph.graph import StateGraph, END
from streamlit import context
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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
    
# --- vector store ---
# documentos de ejemplo para RAG
ejemplo_documentos_rag = [
    Document(
        page_content="""
        La probabilidad condicional es la probabilidad de que ocurra un evento A,
        sabiendo que también ha ocurrido otro evento B. Se denota como P(A|B) y
        se calcula como: P(A|B) = P(A∩B) / P(B)

        La regla de la cadena establece que: P(A∩B) = P(A|B) × P(B)
        """
    ),
    Document(
        page_content="""
        El Teorema de Bayes es una herramienta fundamental que relaciona las
        probabilidades condicionales de dos eventos. Su fórmula es:
        P(A|B) = P(B|A) × P(A) / P(B)

        Este teorema es especialmente útil cuando tenemos probabilidades previas
        y queremos actualizar nuestro conocimiento con nueva información.
        """
    ),
    Document(
        page_content="""
        Los árboles de probabilidad son herramientas visuales que ayudan a
        resolver problemas de probabilidad condicional. Cada rama representa
        un evento y sus probabilidades asociadas.
        """
    )
]
# Configuración de componentes
embeddings = OpenAIEmbeddings()

# Crear y configurar Chroma
vectorstore = Chroma.from_documents(
    documents=ejemplo_documentos_rag,
    embedding=embeddings,
    collection_name="estadistica_probabilidad",
    persist_directory="./chroma"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
llm = ChatOpenAI(model="o4-mini")

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
prompt_clasificar = ChatPromptTemplate.from_messages([
    ("system",
    "Clasifica la siguiente intención del usuario SOLO como 'guiado' o 'libre'. "
    "Si el usuario quiere que lo guíes paso a paso, responde 'guiado'. "
    "Si solo quiere una respuesta directa, responde 'libre'. "
    "Intención del usuario: {user_input}\n"
    "Respuesta:")
])

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

## --- nodo clasificar ---
def nodo_clasificacion_modo(state: State):
    user_input = state["user_input"]
    if not user_input:
            return {"modo": ""}
        
    llm_chain = prompt_clasificar | llm
    respuesta = llm_chain.invoke({"user_input": user_input}).content.strip().lower()
    
    if "guiado" in respuesta:
        modo = "guiado"
    elif "libre" in respuesta:
        modo = "libre"
    else:
        modo = "libre"
    return {"modo": modo}

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
    print(tema_idx)
    tema_actual = state["temas"][tema_idx]
    print(tema_actual)
    subtemas = state["subtemas"][tema_actual]
    #subtemas = subtemas_todos.get(tema_actual,[])
    #subtemas = ["hola"]
    print(subtemas)
    
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
    messages = PROMPT_EXPLICACION.format_messages(
        contexto=contexto,
        tema_actual=tema_actual,
        subtemas_str=subtemas_str
    )

    explicacion = llm.invoke(messages).content

    return {**state, "explicacion": explicacion}
    

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

    return graph_modo, graph_feedback , graph_plan , graph_explicacion