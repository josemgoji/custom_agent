from pathlib import Path
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode

from tools import informe_estadistico_tool
from utils import crear_prompt

load_dotenv()

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
search_tool = DuckDuckGoSearchRun()

tools = [search_tool, informe_estadistico_tool]

# --- prompts ---
prompts_path = Path('./prompts')

PROMPT_QUIZ = crear_prompt(prompts_path / 'prompt_quiz.txt')
PROMPT_PLAN = crear_prompt(prompts_path / 'prompt_plan.txt')
PROMPT_EXPLICACION = crear_prompt(prompts_path / 'prompt_exp.txt')

# --- definir llm ---
llm = ChatOpenAI(model="o4-mini")

# --- nodos ---
## --- nodo calificar quiz y feedback ---
def nodo_generar_feedback(state: State):
    parser = JsonOutputParser()
    chain_quiz = PROMPT_QUIZ | llm | parser
    
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
    prompt_str = PROMPT_QUIZ.format(respuestas_usuario=str(respuestas_usuario))
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
    chain = PROMPT_PLAN | llm | parser

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