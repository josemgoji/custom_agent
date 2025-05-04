import pandas as pd
from dotenv import load_dotenv
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from graphs import build_graphs

graph_feedback, graph_plan, graph_explicacion, graph_libre = build_graphs()

# Cargar .env
load_dotenv()

# Cargar dataset


BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR/'evaluacion/datasets/eval_dataset.csv')

# LLM para evaluación
llm_evaluacion = ChatOpenAI(model="gpt-4.1", temperature=1)

PROMPT = PromptTemplate.from_template(
        "Dada la pregunta: {query}\n"
        "La respuesta del estudiante: {result}\n"
        "La respuesta correcta esperada: {answer}\n\n"
        "¿La respuesta es correcta? Responde sólo 'Sí' o 'No', y explica en una línea por qué."
    )

eval_chain = QAEvalChain.from_llm(
    llm=llm_evaluacion,
    criteria=["correctness"],
    prompt=PROMPT
)

# Preparar ejemplos
examples = df.to_dict(orient="records")

# Evaluar
results = []
respuestas_generadas = []
veredicto = []
scores = []

for example in examples:
    # Generar respuesta de estudiante
    respuesta = graph_libre.invoke({"messages": example["query"]})
    respuesta = respuesta["messages"][-1].content

    respuestas_generadas.append(respuesta)
    
    graded = eval_chain.evaluate_strings(
        input=example["query"],
        prediction=respuesta,
        reference=example["answer"]
    )
    
    resultado = graded['reasoning']
    if resultado.strip().lower().startswith('sí'):
        graded['value'] = 'Correcto'
        graded['score'] = 1
    elif resultado.strip().lower().startswith('no'):
        graded['value'] = 'Incorrecto'
        graded['score'] = 0

    lc_verdict = graded.get("value", "UNKNOWN")
    is_correct = graded.get("score", 0)
    
    results.append(graded)
    veredicto.append(lc_verdict)
    scores.append(is_correct)   
    
df["respuesta_agente"] = respuestas_generadas
df["evaluacion"] = [r["reasoning"] for r in results]
df["veredicto"] = veredicto
df["puntaje"] = scores

# Guardar
df.to_csv(BASE_DIR/'evaluacion/results/resultados_test.csv', index=False)
