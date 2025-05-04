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

df = pd.read_csv(BASE_DIR/'evaluacion/datasets/feedback_dataset.csv')

# LLM para evaluación
llm_evaluacion = ChatOpenAI(model="gpt-4.1", temperature=1)

feedback_prompt = PromptTemplate.from_template(
    """Dada la respuesta del estudiante: {query}
    La retroalimentación generada por el modelo: {result}
    La retroalimentación real: {answer}

    ¿La retroalimentación generada por el modelo es adecuada (No penalices el hecho de que de mas informacion que la sola calificacion)? Responde sólo 'Sí' o 'No', y explica en una línea por qué.
    """
)

eval_chain = QAEvalChain.from_llm(
    llm=llm_evaluacion,
    criteria=["correctness"],
    prompt=feedback_prompt
)

# Preparar ejemplos
examples = df.to_dict(orient="records")

# Evaluar
results = []
feedback_generado = []
veredicto = []
scores = []
tema = None

for example in examples:
    # Generar respuesta de estudiante 
    pregunta = example["pregunta"]
    pregunta_dict = {
        "pregunta": pregunta,
        "tema": tema
    }
    pregunta_dict = [pregunta_dict]
    
    respuesta = graph_feedback.invoke({
        "respuestas": [example["query"]],
        "preguntas_seleccionadas": pregunta_dict
        })
    
    feedback = respuesta['feedback']['detalle'][0]['feedback']

    feedback_generado.append(feedback)
    
    graded = eval_chain.evaluate_strings(
        input=example["query"],
        prediction=feedback,
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
    
df["feedback_generado"] = feedback_generado
df["evaluacion_feedback"] = [r["reasoning"] for r in results]
df["veredicto"] = veredicto
df["puntaje"] = scores

# Guardar
df.to_csv(BASE_DIR/'evaluacion/results/resultados_test_feedback.csv', index=False)
