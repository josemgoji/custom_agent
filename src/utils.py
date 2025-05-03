import random
import json
from langchain.prompts import ChatPromptTemplate
from pathlib import Path

preguntas_path = Path('./data/preguntas.json')

with open(preguntas_path, 'r', encoding='utf-8') as f:
    quiz_preguntas = json.load(f)

#--- Selección de preguntas para el quiz ---
def seleccionar_preguntas(nivel: str):
    preguntas = quiz_preguntas[nivel]
    temas = list(set([p["tema"] for p in preguntas]))
    seleccionadas = []
    for tema in temas:
        preguntas_tema = [p for p in preguntas if p["tema"] == tema]
        seleccionadas.append(random.choice(preguntas_tema))
    return seleccionadas

#--- cargar promts ---
def crear_prompt(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        texto_prompt = f.read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", texto_prompt)
    ])
    return prompt