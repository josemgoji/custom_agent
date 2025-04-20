# Aca iran als tools a las que tengan acceso los agentes
# ejemplo funcion cargar promts darkanita
'''
def load_prompt(version="v1_asistente_rrhh"):
    prompt_path = os.path.join(PROMPT_DIR, f"{version}.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    with open(prompt_path, "r") as f:
        prompt_text = f.read()
    return PromptTemplate(input_variables=["context", "question"], template=prompt_text)
'''