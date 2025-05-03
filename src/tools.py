import numpy as np
from langchain.tools import tool

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