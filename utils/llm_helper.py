# utils/llm_helper.py
import openai

def get_improvement_suggestions(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You're a resume optimization assistant. Suggest concise improvements."
        }, {
            "role": "user",
            "content": f"Suggest better phrasing for: {text}"
        }]
    )
    return response.choices[0].message.content