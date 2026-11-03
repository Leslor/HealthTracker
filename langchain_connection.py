from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr
import os

#API Key Management
from dotenv import load_dotenv
load_dotenv()

#Prompt template
prompt_template_str = """
Calcula el valor energético estimado en kcal para una porción individual estándar de: {input_text}.

Consideraciones:
1. Si se trata de un plato o comida cocinada (ej. lentejas), asume una porción personal típica. Si es una unidad (ej. una manzana), calcula la pieza entera.
2. Estructura el desglose de forma limpia y directa, con la precisión de una métrica bien calibrada.
3. Sé conciso, ve al grano y presenta el resultado de forma intuitiva.
"""

prompt_template = PromptTemplate.from_template(prompt_template_str)
input_text = "Kiwi"

def generete_explanation(input_text):
    prompt = prompt_template.format(input_text=input_text)
    # Model interface
    model = init_chat_model(model="google_genai:gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    # Call the model with the prompt
    response = model.invoke(prompt)
    return response.text

demo = gr.Interface(
    fn=generete_explanation,
    inputs=gr.Textbox(label="Enter an aliment", placeholder="e.g., Kiwi"),
    outputs=gr.Textbox(label="Energetic Value"),
    title="Energetic Value Calculator",
    description="Enter an aliment, and the model will calculate its estimated energetic value",
)

if __name__ == "__main__":
    explanation = generete_explanation(input_text)
    demo.launch(theme=gr.themes.Ocean())