from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr
import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

#Prompt template
prompt_template_str = """
You are an expert nutritional analyst. Analyze the following food item or meal input:

[INPUT DATA]
{input_text}

[ANALYSIS INSTRUCTIONS]
1. Component Breakdown: Deconstruct the input into its individual food items (e.g., separate a cooked meal into its main component ingredients).
2. Portion Determination:
   - If explicit nutritional data or weight is provided in the input, prioritize those exact figures.
   - If a cooked dish is listed (e.g., "lentils with rice"), assume a standard single-person serving size.
   - If a single whole item is listed (e.g., "one apple"), calculate for one whole medium unit.
   - For all other cases, rely on standard nutritional databases (such as USDA) to estimate values.
3. Extracted Attributes:
   - Calories (kcal)
   - Proteins (g)
   - Carbohydrates (g)
   - Fats (g)
   - suggested_portion 

Formatting guidance:
- "Alimento": Component name, including estimated portion weight in parentheses (e.g., "Lentils (150g)").
- "Calories", "Proteins", "Carbohydrates", "Fats": Value followed by unit ("kcal" or "g").
"""

class MealCategory(str, Enum):
    BREAKFAST = "Breakfast"
    LUNCH = "Lunch" 
    DINNER = "Dinner"
    SNACK = "Snack"

class MealRequest(BaseModel):
    category: MealCategory
    input_text: str # picture of the meal or food item


# Define the Pydantic model for output
class MealAnalysisOutput(BaseModel):
    alimento: str = Field(description="Name of the food item or meal component")
    calories: float = Field(description="Calories in kcal")
    proteins: float = Field(description="Proteins in grams")
    carbohydrates: float = Field(description="Carbohydrates in grams")
    fats: float = Field(description="Fats in grams")
    suggested_portion: Optional[str] = Field(description="Suggested portion size if applicable")

def get_response(prompt_template_str, input_text):
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    prompt = prompt_template.format(input_text=input_text)
    # Model interface
    model = init_chat_model(model="google_genai:gemini-3.6-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    # Call the model with the prompt
    structured_response = model.with_structured_output(MealAnalysisOutput)
    response = structured_response.invoke(prompt)
    return response

#demo = gr.Interface(
#    fn=get_response,
#    inputs=gr.Textbox(label="Enter an aliment", placeholder="e.g., Kiwi"),
#    outputs=gr.Textbox(label="Energetic Value"),
#    title="Energetic Value Calculator",
#    description="Enter an aliment, and the model will calculate its estimated energetic value",
#)

if __name__ == "__main__":

    alimento = "1 cuadrado de chocolate amargo (10g)"
    MealAnalysis = get_response(prompt_template_str, alimento)
    print(MealAnalysis)
    #demo.launch(theme=gr.themes.Ocean())