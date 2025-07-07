from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from openai import OpenAI

# Carga el dataset
df = pd.read_csv("ventas_inventario_diario.csv", parse_dates=["fecha"])

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializa la app FastAPI
app = FastAPI()

# Modelo de entrada
class Pregunta(BaseModel):
    pregunta: str

# Endpoint principal
@app.post("/accion")
async def responder(p: Pregunta):
    resumen = df.groupby(["id_tienda", "producto_nombre"]).agg({
        "ventas_unidades": "sum",
        "stock_actual": "mean",
        "quiebre_stock": "sum",
        "pedido_sugerido": "mean"
    }).reset_index().head(20).to_csv(index=False)

    prompt = f"""
Eres NetoGPT, un copiloto inteligente para el gerente regional de una cadena de tiendas hard discount como Neto.

Tu rol es dar respuestas operativas, claras y accionables, basadas en datos reales. A continuación te paso un resumen de datos operativos de las tiendas:

{resumen}

La pregunta del gerente es:
{p.pregunta}

Responde de manera profesional y concreta. Si hay riesgos, alertas o buenas prácticas que aplicar, menciónalo.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return {"respuesta": response.choices[0].message.content.strip()}

