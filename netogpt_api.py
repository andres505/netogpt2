from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import openai
import os

# Cargar datos
df = pd.read_csv("ventas_inventario_diario.csv", parse_dates=["fecha"])

# Verificar que la API Key esté disponible
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está definida.")
openai.api_key = api_key

# Inicializar FastAPI
app = FastAPI()

# Modelo de entrada
class Pregunta(BaseModel):
    pregunta: str

# Endpoint principal
@app.post("/accion")
async def responder(p: Pregunta):
    try:
        # Generar resumen de datos
        resumen = df.groupby(["id_tienda", "producto_nombre"]).agg({
            "ventas_unidades": "sum",
            "stock_actual": "mean",
            "quiebre_stock": "sum",
            "pedido_sugerido": "mean"
        }).reset_index().head(20).to_csv(index=False)

        # Crear prompt
        prompt = f"""
Eres NetoGPT, un copiloto inteligente para el gerente regional de una cadena de tiendas hard discount como Neto.

Tu rol es dar respuestas operativas, claras y accionables, basadas en datos reales. A continuación te paso un resumen de datos operativos de las tiendas:

{resumen}

La pregunta del gerente es:
{p.pregunta}

Responde de manera profesional y concreta. Si hay riesgos, alertas o buenas prácticas que aplicar, menciónalo.
"""

        # Llamar a la API de OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        # Extraer y devolver respuesta
        respuesta = response["choices"][0]["message"]["content"].strip()
        return {"respuesta": respuesta}

    except Exception as e:
        # Manejo de errores con respuesta HTTP 500
        return JSONResponse(status_code=500, content={"error": str(e)})
