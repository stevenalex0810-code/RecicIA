import base64
import json
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
CORS(app)

# Limitar el tamaño de las imágenes recibidas a 8 MB.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=api_key) if api_key else None

MIME_TYPES = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@app.get("/")
def inicio():
    return jsonify({
        "mensaje": "API de EcoAI funcionando",
        "estado": "ok"
    })


@app.errorhandler(413)
def imagen_demasiado_grande(error):
    return jsonify({
        "error": "La imagen supera el límite de 8 MB."
    }), 413


@app.post("/api/clasificar")
def clasificar_residuo():
    if client is None:
        return jsonify({
            "error": "Falta configurar OPENAI_API_KEY en .env"
        }), 500

    if "imagen" not in request.files:
        return jsonify({
            "error": "Debes enviar una imagen en el campo 'imagen'."
        }), 400

    archivo = request.files["imagen"]

    if not archivo.filename:
        return jsonify({
            "error": "No se seleccionó ninguna imagen."
        }), 400


    # Leer los bytes de la imagen.
    contenido = archivo.read()

    if not contenido:
        return jsonify({
            "error": "El archivo está vacío."
        }), 400

    # Detectar el formato real de la imagen.
    try:
        imagen = Image.open(BytesIO(contenido))
        formato = imagen.format
        imagen.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "El archivo no es una imagen válida o está dañado."
        }), 400

    # Convertir el formato detectado al MIME correspondiente.
    formatos_permitidos = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }

    mime_type = formatos_permitidos.get(formato)

    if mime_type is None:
        return jsonify({
            "error": (
                f"Formato detectado no compatible: {formato}. "
                "Usa JPG, PNG o WEBP."
            )
        }), 400

    # Preparar la imagen para enviarla a la IA.
    imagen_b64 = base64.b64encode(contenido).decode("utf-8")
    data_url = f"data:{mime_type};base64,{imagen_b64}"


    instrucciones = """
    Eres EcoAI, un asistente de clasificación de residuos en Colombia.

    Analiza la fotografía e identifica el residuo principal.

    Categorías permitidas:
    - aprovechable
    - no_aprovechable
    - organico
    - especial
    - desconocido

    Orientación general:
    - aprovechable: recipiente blanco, si el material es
      aceptado para aprovechamiento y está adecuadamente separado.
    - no_aprovechable: recipiente negro.
    - organico: recipiente verde, si es un residuo orgánico
      aprovechable.
    - especial: requiere un sistema de recolección específico.
    - desconocido: no hay información suficiente para clasificarlo.

    No asumas que un material está limpio solo por su apariencia.
    Si la imagen no permite identificar el residuo, usa desconocido.
    No inventes puntos de recolección ni direcciones específicas.
    No recomiendes manipular residuos peligrosos.

    Responde exclusivamente con un objeto JSON válido con estas claves:
    {
      "objeto": "nombre del residuo",
      "material": "material probable",
      "categoria": "una de las categorías permitidas",
      "recipiente": "recipiente o sistema de recolección",
      "instrucciones": "recomendación breve",
      "confianza": "alta, media o baja"
    }
    """

    try:

        respuesta = client.models.generate_content(
            model=model_name,
            contents=[
                instrucciones,
                types.Part.from_bytes(
                    data=contenido,
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        texto = respuesta.text

        if not texto:
            raise ValueError("Gemini devolvió una respuesta vacía.")

        resultado = json.loads(texto)

        categorias_validas = {
            "aprovechable",
            "no_aprovechable",
            "organico",
            "especial",
            "desconocido"
        }

        if resultado.get("categoria") not in categorias_validas:
            raise ValueError("Categoría no válida")

        claves = {
            "objeto",
            "material",
            "categoria",
            "recipiente",
            "instrucciones",
            "confianza"
        }

        if not claves.issubset(resultado.keys()):
            raise ValueError("La respuesta está incompleta")

        return jsonify(resultado)

    except json.JSONDecodeError:
        app.logger.exception("La IA no devolvió JSON válido")
        return jsonify({
            "error": "No se pudo interpretar la respuesta de la IA."
        }), 502

    except Exception:
        app.logger.exception("Error durante la clasificación")
        return jsonify({
            "error": "No se pudo clasificar el residuo. Inténtalo de nuevo."
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
