# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import io
import json
import base64
import tempfile
import requests
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.document_loaders import JSONLoader
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which
import magic
import sqlite3
import traceback

AudioSegment.converter = which("ffmpeg")

# Load environment variables
load_dotenv()
AZURE_AKS_API_KEY = os.getenv("AZURE_AKS_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- OCR ---
class getFromCnn:
    def __init__(self):
        api_key = os.environ.get('AZURE_AKS_API_KEY')
        if not api_key:
            raise ValueError("API key not set.")
        self.url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'

    def connect(self, image_bytes):
        if len(image_bytes) > 41943040:
            return None
        json_data = {
            "requests": [{
                "features": [{"maxResults": 50, "model": "builtin/latest", "type": "DOCUMENT_TEXT_DETECTION"}],
                "image": {"content": base64.b64encode(image_bytes).decode('utf-8')},
                "imageContext": {"cropHintsParams": {"aspectRatios": [0.8, 1, 1.2]}}
            }]
        }
        try:
            response = requests.post(self.url, json=json_data)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Exception: {str(e)}")
        return None

class PageImageProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_image_bytes(self, pagenum=0):
        pages = convert_from_path(self.file_path)
        if pagenum >= len(pages):
            raise ValueError("Page number exceeds document.")
        buffer = io.BytesIO()
        pages[pagenum].save(buffer, format="PNG")
        return buffer.getvalue()

def ocr_page_to_json(file_path, page=0):
    gcVision = getFromCnn()
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        with open(file_path, "rb") as img_file:
            image_bytes = img_file.read()
    elif file_path.lower().endswith(".pdf"):
        processor = PageImageProcessor(file_path)
        image_bytes = processor.get_image_bytes(page)
    else:
        raise ValueError("Unsupported file type.")

    resp = gcVision.connect(image_bytes)
    if not resp:
        raise Exception("OCR processing failed.")
    if "responses" not in resp or "fullTextAnnotation" not in resp["responses"][0]:
        raise Exception("No text found in document.")

    text = resp["responses"][0]["fullTextAnnotation"]["text"]
    tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp_json.name, "w") as f:
        json.dump({"fullTextAnnotation": {"text": text}}, f)
    return tmp_json.name

# --- Model loading ---

def load_model():
    print(f"api key is:/", {AZURE_AKS_API_KEY})
    raise Exception("testing")
    model_id = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device_map="auto", truncation=True)

llm_pipeline = load_model()

# --- Smart loader ---
def smart_json_loader(file_path):
    schemas = ['.fullTextAnnotation.text', '.form[].text', '.data.text', '.text', '.pages[].text', '.document.text', '.results[].text']
    for schema in schemas:
        try:
            loader = JSONLoader(file_path=file_path, jq_schema=schema, text_content=False)
            data = loader.load()
            if data and any(d.page_content.strip() for d in data):
                return data
        except:
            continue
    raise ValueError("No valid schema found for document content.")

# --- Process Document ---
def process_document(file_obj, question):
    try:
        file_path = file_obj.name
        print(f"\n📄 Received file: {file_path}")
        print(f"❓ Question: {question}")

        if file_path.endswith((".pdf", ".png", ".jpg", ".jpeg")):
            print("🔍 Running OCR via Google Vision...")
            file_path = ocr_page_to_json(file_path)
            print(f"✅ OCR result saved to: {file_path}")

        data = smart_json_loader(file_path)
        merged_text = " ".join([doc.page_content.strip() for doc in data])
        print(f"📚 Merged text length: {len(merged_text)} characters")

        if not merged_text:
            return "Document contains no extractable text content."

        prompt = f"""You are a helpful assistant. Extract the most relevant answer to the question from the document.
If not found, respond with \"Information not found in document\".

Document:
{merged_text}

Question: {question}
Answer:"""

        print("✍️ Prompt sent to model:", prompt[:300], '...')
        result = llm_pipeline(prompt)[0]["generated_text"]
        print("🤖 Model raw output:", result[:300], '...')

        answer_only = result.split("Answer:")[-1].strip()
        return answer_only if answer_only else "Answer could not be generated."

    except Exception as e:
        print("❌ Error in process_document:", str(e))
        return "An error occurred while processing the document."

# --- Upload API ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

# --- Ask API ---
@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    filepath = data.get('filepath')
    question = data.get('question')

    if not filepath or not question:
        return jsonify({'error': 'Missing file or question'}), 400

    doc_name = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        answer = process_document(f, question)

    # Save chat to DB
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT,
                question TEXT,
                answer TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("INSERT INTO chats (document_name, question, answer) VALUES (?, ?, ?)",
                (doc_name, question, answer))
        conn.commit()
        conn.close()
    except Exception as e:
        print("❌ Error saving chat:", e)

    return jsonify({'answer': answer})

@app.route('/history/<doc_name>', methods=['GET'])
def get_chat_history(doc_name):
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("SELECT question, answer, timestamp FROM chats WHERE document_name = ? ORDER BY timestamp DESC", (doc_name,))
        rows = c.fetchall()
        conn.close()

        history = [{"question": row[0], "answer": row[1], "timestamp": row[2]} for row in rows]
        return jsonify(history)

    except Exception as e:
        return jsonify({'error': f"Failed to retrieve history: {str(e)}"}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp:
            audio_file.save(temp.name)
            input_path = temp.name

        mime_type = magic.from_file(input_path, mime=True)
        print(f"📄 Detected MIME type: {mime_type}")

        if "webm" in mime_type:
            fmt = "webm"
        elif "wav" in mime_type or "x-wav" in mime_type:
            fmt = "wav"
        elif "ogg" in mime_type or "opus" in mime_type:
            fmt = "ogg"
        else:
            return jsonify({'error': f"Unsupported audio type: {mime_type}"}), 400

        try:
            audio = AudioSegment.from_file(input_path, format=fmt)
        except Exception as decode_error:
            print("❌ Failed to decode audio:", decode_error)
            return jsonify({'error': 'Failed to decode audio file'}), 500

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio.export(temp_wav.name, format="wav")
            wav_path = temp_wav.name
            print("🎧 Converted to WAV:", wav_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
            print("📝 Transcript:", transcript)
            return jsonify({'transcript': transcript})

    except Exception as e:
        print("❌ Transcription error:", str(e))
        traceback.print_exc()
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

if __name__ == '__main__':
    from pyngrok import ngrok
    port = 5000
    public_url = ngrok.connect(port)
    print(f"🚀 Public URL: {public_url}")
    app.run(port=port)
