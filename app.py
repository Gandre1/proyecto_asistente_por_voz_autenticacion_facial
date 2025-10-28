import os
import time
import unicodedata
import webbrowser
import pickle
import uuid
import re
import subprocess
import platform
from datetime import datetime, timedelta

import gradio as gr
import numpy as np
from gtts import gTTS
from PIL import Image
import speech_recognition as sr
import cv2
import locale

try:
    locale.setlocale(locale.LC_TIME, 'es_ES.utf8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')
    except:
        pass

# --- RUTAS/DATOS ---
DATA_DIR = "lbph_data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_FILE = os.path.join(DATA_DIR, "recognizer.yml")
LABELS_FILE = os.path.join(DATA_DIR, "labels.pkl")
AUDIO_DIR = "tmp_audio"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Haarcascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False
labels = {}
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)
if os.path.exists(MODEL_FILE):
    try:
        recognizer.read(MODEL_FILE)
        model_trained = True
    except Exception:
        model_trained = False

# --- UTILIDADES TTS / audio ---
def text_to_speech(text, out_dir=AUDIO_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"reply_{uuid.uuid4().hex}.mp3"
    path = os.path.join(out_dir, fname)
    try:
        tts = gTTS(text=text, lang="es")
        tts.save(path)
        try:
            with open(path, "rb") as f:
                f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
        return path
    except Exception as e:
        print("gTTS fallo:", e)
        return None

def cleanup_old_audio(dir=AUDIO_DIR, max_age_seconds=300):
    now = time.time()
    try:
        for fn in os.listdir(dir):
            path = os.path.join(dir, fn)
            try:
                if now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
            except Exception:
                pass
    except Exception:
        pass

# --- TRANSCRIPCIÓN ---
def transcribe_audio(wav_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language="es-ES")
    except Exception:
        text = ""
    return text

# --- FACE UTIL ---
def detect_face_gray(np_image):
    if isinstance(np_image, np.ndarray):
        img = np.array(Image.fromarray(np_image).convert("RGB"))
    else:
        img = np.array(Image.open(np_image).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    return face_img, (x, y, w, h)

def save_enrollment_images(name, face_img):
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).strip()
    base = os.path.join(IMAGES_DIR, safe_name)
    i = 0
    while os.path.exists(f"{base}_{i}.png"):
        i += 1
    path = f"{base}_{i}.png"
    cv2.imwrite(path, face_img)
    return os.path.basename(path)

def train_recognizer():
    global labels, recognizer, model_trained
    image_paths = []
    label_ids = {}
    current_id = 0
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(IMAGES_DIR, filename)
        image_paths.append(path)
        name = "_".join(filename.split("_")[:-1])
        if name == "":
            name = "user"
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1
    if not image_paths:
        return False, "No hay imágenes para entrenar. Registra al menos una foto por usuario."
    y_labels = []
    x_train = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        roi = img[y:y+h, x:x+w]
        name = "_".join(os.path.basename(path).split("_")[:-1])
        label = label_ids.get(name, 0)
        x_train.append(roi)
        y_labels.append(label)
    if not x_train:
        return False, "No pude extraer rostros de las imágenes guardadas."
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, np.array(y_labels))
    recognizer.write(MODEL_FILE)
    labels = {v: k for k, v in label_ids.items()}
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(labels, f)
    model_trained = True
    return True, f"Entrenamiento completado con {len(labels)} usuarios."

# --- MAPS ---
SITE_MAP = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "wikipedia": "https://www.wikipedia.org",
}

APP_MAP = {
    "bloc de notas": {"win": ["notepad"]},
    "calculadora": {"win": ["calc"]},
    "paint": {"win": ["mspaint"]},
    "spotify": {"win": ["spotify"]},
}

#HANDLERS NO SENSITIVE
def handler_saludo(params): return "¡Hola! ¿En qué puedo ayudarte?"
def handler_hora(params): return time.strftime("Son las %H:%M:%S")
def handler_day(params):
    kind = params.get("kind", "hoy")
    today = datetime.now().date()
    if kind == "hoy": return f"Hoy es {today.strftime('%A %d %B %Y')}"
    if kind == "manana": return f"Mañana será {(today + timedelta(days=1)).strftime('%A %d %B %Y')}"
    if kind == "ayer": return f"Ayer fue {(today - timedelta(days=1)).strftime('%A %d %B %Y')}"
    return f"Fecha: {today.strftime('%A %d %B %Y')}"

def handler_open(params):
    site = params.get("site_key")
    app = params.get("app_key")
    url = params.get("url")
    if app:
        entry = APP_MAP.get(app.lower())
        if entry and platform.system().lower().startswith("win"):
            cmd = entry["win"]
            try:
                subprocess.Popen(cmd)
                return f"Abrí la aplicación {app}."
            except Exception as e:
                return f"No pude abrir la aplicación {app}: {e}"
        return "Abrir aplicaciones automáticas está implementado solo para Windows."
    if site:
        # si site es google_search:term ya fue convertido a url en parser
        url = SITE_MAP.get(site.lower(), f"https://www.google.com/search?q={site.replace(' ','+')}")
    if url:
        try:
            webbrowser.open(url)
            return f"Abriendo {url}"
        except Exception as e:
            return f"No pude abrir {url}: {e}"
    return "No pude identificar qué abrir."

#COMANDOS y PARSER
COMMANDS = [
    {"name":"saludo","patterns":[r"\b(hola|buenas|saludo|saluda)\b"], "handler": handler_saludo, "sensitive": False},
    {"name":"hora","patterns":[r"\b(qué hora es|que hora es|dime la hora|hora)\b"], "handler": handler_hora, "sensitive": False},
    {"name":"dia_hoy","patterns":[r"\b(qué día es hoy|que dia es hoy|qué dia es hoy)\b"], "handler": lambda p: handler_day({"kind":"hoy"}), "sensitive": False},
    {"name":"dia_manana","patterns":[r"\b(qué día será mañana|que dia sera manana|que dia es mañana|qué dia será mañana)\b"], "handler": lambda p: handler_day({"kind":"manana"}), "sensitive": False},
    {"name":"dia_ayer","patterns":[r"\b(qué día fue ayer|que dia fue ayer|ayer que dia)\b"], "handler": lambda p: handler_day({"kind":"ayer"}), "sensitive": False},
    # abrir sitios/apps
    {"name":"open_simple","patterns":[r"\babrir (google|youtube|gmail|wikipedia)\b", r"\babrir (bloc de notas|calculadora|paint|notepad|calc|mspaint|spotify|)\b", r"\babrir sitio (.+)\b"], "handler": handler_open, "sensitive": True},
    # búsquedas
    {"name":"buscar_google","patterns":[r"\bbuscar (en )?google (.+)", r"\bbuscar en google (.+)", r"\bBusca en google (.+)"], "handler": handler_open, "sensitive": True},
    {"name":"buscar_youtube","patterns":[r"\bBuscar (en )?youtube (.+)", r"\bBuscar en youtube (.+)", r"\bBusca en youtube (.+)"], "handler": handler_open, "sensitive": True},
]

import urllib.parse

def parse_command(text):
    t = (text or "").lower()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t)
        if unicodedata.category(c) != "Mn"
    )
    for cmd in COMMANDS:
        for pat in cmd["patterns"]:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                params = {}
                name = cmd["name"]
                if name in ("buscar_google", "buscar_youtube"):
                    # buscar query: try group 2 or 1
                    q = None
                    if m.lastindex and m.lastindex >= 2:
                        q = m.group(2)
                    elif m.lastindex and m.lastindex >= 1:
                        q = m.group(1)
                    if q:
                        q = q.strip()
                        if name == "buscar_google":
                            params["url"] = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
                        else: # youtube
                            params["url"] = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)
                elif name == "open_simple":
                    g = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                    if g:
                        g = g.strip()
                        if g in SITE_MAP:
                            params["site_key"] = g
                        elif g in APP_MAP:
                            params["app_key"] = g
                        else:
                            params["site_key"] = g
                else:
                    # generic fallback: capture URL if present
                    url_m = re.search(r"https?://\S+", t)
                    if url_m:
                        params["url"] = url_m.group(0)
                return {"name": name, "sensitive": cmd["sensitive"], "handler": cmd["handler"], "params": params}
    return {"name": None, "sensitive": False, "handler": None, "params": {}}

# --- UI Handlers ---
def enroll_user(name, webcam_image):
    if webcam_image is None or name.strip() == "":
        return "Debes dar un nombre y tomar/subir una foto.", None
    face_img, bbox = detect_face_gray(webcam_image)
    if face_img is None:
        return "No detecté cara. Acércate y con buena iluminación.", None
    saved = save_enrollment_images(name, face_img)
    return f"Imagen guardada: {saved}. Pulsa 'Entrenar modelo' para usarla.", saved

def do_train():
    ok, msg = train_recognizer()
    return msg

def recognize_flow(audio, webcam_image):
    if audio is None:
        return "No detecté audio.", "", None

    text = transcribe_audio(audio)
    if not text:
        reply_text = "No entendí lo que dijiste. Intenta de nuevo, más claro."
        audio_file = text_to_speech(reply_text)
        cleanup_old_audio()
        return reply_text, "", audio_file

    parsed = parse_command(text)
    if parsed["name"] is None:
        reply_text = ("He entendido: «%s». Pero necesita la verificación facial o no tengo una acción programada para eso. "
                      "Prueba: 'abrir google', 'buscar recetas de arequipe', 'buscar en youtube musica', ") % text
        audio_file = text_to_speech(reply_text)
        cleanup_old_audio()
        return reply_text, text, audio_file

    user = None
    if parsed["sensitive"]:
        if webcam_image is None:
            reply_text = "Este comando requiere autenticación facial. Por favor sube o toma una foto."
            audio_file = text_to_speech(reply_text)
            cleanup_old_audio()
            return reply_text, text, audio_file
        if not model_trained:
            reply_text = "Modelo no entrenado. Registra y entrena primero."
            audio_file = text_to_speech(reply_text)
            cleanup_old_audio()
            return reply_text, text, audio_file
        face_img, _ = detect_face_gray(webcam_image)
        if face_img is None:
            reply_text = "No detecté cara en la foto. Intenta con mejor iluminación."
            audio_file = text_to_speech(reply_text)
            cleanup_old_audio()
            return reply_text, text, audio_file
        try:
            label_id, conf = recognizer.predict(face_img)
        except Exception as e:
            reply_text = f"Error durante reconocimiento facial: {e}"
            audio_file = text_to_speech(reply_text)
            cleanup_old_audio()
            return reply_text, text, audio_file
        threshold = 70
        if conf >= threshold or label_id not in labels:
            reply_text = f"Autenticación fallida (conf={int(conf)})."
            audio_file = text_to_speech(reply_text)
            cleanup_old_audio()
            return reply_text, text, audio_file
        user = labels[label_id]

    # ejecutar handler
    try:
        result_text = parsed["handler"](parsed["params"])
    except Exception as e:
        result_text = f"Error ejecutando {parsed['name']}: {e}"

    audio_file = text_to_speech(result_text)
    try:
        with open("actions_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.time()},{user},{parsed['name']},{text},{result_text}\n")
    except Exception:
        pass

    cleanup_old_audio()
    return result_text, text, audio_file

# --- INTERFAZ GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown("# Asistente por voz + autenticación facial")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Registrar usuario")
            name_in = gr.Textbox(label="Nombre")
            cam_img = gr.Image(type="numpy", label="Foto del rostro (sube o arrastra)")
            enroll_btn = gr.Button("Registrar")
            enroll_out = gr.Textbox()
            enroll_img = gr.Image(label="Última imagen guardada")
            enroll_btn.click(enroll_user, inputs=[name_in, cam_img], outputs=[enroll_out, enroll_img])

            train_btn = gr.Button("Entrenar modelo")
            train_out = gr.Textbox()
            train_btn.click(do_train, inputs=None, outputs=[train_out])

        with gr.Column():
            gr.Markdown("## Usar asistente")
            mic = gr.Audio(type="filepath", label="Graba o sube audio")
            cam_use = gr.Image(type="numpy", label="Foto para autenticación (solo necesaria para comandos sensibles)")
            run_btn = gr.Button("Enviar")
            resp_text = gr.Textbox(label="Respuesta")
            trans_text = gr.Textbox(label="Transcripción")
            audio_out = gr.Audio(label="Audio respuesta")
            run_btn.click(recognize_flow, inputs=[mic, cam_use], outputs=[resp_text, trans_text, audio_out])

    gr.Markdown("**Comandos principales:**\n\n- Hola / Saluda\n- ¿Qué hora es?\n- ¿Qué día es hoy? / " \
    "¿Qué día será mañana?\n\n**Comando " \
    "sensible (requiere foto y modelo entrenado):**\n- Abrir google / Abrir youtube / Abrir wikipedia / Abrir Bloc de Notas\n- Buscar <término> (Buscar en google) " \
    "— ej: 'buscar clima Bogotá'\n- Abrir en YouTube <término> — ej: 'Buscar en youtube música latina'")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
