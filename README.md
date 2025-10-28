[README.md](https://github.com/user-attachments/files/23198096/README.md)
# 🤖 Asistente por Voz con Autenticación Facial (Python + Gradio)

Proyecto que combina **reconocimiento de voz (ASR)**, **síntesis de voz (TTS)** y **reconocimiento facial (LBPH con OpenCV)** en una interfaz web simple usando **Gradio**.  
Permite registrar usuarios con su rostro, entrenar un modelo LBPH y ejecutar comandos por voz.  
Algunos comandos requieren **autenticación facial** antes de ejecutarse.

---

## 📂 Estructura del proyecto

```
App2/
│
├── app.py                 # Código principal
├── requirements.txt        # Dependencias (opcional)
├── lbph_data/              # Datos de reconocimiento facial
│   ├── images/             # Fotos registradas por usuario
│   ├── recognizer.yml      # Modelo LBPH entrenado
│   └── labels.pkl          # Mapeo ID → nombre
│
├── tmp_audio/              # Archivos MP3 generados por TTS
├── actions_log.txt         # Log de acciones (opcional)
└── README.md               # Este archivo
```

---

## ⚙️ Requisitos del sistema

- **Python:** 3.9 / 3.10 / 3.11  
  (⚠️ Python 3.12 puede fallar con OpenCV)
- **Sistema operativo:** Windows 10/11, Linux o macOS  
- **Internet:** necesario para gTTS y reconocimiento de voz  

---

## 🚀 Instalación paso a paso (Windows / Git Bash o PowerShell)

### 1️⃣ Crear y activar entorno virtual
```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash / WSL
# .\venv\Scripts\Activate.ps1  # PowerShell
```

> Si PowerShell bloquea scripts:  
> Ejecuta como admin →  
> `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### 2️⃣ Actualizar pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

---

### 3️⃣ Instalar dependencias
```bash
pip install gradio gTTS SpeechRecognition pillow numpy
pip install opencv-contrib-python
```

---

### 4️⃣ (Opcional) Soporte para micrófono local
```bash
pip install pipwin
pipwin install pyaudio
```

---

### 5️⃣ (Opcional) Generar requirements.txt
```bash
pip freeze > requirements.txt
```

---

## 🧠 Entrenamiento y uso

### 🔹 Registrar usuario
1. En la interfaz de Gradio, escribe un nombre.  
2. Sube o toma una foto del rostro.  
3. Pulsa **Registrar** → la imagen se guarda en `lbph_data/images/`.

---

### 🔹 Entrenar modelo
1. Pulsa **Entrenar modelo**.  
2. Se genera el modelo `lbph_data/recognizer.yml` y etiquetas `labels.pkl`.  

---

### 🔹 Usar asistente
1. Sube o graba un audio.  
2. (Opcional) Sube foto para autenticación si el comando lo requiere.  
3. Pulsa **Enviar** → El asistente transcribe, responde con voz y texto.

---

## 🗣️ Comandos disponibles

| Tipo | Ejemplo | Requiere rostro |
|------|----------|----------------|
| Información | “qué hora es”, “qué día es hoy” | ❌ |
| Navegación | “abrir google”, “buscar clima Bogotá” | ✅ |
| Música / Búsqueda | “buscar en YouTube música latina” | ✅ |
| Saludo | “hola”, “buen día” | ❌ |
