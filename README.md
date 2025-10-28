README — Asistente por voz + autenticación facial (Python / Gradio)

Resumen:
Proyecto que combina reconocimiento de voz (ASR), síntesis de voz (TTS) y reconocimiento facial (LBPH con OpenCV) en una interfaz web simple (Gradio). Permite registrar usuarios (foto), entrenar un modelo LBPH y ejecutar comandos por voz. Algunos comandos pueden requerir autenticación facial antes de ejecutarse.

Estructura principal de archivos (lo que importa)

app.py — código principal (la versión que compartiste).

lbph_data/ — carpeta creada automáticamente para almacenar imágenes y modelo:

lbph_data/images/ — fotos de registro (una por usuario o varias).

lbph_data/recognizer.yml — modelo LBPH guardado (después de entrenar).

lbph_data/labels.pkl — mapeo id → nombre.

tmp_audio/ — archivos mp3 temporales de TTS (se limpian automáticamente).

actions_log.txt — log simple de acciones (opcional).

Requisitos de sistema (recomendado)

Python 3.9 / 3.10 / 3.11 (3.12 puede funcionar pero algunas ruedas binarias fallan en Windows).

Windows 10/11 (las instrucciones incluyen comandos Windows/Git Bash). Linux / macOS funcionan también con ajustes menores.

Conexión a internet (gTTS y Google Speech API necesitan internet).

Instalación paso a paso (Windows, Git Bash / PowerShell)

Abre Git Bash o PowerShell en la carpeta del proyecto.

1) Crear y activar entorno virtual
# Crear venv
python -m venv venv

# Activar en Git Bash / WSL:
source venv/Scripts/activate

# O activar en PowerShell:
# .\venv\Scripts\Activate.ps1

# O activar en CMD:
# venv\Scripts\activate


Si PowerShell bloquea la ejecución de scripts: abre PowerShell como administrador y ejecuta Set-ExecutionPolicy RemoteSigned -Scope CurrentUser (si entiendes los riesgos).

2) Actualizar pip, wheel y setuptools
python -m pip install --upgrade pip setuptools wheel

3) Instalar dependencias básicas
pip install gradio gTTS SpeechRecognition pillow numpy

4) Instalar OpenCV con soporte face (LBPH)

Necesitas la versión con contrib (incluye cv2.face):

pip install opencv-contrib-python


Problemas habituales:

Si pip intenta compilar opencv y falla, puede deberse a versión de Python o a falta de ruedas precompiladas para tu versión. Opciones:

Usar Python 3.10/3.11 en vez de 3.12.

Instalar desde conda: conda install -c conda-forge opencv (si usas conda).

Instalar una rueda precompilada adecuada (si sabes hacerlo).

5) (Opcional) Soporte para grabar desde micrófono localmente

Si quieres grabar micro desde Python local (no solo en Gradio), necesitas PyAudio. En Windows es sencillo con pipwin:

pip install pipwin
pipwin install pyaudio


Si no vas a grabar desde Python (Gradio/Browser puede grabar), no es obligatorio.

6) Generar requirements.txt (opcional)
pip freeze > requirements.txt

Ejecutar la aplicación

Con el venv activado:

python app.py


Al iniciarse Gradio mostrará una URL tipo http://127.0.0.1:7860 o http://0.0.0.0:7860. Ábrela en el navegador.
