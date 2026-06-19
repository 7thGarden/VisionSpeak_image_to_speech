from flask import Flask, request, render_template
import os, uuid

# 1. Import your newly integrated model functions
from models.caption_model import get_caption
from models.tts_model import save_audio

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER  = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER,  exist_ok=True)


def generate_caption(image_path: str) -> str:
    # 2. Call the real captioning model
    return get_caption(image_path)


def generate_audio(caption: str, filename: str) -> str:
    audio_name = filename + ".wav"
    audio_path = os.path.join(AUDIO_FOLDER, audio_name)
    
    # 3. Call the real TTS model
    save_audio(caption, audio_path)
    
    return audio_name

# ... (Keep all your existing @app.route functions exactly as they are) ...


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    action = request.form.get('action')   # "caption" | "voice" | "both"
    file   = request.files.get('image')

    if not file or not action:
        return render_template('index.html')

    # Save uploaded image
    img_name = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    img_path = os.path.join(UPLOAD_FOLDER, img_name)
    file.save(img_path)

    caption = None
    audio   = None

    # ── Action branching ──────────────────────────────────────
    if action == 'caption':
        # Caption ONLY – no audio generated
        caption = generate_caption(img_path)

    elif action == 'voice':
        # Voice ONLY – still needs caption internally, but don't show it
        hidden_caption = generate_caption(img_path)
        audio = generate_audio(hidden_caption, img_name.split('.')[0])

    elif action == 'both':
        # Caption AND Voice
        caption = generate_caption(img_path)
        audio   = generate_audio(caption, img_name.split('.')[0])
    # ─────────────────────────────────────────────────────────

    return render_template(
        'result.html',
        image=img_name,
        caption=caption,
        audio=audio
    )


if __name__ == '__main__':
    app.run(debug=True)