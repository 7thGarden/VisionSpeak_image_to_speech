import torch
import librosa
import soundfile as sf
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier

# Adjust paths if your folders are located differently relative to app.py
LOCAL_MODEL_DIR = "../local_models"
HF_CACHE_DIR = f"{LOCAL_MODEL_DIR}/huggingface"
TTS_MODEL_PATH = "../speecht5_finetuned_ljspeech/final"
REF_AUDIO_PATH = "../LJSpeech-1.1/wavs/LJ001-0001.wav"

print("Loading TTS Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained(TTS_MODEL_PATH)
model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL_PATH).to(device)
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan", cache_dir=HF_CACHE_DIR, local_files_only=True
).to(device)

spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir=f"{LOCAL_MODEL_DIR}/speechbrain_voxceleb", 
    run_opts={"device": device}
)

print("Extracting speaker embeddings...")
audio_array, _ = librosa.load(REF_AUDIO_PATH, sr=16000)
audio_tensor = torch.tensor(audio_array).unsqueeze(0).float().to(device)

with torch.no_grad():
    embedding = spk_model.encode_batch(audio_tensor)
    embedding = torch.nn.functional.normalize(embedding, dim=2)
    SPEAKER_EMBEDDINGS = embedding.squeeze().unsqueeze(0).to(device)

def save_audio(text: str, output_path: str):
    """Generates speech from text and saves it as a .wav file."""
    inputs = processor(text=text, return_tensors="pt")
    
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"].to(device), 
            SPEAKER_EMBEDDINGS, 
            vocoder=vocoder
        )
    
    # Save directly to the static/audio/ folder path provided by app.py
    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)