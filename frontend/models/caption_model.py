import torch
from transformers import pipeline, VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer

# Adjust paths if your folders are located differently relative to app.py
LOCAL_MODEL_DIR = "../local_models"
HF_CACHE_DIR = f"{LOCAL_MODEL_DIR}/huggingface"
MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

print("Loading Image Captioning Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True)
caption_model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True)

captioner = pipeline(
    "image-to-text", 
    model=caption_model, 
    image_processor=image_processor, 
    tokenizer=tokenizer,
    device=device
)

def get_caption(image_path: str) -> str:
    """Passes the image to the pipeline and returns the generated text."""
    result = captioner(image_path)
    return result[0]['generated_text']