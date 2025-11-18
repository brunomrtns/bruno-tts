import torch
from TTS.api import TTS

# 1) Escolhe dispositivo (GPU se tiver, senão CPU mesmo)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Modelo XTTS-v2 (multilíngue + clonagem de voz)
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# 3) Caminho do áudio de referência da voz (pt-BR)
# Grave um .wav seu falando em pt-BR (6–10s) e coloque aqui:
SPEAKER_WAV = "/home/bruno.martins/tts/voice.wav"

# 4) Texto e saída
TEXT = (
    "Veja esse travesseiro modular popular que vira uma cama automaticamente, "
    "Ele é sofisticado, com tecidos de alta qualidade e obsolescência!"
)
OUT_PATH = "saida_xtts_ptbr.wav"

print(f"Carregando modelo: {MODEL_NAME}")
tts = TTS(MODEL_NAME).to(device)

print("Gerando áudio em pt-BR com XTTS-v2 (voz clonada)...")
tts.tts_to_file(
    text=TEXT,
    file_path=OUT_PATH,
    speaker_wav=[SPEAKER_WAV],  # pode passar lista com 1 ou vários áudios
    language="pt",              # código de idioma para português / pt-BR
)

print(f"\nÁudio gerado em: {OUT_PATH}")
