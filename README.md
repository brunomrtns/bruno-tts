# Bruno TTS

Text-to-Speech com XTTS v2 — geracao de audio em PT-BR a partir de uma voz de referencia.

## Como funciona

Usa o modelo [XTTS v2](https://huggingface.co/coqui/XTTS-v2) da Coqui AI para clonar a voz a partir de um arquivo de audio curto (6+ segundos) e gerar fala em portugues com a mesma timbre vocal.

## Requisitos

- Python 3.10+
- PyTorch (CPU ou CUDA)
- 4GB+ RAM (CPU) ou GPU com 2GB+ VRAM

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install TTS torch
```

## Uso

```bash
python main.py \
  --speaker-wav /path/to/voice.wav \
  --text "Veja esse travesseiro modular popular que vira uma cama automaticamente." \
  --out saida.wav \
  --lang pt \
  --device auto
```

### Parametros

| Parametro | Descricao | Padrao |
|-----------|-----------|--------|
| `--speaker-wav` | Arquivo de audio de referencia (6+ segundos) | Obrigatorio |
| `--text` | Texto para sintetizar | Obrigatorio |
| `--out` | Arquivo de saida | `output.wav` |
| `--lang` | Idioma (`pt`, `en`, `es`, `fr`, etc.) | `pt` |
| `--device` | Dispositivo (`auto`, `cpu`, `cuda`) | `auto` |

## Licenca

MIT
