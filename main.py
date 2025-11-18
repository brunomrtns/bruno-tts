#!/usr/bin/env python3
"""
CLI simples para síntese de fala usando XTTS-v2 com clonagem de voz.

Exemplos de uso:

    # Texto direto na linha de comando
    python main.py \
        --speaker-wav ~/tts/voice.wav \
        --text "Veja esse travesseiro modular popular que vira uma cama automaticamente." \
        --out saida_xtts_ptbr.wav

    # Texto vindo de um arquivo
    python main.py \
        --speaker-wav ~/tts/voice.wav \
        --text-file texto.txt \
        --out outputs/tts.wav \
        --lang pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from TTS.api import TTS


DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera áudio (TTS) usando XTTS-v2 com clonagem de voz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Nome do modelo TTS a ser usado.",
    )

    parser.add_argument(
        "--speaker-wav",
        type=str,
        required=True,
        help="Caminho para o arquivo WAV de referência da voz (6–10s).",
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        help="Texto a ser sintetizado.",
    )
    text_group.add_argument(
        "--text-file",
        type=str,
        help="Caminho para um arquivo de texto com o conteúdo a ser sintetizado.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Caminho do arquivo de saída (WAV).",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="pt",
        help="Código de idioma (por ex.: 'pt', 'en', 'es').",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo de execução do modelo.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de log.",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(message)s",
    )


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            logging.info("CUDA disponível, usando GPU.")
            return "cuda"
        logging.info("CUDA não disponível, usando CPU.")
        return "cpu"

    if device_arg == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA selecionado, mas não disponível. Usando CPU.")
        return "cpu"

    return device_arg


def read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()

    # Lê de arquivo
    text_path = Path(os.path.expanduser(args.text_file))
    if not text_path.is_file():
        raise FileNotFoundError(f"Arquivo de texto não encontrado: {text_path}")

    content = text_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Arquivo de texto está vazio: {text_path}")
    return content


def validate_paths(speaker_wav: Path, out_path: Path) -> None:
    if not speaker_wav.is_file():
        raise FileNotFoundError(f"Arquivo de voz não encontrado: {speaker_wav}")

    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        logging.info("Criando diretório de saída: %s", out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)


def synthesize(
    model_name: str,
    speaker_wav: Path,
    text: str,
    out_path: Path,
    language: str,
    device: str,
) -> None:
    logging.info("Carregando modelo: %s", model_name)
    tts = TTS(model_name).to(device)
    logging.info("Modelo carregado com sucesso em '%s'.", device)

    logging.info("Iniciando síntese...")
    logging.debug("Texto (%d chars): %s", len(text), text)

    # XTTS aceita string única ou lista de arquivos de referência.
    tts.tts_to_file(
        text=text,
        file_path=str(out_path),
        speaker_wav=[str(speaker_wav)],
        language=language,
    )

    logging.info("Áudio gerado em: %s", out_path)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        device = resolve_device(args.device)

        speaker_wav = Path(os.path.expanduser(args.speaker_wav))
        out_path = Path(os.path.expanduser(args.out))

        validate_paths(speaker_wav, out_path)

        text = read_text(args)

        synthesize(
            model_name=args.model_name,
            speaker_wav=speaker_wav,
            text=text,
            out_path=out_path,
            language=args.lang,
            device=device,
        )

        return 0

    except Exception as exc:  # noqa: BLE001 - aqui queremos logar qualquer erro alto nível
        logging.error("Erro durante a síntese: %s", exc)
        logging.debug("Detalhes completos:", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
