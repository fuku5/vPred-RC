from pathlib import Path

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET
MAX_CAPTCHA = 5

RECOGNIZER_DIR = Path(__file__).resolve().parent / 'results/recognizer'
RECOGNIZER_DIR.mkdir(exist_ok=True, parents=True)

NPZ_DIR = RECOGNIZER_DIR / 'npz'
NPZ_DIR.mkdir(exist_ok=True, parents=True)
