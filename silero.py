import torch
import soundfile as sf
import numpy as np
import re
from scipy.signal import resample_poly

# =========================================================
# Load reference sentence by ID (e.g. 1:)
# =========================================================
def load_reference_by_id(path: str, sentence_id: int) -> str:
    prefix = f"{sentence_id}:"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                # Remove "1:" prefix
                text = line[len(prefix):].strip()
                # Remove optional noise annotations in parentheses
                text = re.sub(r"\(.*?\)", "", text).strip()
                return text

    raise ValueError(f"Sentence ID {sentence_id} not found in transcript file")


# =========================================================
# Punctuation-aware tokenizer (case preserved)
# =========================================================
def tokenize_with_punctuation(text: str) -> list[str]:
    """
    Split text into words and punctuation as separate tokens.
    Example: "horn." -> ["horn", "."]
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens  # preserve original case


# =========================================================
# Word-level alignment (Levenshtein)
# =========================================================
def align_words(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1],  # substitution
                )

    # Backtrace
    i, j = n, m
    ops = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            ops.append(("CORRECT", ref_tokens[i-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("SUB", ref_tokens[i-1], hyp_tokens[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("DEL", ref_tokens[i-1]))
            i -= 1
        else:
            ops.append(("INS", hyp_tokens[j-1]))
            j -= 1

    return reversed(ops)


# =========================================================
# Silero STT transcription
# =========================================================
def transcribe_audio(path: str) -> str:
    device = torch.device("cpu")

    model, decoder, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language="en",
        device=device
    )

    (_, _, _, prepare_model_input) = utils

    # Load audio safely
    audio, sr = sf.read(path)

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to 16kHz (Silero requirement)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr)

    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    input_tensor = prepare_model_input(audio_tensor, device=device)

    output = model(input_tensor)
    return decoder(output[0].cpu())


# =========================================================
# MAIN (Batch processing: 1.wav ↔ 1:, 2.wav ↔ 2:, ...)
# =========================================================
if __name__ == "__main__":

    audio_dir = r"C:\Users\ayham\Desktop\Downloads\HMI SpeechRecognition Dataset\Audio Files HMI\Non-Native Female\wav"
    transcript_path = r"C:\Users\ayham\Desktop\Downloads\HMI SpeechRecognition Dataset\Audio Files HMI\transcript.txt"

    num_sentences = 11
    total_wer = 0.0

    for sentence_id in range(1, num_sentences + 1):

        print("\n" + "=" * 60)
        print(f"Processing sentence {sentence_id}")
        print("=" * 60)

        audio_path = f"{audio_dir}\\{sentence_id}.wav"

        # Load reference
        reference_text = load_reference_by_id(transcript_path, sentence_id)
        print("\nReference text:")
        print(reference_text)

        # Transcribe
        print("\n--- Transcribing audio ---")
        recognized_text = transcribe_audio(audio_path)
        print("\nRecognized text:")
        print(recognized_text)

        # Tokenize
        ref_tokens = tokenize_with_punctuation(reference_text)
        rec_tokens = tokenize_with_punctuation(recognized_text)

        # Align
        ops = list(align_words(ref_tokens, rec_tokens))

        print("\n--- Word-level analysis ---")
        for op in ops:
            if op[0] == "CORRECT":
                print(f"CORRECT: {op[1]}")
            elif op[0] == "SUB":
                print(f"SUBSTITUTION: {op[1]} → {op[2]}")
            elif op[0] == "DEL":
                print(f"DELETION: {op[1]}")
            elif op[0] == "INS":
                print(f"INSERTION: {op[1]}")

        # WER calculation
        subs = sum(1 for o in ops if o[0] == "SUB")
        dels = sum(1 for o in ops if o[0] == "DEL")
        ins  = sum(1 for o in ops if o[0] == "INS")

        wer = (subs + dels + ins) / len(ref_tokens)
        total_wer += wer

        
        print(f"\nWER for sentence {sentence_id}: {wer:.2%}")

    avg_wer = total_wer / num_sentences
    print("\n" + "=" * 60)
    print(f"AVERAGE WER over {num_sentences} sentences: {avg_wer:.2%}")
