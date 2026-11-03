import argparse, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import soundfile as sf

TEXT = "At number five, we have a movie that completely changed the game. The action is intense, the story keeps you guessing, and the final scene is unforgettable. But wait until you see number four."
VOICES = {
    "warguy": "https://raw.githubusercontent.com/justsomeoneretarded/Top10-Movies/main/assets/voices/warguy.ogg",
    "kit-laser": "https://raw.githubusercontent.com/justsomeoneretarded/Top10-Movies/main/assets/voices/kit-laser.ogg",
}
def reference(name):
    import urllib.request
    Path("references").mkdir(exist_ok=True)
    src = Path("references") / (name + ".ogg")
    dst = Path("references") / (name + ".wav")
    urllib.request.urlretrieve(VOICES[name], src)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)], check=True)
    return str(dst)

def generate(model, ref, dest):
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    if model.startswith("chatterbox"):
        if model == "chatterbox-turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            cls = ChatterboxTurboTTS
        elif model == "chatterbox-original":
            from chatterbox.tts import ChatterboxTTS
            cls = ChatterboxTTS
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            cls = ChatterboxMultilingualTTS
        engine = cls.from_pretrained(device="cpu")
        kwargs = {"audio_prompt_path": ref}
        if model == "chatterbox-multilingual":
            kwargs["language_id"] = "en"
        audio = engine.generate(TEXT, **kwargs)
        sf.write(dest, audio.squeeze().detach().cpu().numpy(), engine.sr)
    elif model == "f5-tts":
        from f5_tts.api import F5TTS
        engine = F5TTS(device="cpu")
        wav, sr, _ = engine.infer(ref_file=ref, ref_text="", gen_text=TEXT)
        sf.write(dest, wav, sr)
    elif model == "cosyvoice":
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        engine = CosyVoice2("pretrained_models/CosyVoice2-0.5B", load_jit=False, load_onnx=False, load_trt=False)
        chunks = []
        sr = engine.sample_rate
        for result in engine.inference_zero_shot(TEXT, "This is a sample of the reference speaker.", load_wav(ref, 16000), stream=False):
            chunks.append(result["tts_speech"].squeeze().cpu().numpy())
        sf.write(dest, np.concatenate(chunks), sr)
    elif model == "fish-speech":
        raise RuntimeError("Fish Speech CLI integration requires pinned compatible model weights and environment; not implemented yet")
    else:
        raise ValueError(model)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    failures = []
    for name in VOICES:
        start = time.monotonic()
        try:
            ref = reference(name)
            dest = f"outputs/{args.model}_{name}.wav"
            generate(args.model, ref, dest)
            print(f"SUCCESS {name}: {dest}, seconds={time.monotonic()-start:.1f}", flush=True)
        except Exception:
            traceback.print_exc()
            failures.append(name)
    Path("outputs/report.txt").write_text(f"Model: {args.model}\nFailed voices: {failures}\n")
    if failures:
        sys.exit(1)
if __name__ == "__main__":
    main()
