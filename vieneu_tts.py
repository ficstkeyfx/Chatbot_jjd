"""Local VieNeu v3 speech synthesis; uses the CPU ONNX backend."""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any

from tts_text import prepare_for_speech


class VieneuSpeech:
    def __init__(self) -> None:
        from vieneu import Vieneu

        self.model: Any = Vieneu(backend="onnx")
        self.voices: dict[str, Any] = {"vieneu-default": None}
        for description, voice_id in self.model.list_preset_voices():
            label = unicodedata.normalize("NFKD", str(description)).encode("ascii", "ignore").decode()
            slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:60]
            name = f"vieneu-{slug or len(self.voices)}"
            if name in self.voices:
                name = f"{name}-{len(self.voices)}"
            self.voices[name] = voice_id

    def synthesize(self, text: str, speed: float, voice_name: str) -> bytes:
        if voice_name not in self.voices:
            raise ValueError("Giọng VieNeu không tồn tại. Hãy chọn giọng trong Cài đặt.")
        sentences = prepare_for_speech(text)
        if not sentences:
            raise ValueError("Không có nội dung có thể đọc thành tiếng.")
        voice_id = self.voices[voice_name]
        kwargs = {}
        if voice_id is not None:
            kwargs["voice"] = self.model.get_preset_voice(voice_id)
        audio = self.model.infer(text=" ".join(sentences), **kwargs)
        # VieNeu v3 accepts extra kwargs but does not apply a speed parameter.
        # Stretch the waveform while retaining pitch for the web's speed control.
        if abs(speed - 1.0) > 0.01:
            import librosa

            audio = librosa.effects.time_stretch(audio, rate=speed)
        import soundfile as sf

        output = io.BytesIO()
        sf.write(output, audio, self.model.sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()
