import whisper
import json

model = whisper.load_model("large-v2")
result = model.transcribe(audio = "audios/12_Exercise 1 - Pure HTML Media Player.mp3",
                          language = "hi",
                          task = "translate",
                          word_timestamps=False)

chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})
print(chunks)