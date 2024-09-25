
from faster_whisper import WhisperModel
from time import time



def runDetect(model_size, audio_path, compute_type="int8"):
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)


    segments, info = model.transcribe(audio_path, 
                                      beam_size=5, 
                                      vad_filter = True,
                                      language="zh", 
                                      condition_on_previous_text=False)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    start = time()
    model_size_or_model_path = "medium" # distil-large-v2 distil-medium large-v3  medium
    model_size = "local_models/medium_model"
    audio_path = "media/sample-9s.wav"
    audio_path = "media/Sound-Clip-1-SIMPLIFIED-CHINESE.mp3"
    runDetect(model_size, audio_path)
    end = time()

    print("spent: ", end - start)


