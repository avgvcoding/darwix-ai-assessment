from django.shortcuts import render
import os
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from pydub import AudioSegment

import whisper
from pyannote.audio import Pipeline

ffmpeg_path = os.path.abspath(os.path.join("ai_features", "ffmpeg", "bin", "ffmpeg.exe"))
AudioSegment.converter = ffmpeg_path

ffmpeg_dir = os.path.dirname(ffmpeg_path)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

try:
    whisper_model = whisper.load_model("small")  
except Exception as e:
    print("Failed to load Whisper model:", e)
    raise

try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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"
    )
except Exception as e:
    print("Failed to load Pyannote pipeline:", e)
    raise


@csrf_exempt
def transcribe_audio(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST requests are allowed.")

    audio_file = request.FILES.get("audio_file")
    if audio_file is None:
        return HttpResponseBadRequest("No audio_file field in request.")

    uploads_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    orig_path = os.path.join(uploads_dir, audio_file.name)
    with open(orig_path, "wb") as out_file:
        for chunk in audio_file.chunks():
            out_file.write(chunk)

    processed_dir = os.path.join(settings.MEDIA_ROOT, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    base_name, _ = os.path.splitext(audio_file.name)
    processed_filename = base_name + ".wav"
    processed_path = os.path.join(processed_dir, processed_filename)

    try:
        sound = AudioSegment.from_file(orig_path)  
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(processed_path, format="wav")
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Audio conversion failed: {str(e)}"
        }, status=500)

    try:
        diarization = diarization_pipeline(processed_path)
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Diarization failed: {str(e)}"
        }, status=500)

    try:
        full_audio = AudioSegment.from_wav(processed_path)
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Failed to reload processed WAV: {str(e)}"
        }, status=500)

    segments_dir = os.path.join(settings.MEDIA_ROOT, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    response_segments = []

    for turn_index, (segment, _track_index, speaker_label) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
        start_s = segment.start  
        end_s = segment.end

        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)

        slice_filename = f"{base_name}_seg{turn_index}.wav"
        slice_path = os.path.join(segments_dir, slice_filename)

        try:
            audio_slice = full_audio[start_ms:end_ms]
            audio_slice.export(slice_path, format="wav")
        except Exception:
            continue  

        try:
            result = whisper_model.transcribe(slice_path, fp16=False)
            transcript_text = result.get("text", "").strip()
        except Exception as e:
            transcript_text = f"[Error transcribing segment: {str(e)}]"

        response_segments.append({
            "speaker": speaker_label,
            "start_time": round(start_s, 3),
            "end_time": round(end_s, 3),
            "transcript": transcript_text
        })

    transcript_txt_filename = f"{base_name}_transcript.txt"
    transcript_txt_path = os.path.join(segments_dir, transcript_txt_filename)

    try:
        with open(transcript_txt_path, "w", encoding="utf-8") as txt_file:
            for seg in response_segments:
                start_mmss = f"{int(seg['start_time'] // 60):02d}:{(seg['start_time'] % 60):06.3f}"
                end_mmss = f"{int(seg['end_time']   // 60):02d}:{(seg['end_time']   % 60):06.3f}"

                line = f"[{start_mmss} ‒ {end_mmss}] {seg['speaker']}: {seg['transcript']}\n"
                txt_file.write(line)
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Failed to write transcript text file: {str(e)}"
        }, status=500)

    return JsonResponse({
        "status": "success",
        "segments": response_segments,
        "transcript_txt": transcript_txt_path
    })
