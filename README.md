# Darwix AI Assessment

This repository contains a Django-based web application with two AI-powered features:

1. **Audio Transcription with Speaker Diarization** (Feature 1)(Also supports multilingual audio)
2. **Blog Post Title Suggestions** (Feature 2)

Below you’ll find:

* How each feature is implemented and where the code lives
* Setup instructions (dependencies, migration, running the server)
* How to test both features via `curl` commands

---

## Project Structure

```
darwix-ai-assessment/
├── ai_features/
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py             ← Feature 1 implementation (transcription + diarization)
│   ├── title_suggestions.py  ← Feature 2 implementation (AI title generation)
│   └── … (other standard Django files)
├── darwix_assessment/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── … (other standard Django files)
├── manage.py
├── requirements.txt
└── README.md               
```

* **Feature 1 (Audio Transcription + Diarization)**

  * **Code in:** `ai_features/views.py`
  * Entry point: Django view `transcribe_audio(request)`
  * Uses:

    * **Pyannote** (`pyannote.audio.Pipeline`) for speaker diarization
    * **Whisper** (`whisper.load_model(...)`) for multilingual speech‐to‐text
    * **pydub** (with bundled FFmpeg) for audio normalization/slicing

* **Feature 2 (Blog Post Title Suggestions)**

  * **Code in:** `ai_features/title_suggestions.py`
  * Entry point: Django view `title_suggestions(request)`
  * Uses:

    * **Flan-T5-Large** (Hugging Face Transformers) to generate multiple title candidates
    * NLTK for sentence tokenization and keyword extraction
    * Heuristics to score, dedupe, and pick three diverse titles

---

## 1. Setup & Installation

1. **Clone the repository** (or download as ZIP and extract):

   ```bash
   git clone https://github.com/<your-username>/darwix-ai-assessment.git
   cd darwix-ai-assessment
   ```

2. **Create and activate a Python 3.11 virtual environment** (Windows example):

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > The main dependencies include:
   >
   > * Django 5.2.1
   > * Pyannote.audio (for speaker diarization)
   > * OpenAI Whisper (for speech‐to‐text)
   > * Transformers (Flan-T5-Large)
   > * Pydub + FFmpeg (for audio conversion)
   > * NLTK


4. **Collect or verify FFmpeg**

   * We expect FFmpeg to be placed under `ai_features/ffmpeg/bin/ffmpeg.exe`.
   * If you don’t have it there, install FFmpeg for Windows and point `AudioSegment.converter` in `views.py` to that path, or add FFmpeg to your system `PATH`.

5. **Apply Migrations & Start the Server**:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

   The development server will start at `http://127.0.0.1:8000/`.

---

## 2. Feature 1: Audio Transcription with Speaker Diarization

### 2.1 Implementation

* **Endpoint:**

  ```
  POST  http://127.0.0.1:8000/api/transcribe/
  ```

* **View:**

  ```python
  # ai_features/views.py

  @csrf_exempt
  def transcribe_audio(request):
      # 1. Save uploaded file under media/uploads/
      # 2. Normalize → 16kHz mono WAV → media/processed/
      # 3. Run Pyannote diarization pipeline on processed WAV
      # 4. Slice each speaker segment, save under media/segments/
      # 5. Transcribe each slice via Whisper → segments list
      # 6. Return JSON: {"status":"success", "segments":[ { speaker, start_time, end_time, transcript } ]}
  ```

* **Model Loading** (module‐level in `views.py`):

  ```python
  import whisper
  from pyannote.audio import Pipeline

  whisper_model = whisper.load_model("small")  # or "base", "tiny"
  diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
  ```

* **Audio Conversion:**

  ```python
  sound = AudioSegment.from_file(orig_path)
  sound = sound.set_frame_rate(16000).set_channels(1)
  sound.export(processed_path, format="wav")
  ```

* **Output JSON Format:**

  ```json
  {
    "status": "success",
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "start_time": 0.031,
        "end_time": 4.283,
        "transcript": "Hello, this is speaker zero ..."
      },
      {
        "speaker": "SPEAKER_01",
        "start_time": 5.684,
        "end_time": 15.910,
        "transcript": "Speaker one says something ..."
      }
      // ...
    ]
  }
  ```

### 2.2 Testing with `curl`

From your Windows command prompt (assuming the server is running at port 8000):

```powershell
curl -X POST "http://localhost:8000/api/transcribe/" -F "audio_file=@D:\darwix-ai-assessment\media\uploads\sample-2.WAV"
```

* 
* The response will be printed in JSON, showing speaker‐segmented transcripts. Also there will be a .txt file created for you to view the outputs of each speaker and what they say in the .txt file present in segments folder inside media folder.

---

## 3. Feature 2: AI-Powered Blog Title Suggestions

### 3.1 Implementation

* **Endpoint:**

  ```
  POST  http://127.0.0.1:8000/api/title_suggestions/
  ```

* **View:**

  ```python
  # ai_features/title_suggestions.py

  @csrf_exempt
  def title_suggestions(request):
      # 1. Parse JSON body: {"content": "<full blog post text>"}
      # 2. “Smart” summary + keyword extraction via NLTK
      # 3. Use Flan-T5-Large (Hugging Face) to generate ~15 candidate titles across 3 random prompt templates
      # 4. Clean and dedupe candidates (strip numbering, punctuation, title‐case)
      # 5. Score each candidate by:
      #      • Keyword presence
      #      • Ideal length (~8 words)
      #      • Penalize high overlap with summary
      #      • Bonus for question marks or numbers
      #      • Penalize generic starters (“a guide to…”)
      # 6. Select top 3, ensuring diversity via Jaccard similarity
      # 7. Return JSON: {"titles": ["Title 1", "Title 2", "Title 3"]}
  ```

* **Model Loading**:

  ```python
  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

  model_name = "google/flan-t5-large"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
  ```

* **Smart Summarization & Keywords**:

  ```python
  class SmartContentExtractor:
      def extract_keywords(...):  # top N by frequency (non-stopwords)
      def short_summary(...):     # pick ~30–70 words from first/important sentences
  extractor = SmartContentExtractor()
  ```

* **Candidate Generation**: Randomly pick 3 strategies out of 5, each with:

  * Different prompt templates (Catchy, Question, Benefit, Descriptive, Intriguing)
  * Slightly different `temperature`, `top_p`, `no_repeat_ngram_size`
  * Request `num_return_sequences = total_candidates_target // 3` (recommended 20 total → \~6–7 per strategy)

* **Scoring & Selection**:

  ```python
  def jaccard_similarity(a, b): ...
  def score_and_select_titles(content, candidates, top_n=3, diversity_threshold=0.35): ...
  ```

* **Output JSON Format:**

  ```json
  {
    "titles": [
      "Integrating AI Models into Django: A 2025 Guide",
      "Building SEO-Friendly Django Headlines with Flan-T5",
      "Real-Time AI-Powered Blog Titles Using Open-Source Tools"
    ]
  }
  ```

### 3.2 Testing with `curl`

```powershell
curl -X POST http://localhost:8000/api/title_suggestions/ -H "Content-Type: application/json" -d "{\"content\": \"In the fast-paced world of 2025, the integration of Python, Django, and AI technologies is not just a trend but a necessity for modern web applications. Developers are increasingly combining the simplicity and power of Django with cutting-edge AI models to build smart, responsive, and user-friendly systems. This blog explores best practices for such integration, including architectural patterns, deployment strategies, and model selection. From utilizing pre-trained transformers for text summarization and classification, to embedding real-time inference capabilities with Django REST Framework, the landscape has drastically evolved. Moreover, the use of open-source libraries like Hugging Face Transformers, PyTorch, TensorFlow, and spaCy makes it easier than ever to implement powerful AI features with minimal overhead. We also explore how asynchronous views and Celery can improve performance, especially when dealing with heavy models. Containerization using Docker, automated deployments using GitHub Actions, and monitoring with Prometheus and Grafana are also discussed. As privacy concerns rise, we cover responsible AI practices including ethical data sourcing, explainability, and fairness metrics. Finally, we look into the future of AI-native web applications where users interact with intelligent systems in real time, and where developers are empowered with tools that abstract away the complexity of model serving. This comprehensive guide is meant for intermediate to advanced developers looking to stay ahead in the game by building AI-enhanced Django applications that are production-ready, scalable, and maintainable.\"}"
```

* Note the escaping of quotes in Windows `cmd`.
* The response JSON will list three suggested blog titles.

---

## 4. Additional Tips

* **Media Folder**

  * By default, uploaded audio files should go under `media/uploads/`.
  * Diarized slices are saved to `media/segments/`; processed 16 kHz WAVs go to `media/processed/`.

* **FFmpeg**

  * If you encounter an FFmpeg “not found” error, download the [static build of FFmpeg for Windows](https://ffmpeg.org/download.html), place `ffmpeg.exe` under, e.g., `ai_features/ffmpeg/bin/ffmpeg.exe`, and ensure in `views.py`:

    ```python
    AudioSegment.converter = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
    ```

* **GPU vs. CPU**

  * Whisper and Pyannote models run faster on GPU.
  * If you don’t have a GPU, they’ll run on CPU but may be slow for long audio.
  * Flan-T5-Large also benefits from GPU; on CPU it can be noticeably slower. Consider switching to `flan-t5-base` if CPU inference is too slow.

* **Error Handling**

  * If an audio slice fails to export or Whisper fails on a slice, the code continues to the next segment so you still get partial output.
  * If title generation fails or the input is < 50 characters, you get a JSON error response.

---

## 5. Summary

* **Feature 1 (Audio Transcription)**

  1. Upload audio via `/api/transcribe/`
  2. Diarize speakers, slice, and transcribe with Whisper
  3. Get JSON with speaker timestamps and transcripts

* **Feature 2 (Title Suggestions)**

  1. Send blog content to `/api/title_suggestions/` (JSON‐formatted)
  2. Pipeline: summary + keywords → Flan-T5 candidate generation → scoring & diversity
  3. Get JSON with three AI-generated titles

You can test both from the command line via the provided `curl` commands. Simply run the Django dev server and execute those `curl` lines exactly as shown.

---

> *Created as part of the Darwix AI Assessment by Aviral.*
