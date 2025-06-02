import json
import torch
import re
from collections import Counter
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random 

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

MODEL_NAME = "google/flan-t5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval() 

class SmartContentExtractor:
    def __init__(self):
        self.stop = set(stopwords.words("english"))
        self.year_pat = re.compile(r"\b(19|20)\d{2}\b") 

    def extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        tokens = [
            w
            for w in word_tokenize(text.lower())
            if w.isalnum() and w not in self.stop and len(w) > 3
        ]
        most_common = Counter(tokens).most_common(top_n)
        return [w for w, _ in most_common]

    def short_summary(self, text: str, max_words: int = 70, min_words: int = 30) -> str:
        raw = self.year_pat.sub("<YEAR>", text)
        sents = sent_tokenize(raw)
        chosen, count = [], 0
        
        current_summary = ""
        for s in sents:
            words = s.split()
            if count + len(words) <= max_words:
                chosen.append(s)
                count += len(words)
                current_summary = " ".join(chosen)
            else:
                if count < min_words:
                    needed = max_words - count
                    chosen.append(" ".join(words[:needed]))
                    count += needed
                    current_summary = " ".join(chosen)
                break 

        if count < min_words and len(sents) > len(chosen):
            chosen, count = [], 0
            for s in sents:
                words = s.split()
                if count < min_words or (count + len(words) <= max_words) :
                    if count + len(words) > max_words: 
                        needed = max_words - count
                        chosen.append(" ".join(words[:needed]))
                        count += needed
                        break
                    else:
                        chosen.append(s)
                        count += len(words)
                else: 
                    break
            current_summary = " ".join(chosen)

        return current_summary.strip()


extractor = SmartContentExtractor()


def clean_title(raw: str) -> str:
    t = raw.strip().strip('\"\'').rstrip(".,;:!?") 
    t = re.sub(r"^\s*(\d+\.\s*|[-•*]\s*)", "", t)
    words = t.split()
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of'}
    capitalized_words = []
    for i, word in enumerate(words):
        if word.isupper() and len(word) > 1:
             capitalized_words.append(word)
        elif i == 0 or word.lower() not in small_words:
            capitalized_words.append(word.capitalize())
        else:
            capitalized_words.append(word.lower())
    
    final_title = " ".join(capitalized_words)
    if final_title and final_title[0].islower() and len(final_title) > 1:
        final_title = final_title[0].upper() + final_title[1:]
    return final_title


def jaccard_similarity(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0



PROMPT_STRATEGIES = [
    {
        "name": "Catchy General",
        "template": "Generate a catchy and engaging blog post title (between 5 and 12 words, title case). Do NOT copy any full sentence verbatim from the content. Keywords: {keywords}. Content (brief): {summary}. Title:",
        "num_beams": 1, "do_sample": True, "temperature": 0.85, "top_p": 0.92, "no_repeat_ngram_size": 2
    },
    {
        "name": "Question Style",
        "template": "Craft an intriguing question as a blog post title (5-12 words, title case) based on the following. Keywords: {keywords}. Content (brief): {summary}. Question Title:",
        "num_beams": 1, "do_sample": True, "temperature": 0.8, "top_p": 0.90, "no_repeat_ngram_size": 3 
    },
    {
        "name": "Benefit Oriented",
        "template": "Highlight the main benefit or a key solution in a blog post title (5-12 words, title case). Focus on what the reader will gain. Keywords: {keywords}. Content (brief): {summary}. Benefit Title:",
        "num_beams": 1, "do_sample": True, "temperature": 0.75, "top_p": 0.90, "no_repeat_ngram_size": 2
    },
    {
        "name": "Direct/Descriptive",
        "template": "Provide a clear, descriptive, and informative blog post title (5-12 words, title case) about the main topic. Keywords: {keywords}. Content (brief): {summary}. Descriptive Title:",
        "num_beams": 1, "do_sample": True, "temperature": 0.7, "top_p": 0.88, "no_repeat_ngram_size": 2
    },
    {
        "name": "Intrigue/Curiosity",
        "template": "Create a blog post title (5-12 words, title case) that sparks curiosity and makes readers want to know more. Keywords: {keywords}. Content (brief): {summary}. Intriguing Title:",
        "num_beams": 1, "do_sample": True, "temperature": 0.9, "top_p": 0.95, "no_repeat_ngram_size": 2 
    }
]

def generate_title_candidates(content: str, total_candidates_target: int = 15) -> list[str]:
    summary = extractor.short_summary(content)
    keywords = extractor.extract_keywords(content, top_n=5)
    kw_str = ", ".join(keywords)

    if not kw_str: 
        kw_str = summary.split()[:3] if summary else "key topics"
        kw_str = ", ".join(kw_str)
        
    all_titles = []
    
    num_strategies_to_use = min(len(PROMPT_STRATEGIES), 3) 
    selected_strategies = random.sample(PROMPT_STRATEGIES, num_strategies_to_use)
    
    candidates_per_strategy = total_candidates_target // num_strategies_to_use
    if candidates_per_strategy == 0: candidates_per_strategy = 1 

    for strategy in selected_strategies:
        prompt_text = strategy["template"].format(keywords=kw_str, summary=summary)
        
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=512, 
        ).to(DEVICE)

        gen_params = {
            "max_new_tokens": 20, 
            "num_beams": strategy.get("num_beams", 1),
            "do_sample": strategy.get("do_sample", True),
            "top_p": strategy.get("top_p", 0.90),
            "temperature": strategy.get("temperature", 0.8),
            "num_return_sequences": candidates_per_strategy,
            "no_repeat_ngram_size": strategy.get("no_repeat_ngram_size", 2),
        }
        if gen_params["do_sample"] and gen_params["num_beams"] == 1:
            del gen_params["num_beams"]

        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_params)
            
            for seq in output_ids:
                decoded = tokenizer.decode(seq, skip_special_tokens=True)
                first_line = decoded.splitlines()[0].strip() if decoded else ""
                cleaned = clean_title(first_line)
                word_count = len(cleaned.split())
                if 5 <= word_count <= 15: 
                    all_titles.append(cleaned)
        except Exception as e:
            print(f"Error during title generation for strategy {strategy['name']}: {e}") 
            continue 

    seen = set()
    unique_titles = []
    for t in all_titles:
        key = t.lower().strip()
        if key and key not in seen: 
            seen.add(key)
            unique_titles.append(t)
            
    return unique_titles


def score_and_select_titles(content: str, candidates: list[str], top_n: int = 3, diversity_threshold: float = 0.3) -> list[str]:
    if not candidates:
        kw = extractor.extract_keywords(content, top_n=2)
        if len(kw) >= 2:
            return [clean_title(f"{kw[0].capitalize()} & {kw[1].capitalize()} Guide")]
        elif kw:
            return [clean_title(f"Insights on {kw[0].capitalize()}")]
        return ["Exploring Key Topics"]


    summary_for_scoring = extractor.short_summary(content, max_words=100) 
    keywords = extractor.extract_keywords(content, top_n=7) 

    scored_titles = []
    for t in candidates:
        score = 0.0
        title_lower = t.lower()
        
        for kw in keywords:
            if kw.lower() in title_lower:
                score += 1.5
            if title_lower.startswith(kw.lower()):
                score += 0.75
        
        length = len(t.split())
        if 5 <= length <= 15: 
            score += max(0, 2.0 - 0.5 * abs(8 - length)) 
        else: 
            score -= 5 


        summary_overlap_penalty_factor = 2.0
        sim_with_summary = jaccard_similarity(t, summary_for_scoring)
        score -= summary_overlap_penalty_factor * sim_with_summary
        
        if "?" in t:
            score += 0.5 
        if any(char.isdigit() for char in t): 
             score += 0.3

        generic_starters = ["a guide to", "an introduction to", "exploring the", "the ultimate guide"]
        for gs in generic_starters:
            if title_lower.startswith(gs):
                score -= 0.5
                
        scored_titles.append({"title": t, "score": score})

    sorted_by_score = sorted(scored_titles, key=lambda x: x["score"], reverse=True)

    final_selection = []
    if not sorted_by_score: 
        return extractor.extract_keywords(content, top_n=1)[0].capitalize() + " Overview" if extractor.extract_keywords(content, top_n=1) else ["Read This Post"]


    for candidate_dict in sorted_by_score:
        if len(final_selection) >= top_n:
            break
        
        candidate_title = candidate_dict["title"]
        is_diverse_enough = True
        for selected_title in final_selection:
            if jaccard_similarity(candidate_title, selected_title) > (1.0 - diversity_threshold): 
                is_diverse_enough = False
                break
        
        if is_diverse_enough:
            final_selection.append(candidate_title)
            

    if len(final_selection) < top_n:
        additional_needed = top_n - len(final_selection)
        current_final_titles_set = set(final_selection)
        for candidate_dict in sorted_by_score:
            if additional_needed == 0:
                break
            if candidate_dict["title"] not in current_final_titles_set:
                final_selection.append(candidate_dict["title"])
                additional_needed -= 1
    
    if not final_selection:
        kw = extractor.extract_keywords(content, top_n=1)
        default_title = f"{kw[0].capitalize()} Explained" if kw else "A Closer Look at This Topic"
        return [default_title] * top_n 

    return final_selection[:top_n]


def generate_three_titles(content: str) -> list[str]:
    candidates = generate_title_candidates(content, total_candidates_target=20) 
    
    if not candidates:
        kw = extractor.extract_keywords(content, top_n=2)
        if len(kw) >= 2:
            return [clean_title(f"Key Takeaways on {kw[0].capitalize()} and {kw[1].capitalize()}")]
        elif kw:
            return [clean_title(f"Understanding {kw[0].capitalize()}")]
        return ["Insights from the Blog Post"]
        
    return score_and_select_titles(content, candidates, top_n=3, diversity_threshold=0.35) 


@csrf_exempt 
@require_POST 
def title_suggestions(request):
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400) 

    content = payload.get("content", "").strip()
    if not content:
        return JsonResponse({"error": "'content' field is required."}, status=400)
    
    if len(content) < 50: 
        return JsonResponse({"error": "Content is too short to generate meaningful titles. Please provide at least 50 characters."}, status=400)


    try:
        with torch.no_grad():
            titles = generate_three_titles(content)
        return JsonResponse({"titles": titles})
    except Exception as e:
        print(f"Title generation error: {e}") 
        return JsonResponse({"error": f"An unexpected error occurred during title generation."}, status=500)

