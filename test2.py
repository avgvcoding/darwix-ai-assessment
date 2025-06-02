#!/usr/bin/env python3

import json
import requests
import os
import sys

"""
Simulate:
  curl -X POST "http://127.0.0.1:8000/api/title_suggestions/" \
       -H "Content-Type: application/json" \
       -d "{\"content\": \"<blog text>\"}"

Usage:
  python curl_sim_feature2.py
Or:
  python curl_sim_feature2.py "Your **short** blog content here..."
"""

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/api/title_suggestions/"

def main():
    if len(sys.argv) > 1:
        blog_content = sys.argv[1]
    else:
        blog_content = (
            "In this example, we explore how to integrate AI models into a Django web application "
            "using only open-source libraries. You’ll learn how to load a transformer-based title generator, "
            "preprocess your blog content, extract keywords and summary, and produce three suggested titles "
            "without relying on any paid APIs. By the end, you’ll have a lightweight pipeline capable "
            "of generating SEO-friendly headlines in real time."
        )

    url = BASE_URL + ENDPOINT
    headers = {"Content-Type": "application/json"}
    payload = {"content": blog_content}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(json.dumps({"status":"error","message":f"Request failed: {e}"}))
        return

    try:
        data = resp.json()
    except ValueError:
        print(json.dumps({"status":"error","message":"Response is not valid JSON","raw_response":resp.text}))
        return

    print(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
