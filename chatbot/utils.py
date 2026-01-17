import string
from .chatbot_data import TAGS

def normalize(text):
    text = text.lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))

def detect_topic(user_input):
    text = user_input.lower()
    for topic, keywords in TAGS.items():
        if any(kw in text for kw in keywords):
            return topic
    return None

def log_unmatched_query(user_input):
    try:
        with open("unmatched_queries.log", "a") as f:
            f.write(user_input + "\n")
    except Exception as e:
        print("Failed to log unmatched query:", e)

def score_tag(query, nlp):
    query_doc = nlp(query.lower())
    scores = {}
    for tag, keywords in TAGS.items():
        tag_doc = nlp(" ".join(keywords))
        scores[tag] = query_doc.similarity(tag_doc)
    
    # Return the tag with the highest score if it is above a certain threshold, e.g. 0.5
    best_match_tag = max(scores, key=scores.get) if scores else None
    if best_match_tag and scores[best_match_tag] > 0.5:
        return best_match_tag
    return None