
import spacy
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .chatbot_data import NORMALIZED_INPUTS, TAGS
from .models import Question
import string

# Load spacy model once.
# 'en_core_web_md' is a medium-sized model with word vectors.
nlp = spacy.load("en_core_web_md")

# Keyword tagging

# Topic buttons — moved outside for global access
topic_buttons = {
    "acne": [
        {"text": "What is acne?", "payload": "what is acne"},
        {"text": "Acne causes", "payload": "acne causes"},
        {"text": "Acne treatments", "payload": "acne remedies"},
        {"text": "Types of acne", "payload": "acne types"}
    ],
    "blackheads": [
        {"text": "What are blackheads?", "payload": "what are blackheads"},
        {"text": "Blackhead removal", "payload": "blackhead remedies"},
        {"text": "Blackheads vs whiteheads", "payload": "blackhead vs whitehead"}
    ],
    "freckles": [
        {"text": "What are freckles?", "payload": "what are freckles"},
        {"text": "Freckle causes", "payload": "freckles causes"},
        {"text": "Lighten freckles", "payload": "freckles remedies"}
    ],
    "eye_bags": [
        {"text": "What are eye bags?", "payload": "what are eye bags"},
        {"text": "Reduce puffiness", "payload": "eye bags remedies"},
        {"text": "Cold spoon remedy", "payload": "chilled spoons or cold compress"}
    ],
    "dark_circles": [
        {"text": "Dark circle causes", "payload": "dark circles causes"},
        {"text": "Under eye treatments", "payload": "dark circles remedies"},
        {"text": "Cucumber remedy", "payload": "chilled cucumber slices"}
    ],
    "oily_skin": [
        {"text": "Oily skin care", "payload": "what is oily skin"},
        {"text": "Products for oily skin", "payload": "using non-comedogenic products"}
    ],
    "dry_skin": [
        {"text": "Dry skin solutions", "payload": "what is dry skin"},
        {"text": "Best moisturizers", "payload": "wrinkles remedies"}
    ],
    "general_skincare": [
        {"text": "Skincare basics", "payload": "skincare_basics"},
        {"text": "Daily routine", "payload": "washing face tips"},
        {"text": "Product differences", "payload": "skincare_products_difference"}
    ],
    "conversation": [
        {"text": "Start over", "payload": "greeting"},
        {"text": "What can you do?", "payload": "how_it_works"},
        {"text": "Contact support", "payload": "human_help"}
    ],
    "website": [
        {"text": "How it works", "payload": "how_it_works"},
        {"text": "Skin diagnosis", "payload": "skin_diagnosis_info"},
        {"text": "Meet the team", "payload": "developer_info"}
    ],
    "default": [
        {"text": "Common Skin Issues", "payload": "common concerns"},
        {"text": "Skincare Basics", "payload": "skincare_basics"},
        {"text": "Ask a Question", "payload": "general_help"}
    ]
}


# In-memory cache
cached_questions = []
cached_spacy_docs = []

def normalize(text):
    return text.lower().strip().translate(str.maketrans('', '', string.punctuation))

def preload_questions():
    """
    Loads questions from the database and caches them along with their
    spaCy Doc objects for efficient similarity lookups.
    """
    global cached_questions, cached_spacy_docs
    cached_questions = list(Question.objects.select_related("answer").all())
    question_texts = [q.text for q in cached_questions]
    cached_spacy_docs = [nlp(text) for text in question_texts]

# Run once at startup
def ensure_loaded():
    if not cached_questions:
        preload_questions()

def get_best_match_spacy(user_input, questions, spacy_docs, threshold=0.7):
    """
    Finds the best matching question using spaCy's semantic similarity.
    """
    user_doc = nlp(user_input)
    best_match = None
    best_score = 0.0

    # Ensure the user input has a vector to compare against
    if not user_doc.has_vector:
        return None

    for i, question_doc in enumerate(spacy_docs):
        # Ensure the question has a vector and isn't just whitespace
        if question_doc.has_vector and question_doc.text.strip():
            score = user_doc.similarity(question_doc)
            if score > best_score:
                best_score = score
                best_match = questions[i]
    
    return best_match if best_score >= threshold else None

def detect_topic(user_input):
    """
    Detects a topic based on keywords in the user input.
    """
    text = user_input.lower()
    for topic, keywords in TAGS.items():
        if any(kw in text for kw in keywords):
            return topic
    return None

def log_unmatched_query(user_input):
    """
    Logs unmatched queries to a file for analysis.
    """
    try:
        with open("unmatched_queries.log", "a") as f:
            f.write(user_input + "\n")
    except Exception as e:
        print("Failed to log unmatched query:", e)

def score_tag(user_input):
    text = user_input.lower()
    scoring_rules = {
        # ROSACEA RULES
        "rosacea vs acne": [
            lambda q: 2 if "rosacea" in q and "acne" in q else 0,
            lambda q: 2 if any(w in q for w in ["vs", "difference", "compare"]) else 0,
        ],
        "rosacea remedies": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["remedy", "treat", "cure", "solution", "manage"]) else 0,
        ],
        "rosacea causes": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["cause", "why", "reason", "origin"]) else 0,
        ],
        "rosacea symptoms": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["symptom", "sign", "look like"]) else 0,
        ],
        "rosacea triggers": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["trigger", "flare", "flare-up"]) else 0,
        ],
        "rosacea diet": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["food", "diet", "eat", "avoid"]) else 0,
        ],
        "rosacea prevention": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["prevent", "avoid", "sun protection"]) else 0,
        ],
        "rosacea myths": [
            lambda q: 2 if "rosacea" in q and any(w in q for w in ["myth", "misconception", "hygiene"]) else 0,
        ],
        "what is rosacea": [
            lambda q: 1 if "rosacea" in q else 0,
        ],


        # BLACKHEAD RULES
             # BLACKHEAD RULES
        "what causes blackheads": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in [
                "cause", "causes", "reason", "reasons", "origin", "leads to", "trigger", "form", "develop", "formation", "why", "appear", "keep coming"
            ]) else 0,
            lambda q: 1 if "blackhead" in q or "blackheads" in q else 0,
        ],

        "blackhead remedies": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["remedy", "remedies", "treat", "treatment", "cure", "solution", "manage"]) else 0,
        ],

        "blackhead symptoms": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["symptom", "symptoms", "sign", "signs", "look like", "identify", "detect", "appearance"]) else 0,
        ],

        "blackhead prevention": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["prevent", "prevention", "stop", "avoid", "block"]) else 0,
        ],

        "blackhead removal": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["remove", "removal", "extract", "get rid", "clean", "tool", "pore strip", "pop", "squeeze"]) else 0,
        ],

        "blackhead professional treatments": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["dermatologist", "professional", "treatment", "facial", "chemical peel", "microdermabrasion"]) else 0,
        ],

        "blackhead myths": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["myth", "myths", "misconception", "misconceptions", "dirt", "scrub", "hygiene"]) else 0,
        ],

        "blackheads and acne": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and "acne" in q and any(w in q for w in ["difference", "relation", "vs", "link", "compare"]) else 0,
        ],

        "blackhead areas": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["nose", "chin", "forehead", "back", "chest", "shoulder", "t-zone", "face"]) else 0,
        ],

        "blackheads on skin types": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["oily", "dry", "combination", "sensitive", "skin type", "skin"]) else 0,
        ],

        "blackhead products": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["product", "products", "ingredient", "ingredients", "cleanser", "mask", "salicylic", "retinoid", "niacinamide", "benzoyl peroxide"]) else 0,
        ],

        "blackhead diet": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["diet", "food", "dairy", "sugar", "eat", "nutrition"]) else 0,
        ],

        "blackheads by age": [
            lambda q: 2 if ("blackhead" in q or "blackheads" in q) and any(w in q for w in ["teen", "adult", "age", "older", "younger", "puberty"]) else 0,
        ],

        "what are blackheads": [
            lambda q: 1 if "blackhead" in q or "blackheads" in q else 0,
        ],

        
        # ACNE

    "acne causes": [
        lambda q: 2 if "acne" in q and any(w in q for w in ["cause", "why", "reason", "origin", "trigger"]) else 0,
        lambda q: 2 if any(w in q for w in [
            "excess oil", "clogged pores", "dead skin cells", "bacteria", "Cutibacterium",
            "hormones", "puberty", "period", "PCOS", "genetics", "medications", "steroids",
            "lithium", "stress", "diet", "dairy", "sugar", "greasy foods", "smoking",
            "lack of sleep", "sweating", "makeup", "skincare", "poor hygiene", "pollution",
            "environmental", "dirty pillowcase"
        ]) and "acne" in q else 0
    ],

    "acne symptoms": [
        lambda q: 2 if "acne" in q and any(w in q for w in ["symptom", "sign", "look like", "identify", "how to know", "do i have"]) else 0
    ],

    "acne remedies": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "remedy", "remedies", "treat", "get rid", "natural", "home", "cure", "solution",
            "fix", "manage", "removal"
        ]) else 0
    ],

    "acne and diet": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "diet", "food", "eat", "avoid", "trigger", "prevent", "reduce", "worsen", "best foods", "what to eat"
        ]) else 0
    ],

    "acne treats": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "prescription", "oral antibiotics", "doxycycline", "clindamycin", "isotretinoin", "accutane",
            "hormonal therapy", "birth control", "spironolactone", "chemical peels", "laser", "light therapy", "cortisone"
        ]) else 0
    ],

    "acne skincare": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "skincare", "cleanser", "moisturizer", "routine", "non-comedogenic", "products", "ingredients", "sunscreen"
        ]) else 0
    ],

    "acne prevention": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "prevent", "avoid", "trigger", "dirty pillowcase", "maskne", "hygiene", "stop", "lifestyle"
        ]) else 0
    ],

    "acne myths": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "myth", "misconception", "true", "false", "scrubbing", "dirt", "pop pimples", "teenagers only"
        ]) else 0
    ],

    "acne psychology": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "self-esteem", "social anxiety", "depression", "emotional impact", "bullying", "stigma"
        ]) else 0
    ],

    "acne by age": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "teenage", "adult", "baby", "pregnancy", "men", "women", "age"
        ]) else 0
    ],

    "acne location": [
        lambda q: 2 if any(w in q for w in [
            "forehead acne", "chin acne", "nose acne", "cheek acne", "back acne", "chest acne", 
            "shoulder acne", "scalp acne", "butt acne", "acne on"
        ]) else 0
    ],

    "acne scars": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "scar", "scars", "ice pick", "boxcar", "rolling", "post-inflammatory", 
            "hyperpigmentation", "red spots", "acne marks", "microneedling", "laser", "treat scars"
        ]) else 0
    ],

    "acne and condition": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "PCOS", "hormonal imbalance", "gut health", "fungal acne", "bacterial acne"
        ]) else 0
    ],

    "acne triggers": [
        lambda q: 2 if "acne" in q and any(w in q for w in [
            "trigger", "flare", "flare-up", "hormones", "stress", "food", "humidity", 
            "sweat", "medications", "hair products", "skincare products"
        ]) else 0
    ],

    "what is acne": [
        lambda q: 2 if "acne" in q and any(w in q for w in ["what", "define", "meaning", "about", "info", "explain", "summary", "overview", "description"]) else 0,
        lambda q: 1 if "acne" in q else 0
    ],
    # "acne vs rosacea": [
    #     lambda q: 2 if "acne" in q and "rosacea" in q else 0,
    #     lambda q: 2 if any(w in q for w in ["vs", "difference", "compare"]) else 0
    # ]
    
    # DARK CIRCLES QUERY TAGGING

    "what are dark circles": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "what", "define", "meaning", "explain", "about", "description", "overview"
        ]) else 0,
        lambda q: 1 if "dark circles" in q else 0,
    ],

    "dark circles causes": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "cause", "causes", "why", "reason", "reasons", "origin", "due to", "from", "lead to"
        ]) else 0,
    ],

    "dark circles symptoms": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "symptom", "sign", "appearance", "look like", "indicate"
        ]) else 0,
    ],

    "dark circles treatment": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "treatment", "how to treat", "treat", "get rid", "fix", "remove", "reduce", "fade", "solution", "healing"
        ]) else 0,
    ],

    "dark circles remedies": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "remedy", "remedies", "home remedy", "natural remedy", "cure", "DIY", "natural ways", "how to remove"
        ]) else 0,
    ],

    "dark circles prevention": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "prevent", "avoid", "stop", "reduce risk", "how to avoid", "preventing", "lifestyle"
        ]) else 0,
    ],

    "dark circles skincare": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "skincare", "eye cream", "serum", "routine", "eye mask", "ingredients", "products", "best for", "topical"
        ]) else 0,
    ],

    "dark circles by age": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "child", "children", "teen", "adolescent", "adult", "elderly", "age", "aging", "older", "young"
        ]) else 0,
    ],

    "dark circles and sleep": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "sleep", "lack of sleep", "rest", "insomnia", "sleep deprived", "sleep schedule"
        ]) else 0,
    ],

    "dark circles and diet": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "diet", "food", "eat", "nutrition", "vitamin", "iron", "deficiency", "hydration", "water intake", "foods for"
        ]) else 0,
    ],

    "dark circles and health": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "health issue", "underlying", "allergy", "dehydration", "anemia", "sinus", "hereditary", "genetics"
        ]) else 0,
    ],

    "dark circles myths": [
        lambda q: 2 if "dark circles" in q and any(w in q for w in [
            "myth", "misconception", "false", "truth", "common myths"
        ]) else 0,
    ],
    
    "eye bags vs dark circles": [
        lambda q: 2 if "eye bags" in q and "dark circles" in q else 0,
        lambda q: 2 if any(w in q for w in ["vs", "difference", "compare", "distinguish"]) else 0,
    ],

    "eye bags causes": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["cause", "reason", "why", "origin", "lead", "makes", "get", "appear"]) else 0,
        lambda q: 1 if "puffy eyes" in q and any(w in q for w in ["cause", "reason", "why"]) else 0,
    ],

    "eye bags symptoms": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["symptom", "sign", "look like", "identify", "know", "have"]) else 0,
        lambda q: 1 if "puffy eyes" in q and "symptom" in q else 0,
    ],

    "eye bags remedies": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["remedy", "treat", "cure", "fix", "get rid", "solution", "manage", "remove"]) else 0,
        lambda q: 1 if "puffy eyes" in q and any(w in q for w in ["reduce", "treat"]) else 0,
    ],

    "eye bags types": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["types", "kinds", "categories", "classification", "forms"]) else 0,
    ],

    "eyebags vs hollowness": [
        lambda q: 2 if "eye bags" in q and "hollowness" in q else 0,
        lambda q: 2 if any(w in q for w in ["difference", "vs", "compare"]) else 0,
    ],

    "eyebags skincare": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["skincare", "eye cream", "routine", "apply", "serum", "retinol", "hyaluronic", "massage"]) else 0,
    ],

    "eyebags prevention": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["prevent", "avoid", "stop", "tips", "how to stop", "how to avoid"]) else 0,
    ],

    "eyebags and aging": [
        lambda q: 2 if "eye bags" in q and any(w in q for w in ["aging", "age", "older", "get worse with age", "thinning", "fat"]) else 0,
    ],

    "what are eye bags": [
        lambda q: 1 if "eye bags" in q and any(w in q for w in ["what", "define", "explain", "meaning", "info", "understand"]) else 0,
    ],
    
    "freckles types": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["types", "kinds", "categories", "classify", "variants", "variety", "different"]) else 0,
        lambda q: 1 if "freckles" in q else 0,
    ],

    "freckles skincare": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["skincare", "skin care", "routine", "sunscreen", "products", "cream", "serum", "moisturizer"]) else 0,
    ],

    "freckles and sun": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["sun", "sunlight", "uv", "tanning", "summer", "sun exposure", "darken", "sunscreen"]) else 0,
    ],

    "freckles treatments": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["remove", "removal", "treatment", "laser", "peel", "fade", "clinic", "dermatologist", "solution", "cure"]) else 0,
    ],

    "freckles and genetics": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["genetic", "heredity", "inherited", "dna", "genes", "family", "ethnicity"]) else 0,
    ],

    "freckles prevention": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["prevent", "prevention", "avoid", "stop", "reduce", "sunblock", "sunscreen", "tips"]) else 0,
    ],

    "freckles vs others": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["vs", "difference", "compare", "distinguish", "apart"]) else 0,
    ],

    "freckles and makeup": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["makeup", "foundation", "concealer", "cover", "enhance", "routine"]) else 0,
    ],

    "freckles causes": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["cause", "causes", "why", "reason", "origin", "trigger", "make", "form", "factors"]) else 0,
    ],

    "freckles symptoms": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["symptom", "sign", "look like", "appearance", "identify", "recognize"]) else 0,
    ],

    "freckles remedies": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["remedy", "treat", "cure", "solution", "fix", "fade", "lighten", "home", "natural"]) else 0,
    ],

    "what are freckles": [
        lambda q: 1 if "freckles" in q and any(w in q for w in ["what", "define", "explain", "meaning", "tell", "info", "description"]) else 0,
    ],
    

    "pigmentation causes": [
        lambda q: 2 if any(w in q for w in ["cause", "causes", "reason", "reasons", "why do i get", "why do i have", "origin", "genesis", "lead to", "pigmentation causes", "dark spots causes", "hyperpigmentation causes"]) and "pigmentation" in q else 0,
    ],

    "body pigmentation": [
        lambda q: 2 if any(w in q for w in ["dark spots on hands", "dark spots on arms", "dark spots on body", "pigmentation on body"]) else 0,
    ],

    "pigmentation symptoms": [
        lambda q: 2 if any(w in q for w in ["symptoms", "signs", "look like", "how to identify", "how to know", "recognize", "identify"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "pigmentation remedies": [
        lambda q: 2 if any(w in q for w in ["remedy", "treat", "treatment", "get rid of", "home remedy", "natural remedy", "cure", "fix", "solution", "remove", "fade", "manage", "lighten"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "pigmentation types": [
        lambda q: 2 if any(w in q for w in ["types", "kinds", "different types", "classify", "categories", "variety"]) and "pigmentation" in q else 0,
    ],

    "dark spots vs freckles": [
        lambda q: 2 if "freckles" in q and any(w in q for w in ["difference", "vs", "compare"]) and "dark spots" in q else 0,
    ],

    "darkspots vs melasma": [
        lambda q: 2 if "melasma" in q and any(w in q for w in ["difference", "vs", "compare"]) and "dark spots" in q else 0,
    ],

    "darkspots vs age spots": [
        lambda q: 2 if "age spots" in q and any(w in q for w in ["difference", "vs", "compare"]) and "dark spots" in q else 0,
    ],

    "darkspots vs sun spots": [
        lambda q: 2 if "sunspots" in q and any(w in q for w in ["difference", "vs", "compare"]) and "dark spots" in q else 0,
    ],

    "darkspots vs acne scars": [
        lambda q: 2 if "acne scars" in q and any(w in q for w in ["difference", "vs", "compare"]) and "dark spots" in q else 0,
    ],

    "sun exposure and dark spots": [
        lambda q: 2 if any(w in q for w in ["sun", "sunlight", "uv rays", "tanning", "sun exposure"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "prevent dark spots": [
        lambda q: 2 if any(w in q for w in ["prevent", "preventing", "stop", "avoid"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "sunscreen for pigmentation prevention": [
        lambda q: 2 if any(w in q for w in ["sunscreen", "sunblock", "sun protection", "spf"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "lifestyle tips dark spots": [
        lambda q: 2 if any(w in q for w in ["lifestyle", "daily habits", "tips"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "skincare routine pigmentation": [
        lambda q: 2 if any(w in q for w in ["skincare", "routine", "products", "best skincare"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "diet and pigmentation": [
        lambda q: 2 if any(w in q for w in ["diet", "foods", "nutrition", "food"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "avoiding pigmentation triggers": [
        lambda q: 2 if any(w in q for w in ["trigger", "triggers", "avoid"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "serious dark spot signs": [
        lambda q: 2 if any(w in q for w in ["serious", "warning", "red flag", "signs", "symptoms"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "birthmarks and pigmentation": [
        lambda q: 2 if any(w in q for w in ["birthmarks"]) and any(x in q for x in ["pigmentation", "dark spots"]) else 0,
    ],

    "dark spots makeup solutions": [
        lambda q: 2 if any(w in q for w in ["makeup", "conceal", "cover", "foundation", "cosmetics"]) and any(x in q for x in ["dark spots", "pigmentation"]) else 0,
    ],

    "what is pigmentation": [
        lambda q: 1 if any(w in q for w in ["what is pigmentation", "define pigmentation", "explain pigmentation", "meaning of pigmentation", "pigmentation meaning", "info on pigmentation", "what does pigmentation mean", "skin pigmentation", "what are dark spots", "define dark spots", "explain dark spots", "dark spots meaning", "skin spots meaning"]) else 0,
    ],

    "wrinkles causes": [
        lambda q: 2 if any(w in q for w in [
            "cause", "causes", "reason", "reasons", "why do i get", "why do i have",
            "what causes wrinkles", "origin", "genesis", "lead to", "wrinkles causes",
            "wrinkle causes", "how do wrinkles form", "what makes wrinkles appear",
            "aging causes wrinkles", "sun causes wrinkles", "smoking and wrinkles",
            "pollution and wrinkles", "hydration and wrinkles", "facial expressions cause wrinkles",
            "stress and wrinkles", "gravity cause wrinkles"
        ]) and "wrinkle" in q else 0,
    ],

    "wrinkles symptoms": [
        lambda q: 2 if any(w in q for w in [
            "symptoms", "signs", "look like", "how to identify", "how to know", "recognize",
            "identify", "wrinkle signs", "what do wrinkles look like", "how to tell if i have wrinkles"
        ]) and "wrinkle" in q else 0,
    ],

    "wrinkles remedies": [
        lambda q: 2 if any(w in q for w in [
            "remedy", "remedies", "treat", "treatment", "get rid of", "home remedy", "natural remedy",
            "cure", "fix", "solution", "remove", "fade", "manage", "lighten", "wrinkle removal",
            "wrinkle treatment", "how to make wrinkles go away", "what helps wrinkles", "wrinkle solutions"
        ]) and "wrinkle" in q else 0,
    ],

    "wrinkles types": [
        lambda q: 2 if any(w in q for w in [
            "types", "kinds", "different types", "classify", "categories", "variety", "classification"
        ]) and "wrinkle" in q else 0,
    ],

    # Subcategories for causes (optional fine-grained scoring)

    "wrinkles causes aging": [
        lambda q: 2 if any(w in q for w in ["aging", "natural aging", "age-related"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes sun uv": [
        lambda q: 2 if any(w in q for w in ["sun", "uv", "sun exposure", "uv rays", "sunlight", "tanning"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes smoking": [
        lambda q: 2 if any(w in q for w in ["smoking", "cigarette", "tobacco"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes pollution": [
        lambda q: 2 if any(w in q for w in ["pollution", "environmental pollution", "air pollution"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes hydration nutrition": [
        lambda q: 2 if any(w in q for w in ["hydration", "dehydration", "nutrition", "diet"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes facial expressions": [
        lambda q: 2 if any(w in q for w in ["facial expressions", "frowning", "smiling", "expression lines"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes stress lifestyle": [
        lambda q: 2 if any(w in q for w in ["stress", "lifestyle", "unhealthy lifestyle"]) and "wrinkle" in q else 0,
    ],

    "wrinkles causes gravity sagging": [
        lambda q: 2 if any(w in q for w in ["gravity", "skin sagging", "sagging skin"]) and "wrinkle" in q else 0,
    ],

    # Prevention and treatment

    "wrinkles prevention sunscreen uv": [
        lambda q: 2 if any(w in q for w in ["sunscreen", "sunblock", "sun protection", "spf"]) and "wrinkle" in q else 0,
    ],

    "wrinkles prevention moisturizers hydration": [
        lambda q: 2 if any(w in q for w in ["moisturizer", "hydration", "moisturizers"]) and "wrinkle" in q else 0,
    ],

    "wrinkles prevention lifestyle changes": [
        lambda q: 2 if any(w in q for w in ["lifestyle", "diet", "sleep", "hydration", "healthy lifestyle", "drinking water"]) and "wrinkle" in q else 0,
    ],

    "what are wrinkles": [
        lambda q: 1 if any(w in q for w in [
            "what are wrinkles", "define wrinkles", "explain wrinkles", "meaning of wrinkles",
            "wrinkles meaning", "info on wrinkles", "what does wrinkles mean", "skin wrinkles",
            "what's a wrinkle", "what is a wrinkle", "describe wrinkles", "basic info on wrinkles",
            "about skin wrinkles", "tell me about wrinkles", "what exactly are wrinkles",
            "can you explain what wrinkles are"
        ]) else 0,
    ],
    
    "skin cancer causes": [
        lambda q: 2 if any(w in q for w in [
            "cause", "causes", "reason", "reasons", "why do i get", "why do i have", "origin", "genesis", "lead to",
            "skin cancer causes", "skin cancer origin", "skin cancer reason"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer symptoms": [
        lambda q: 2 if any(w in q for w in [
            "symptoms", "signs", "look like", "how to identify", "how to know", "recognize", "identify",
            "skin cancer symptoms", "skin cancer signs"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer remedies": [
        lambda q: 2 if any(w in q for w in [
            "remedy", "treat", "treatment", "get rid of", "home remedy", "natural remedy", "cure", "fix", "solution",
            "remove", "manage", "therapy", "treatment options", "skin cancer removal"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer types": [
        lambda q: 2 if any(w in q for w in [
            "types", "kinds", "different types", "classify", "categories", "variety",
            "basal cell carcinoma", "squamous cell carcinoma", "melanoma"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer risk factors": [
        lambda q: 2 if any(w in q for w in [
            "risk factors", "risk", "susceptibility", "predisposition", "who is at risk", "increases risk",
            "uv exposure", "sunburn", "tanning beds", "fair skin"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer early detection": [
        lambda q: 2 if any(w in q for w in [
            "early detection", "detect early", "self-examination", "abcde", "mole check", "early signs", "spotting early"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer treatment options": [
        lambda q: 2 if any(w in q for w in [
            "treatment options", "surgery", "radiation", "chemotherapy", "immunotherapy", "mohs surgery",
            "cryotherapy", "photodynamic therapy", "targeted therapy"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer prevention tips": [
        lambda q: 2 if any(w in q for w in [
            "prevention", "prevent", "sun protection", "avoid tanning beds", "reduce risk", "sun safety", "how to stop"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer screening guidelines": [
        lambda q: 2 if any(w in q for w in [
            "screening guidelines", "screening frequency", "check-up schedule", "dermatologist check",
            "recommended screening", "how often to screen"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer sunscreen role": [
        lambda q: 2 if any(w in q for w in [
            "sunscreen", "sunblock", "spf", "sun protection", "does sunscreen prevent", "sunscreen benefits"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer recurrence followup": [
        lambda q: 2 if any(w in q for w in [
            "recurrence", "follow-up", "relapse", "after treatment", "monitoring", "post treatment care"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer myths facts": [
        lambda q: 2 if any(w in q for w in [
            "myths", "facts", "misconceptions", "truth", "debunking", "true or false"
        ]) and "skin cancer" in q else 0,
    ],

    "skin cancer tanning behaviour": [
        lambda q: 2 if any(w in q for w in [
            "tanning", "tanning beds", "indoor tanning", "sunbathing", "artificial tanning"
        ]) and "skin cancer" in q else 0,
    ],

    "what is skin cancer": [
        lambda q: 1 if any(w in q for w in [
            "what is skin cancer", "define skin cancer", "explain skin cancer", "meaning of skin cancer", "info about skin cancer",
            "skin malignancy", "cutaneous cancer", "skin neoplasm"
        ]) else 0,
    ],
    

    "sun spots causes": [
        lambda q: 2 if any(w in q for w in [
            "cause", "causes", "reason", "reasons", "why do i get", "why do i have", "origin", "genesis", "lead to",
            "sun spots causes", "sun spots origin", "sun spots reason"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots symptoms": [
        lambda q: 2 if any(w in q for w in [
            "symptoms", "signs", "look like", "how to identify", "how to know", "recognize", "identify",
            "sun spots symptoms", "sun spots signs"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots remedies": [
        lambda q: 2 if any(w in q for w in [
            "remedy", "treat", "treatment", "get rid of", "home remedy", "natural remedy", "cure", "fix", "solution",
            "remove", "manage", "therapy", "treatment options", "sun spots removal"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots types": [
        lambda q: 2 if any(w in q for w in [
            "types", "kinds", "different types", "classify", "categories", "variety"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots vs age spots": [
        lambda q: 2 if any(w in q for w in [
            "vs age spots", "difference between sun spots and age spots", "are sun spots age spots",
            "age spots versus sun spots"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots vs freckles": [
        lambda q: 2 if any(w in q for w in [
            "vs freckles", "difference between sun spots and freckles", "are sun spots freckles",
            "freckles vs sun spots"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots vs pigmentation": [
        lambda q: 2 if any(w in q for w in [
            "hyperpigmentation vs sun spots", "difference between hyperpigmentation and sun spots",
            "are sun spots hyperpigmentation"
        ]) and "sun spots" in q else 0,
    ],

    "dangerous sun spots": [
        lambda q: 2 if any(w in q for w in [
            "dangerous", "cancerous", "harmful", "sign of cancer", "precancerous"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots go away": [
        lambda q: 2 if any(w in q for w in [
            "go away", "disappear naturally", "fade", "natural fading"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots causes sun exposure": [
        lambda q: 2 if any(w in q for w in [
            "sun exposure", "sunlight", "uv rays"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots causes tanning": [
        lambda q: 2 if any(w in q for w in [
            "tanning", "tanning beds", "artificial tanning"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots aging connection": [
        lambda q: 2 if any(w in q for w in [
            "aging", "age spots relate", "sun spots as you age"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots skin type risks": [
        lambda q: 2 if any(w in q for w in [
            "skin type", "which skin types", "fair skin", "skin complexion"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots prevention": [
        lambda q: 2 if any(w in q for w in [
            "prevent", "prevention", "tips to avoid", "stop sun spots from forming"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots prevention sunscreen": [
        lambda q: 2 if any(w in q for w in [
            "sunscreen", "sunblock", "spf", "does sunscreen prevent", "sunscreen benefits"
        ]) and "sun spots" in q else 0,
    ],

    "sun protection tips": [
        lambda q: 2 if any(w in q for w in [
            "sun protection tips", "sun safe", "ways to protect skin"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots prevention after tanning": [
        lambda q: 2 if any(w in q for w in [
            "prevent after tanning", "avoid sun spots after sunbathing", "post-tanning sun spot prevention"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots prevention antioxidants": [
        lambda q: 2 if any(w in q for w in [
            "antioxidants", "do antioxidants prevent", "antioxidant benefits"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots skincare": [
        lambda q: 2 if any(w in q for w in [
            "skincare", "products for sun spots", "best skincare", "treat sun spots with skincare"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots makeup cover": [
        lambda q: 2 if any(w in q for w in [
            "makeup to cover", "concealing sun spots", "cover sun spots with makeup", "cosmetics for sun spots"
        ]) and "sun spots" in q else 0,
    ],

    "sun spots and skin cancer": [
        lambda q: 2 if any(w in q for w in [
            "skin cancer", "turn into skin cancer", "precancerous"
        ]) and "sun spots" in q else 0,
    ],
    
    "what are sun spots": [
        lambda q: 1 if any(w in q for w in [
            "what are sun spots", "define sun spots", "explain sun spots", "meaning of sun spots", "info about sun spots",
            "describe sun spots", "sun spots info", "skin sun spots", "pigmentation spots", "age spots"
        ]) else 0,
    ],
    


    "dry skin causes": [
        lambda q: 2 if any(w in q for w in [
            "cause", "causes", "reason", "reasons", "why do i get", "why do i have", "origin", "lead to",
            "dry skin causes", "dry skin reason", "why skin gets dry"
        ]) and "dry skin" in q else 0,
    ],

    "dry skin symptoms": [
        lambda q: 2 if any(w in q for w in [
            "symptom", "symptoms", "sign", "signs", "look like", "how to identify", "how to know", "recognize", "identify",
            "dry skin symptoms", "dry skin signs", "itching dry skin", "flaking skin", "cracking skin", "tight skin", "rough skin"
        ]) and "dry skin" in q else 0,
    ],
    "dry skin types": [
        lambda q: 2 if any(w in q for w in [
            "types", "kinds", "different types", "classify", "categories", "variety",
            "xerosis", "asteatotic eczema", "dry skin conditions", "forms of dry skin"
        ]) and "dry skin" in q else 0,
    ],
    "dry skin moisturizers": [
        lambda q: 2 if any(w in q for w in [
            "moisturizer", "emollient", "cream for dry skin", "best moisturizer", "hydrating cream", "dry skin lotion",
            "dry skin cream", "ointments for dry skin", "moisturizing products"
        ]) and "dry skin" in q else 0,
    ],
    "dry skin diet": [
        lambda q: 2 if any(w in q for w in [
            "diet", "food", "nutrition", "foods for dry skin", "what to eat", "avoid for dry skin",
            "hydrating foods", "nutrients for skin hydration"
        ]) and "dry skin" in q else 0,
    ],
    "avoiding dry skin tips": [
        lambda q: 2 if any(w in q for w in [
            "avoid getting dry skin", "avoiding dry skin", "how to avoid dry skin", "prevent dry skin tips",
            "bathing habits to prevent dry skin", "how can bathing habits", "bathing routine dry skin",
            "what role does clothing", "clothing impact dry skin", "dressing for dry skin",
            "environmental adjustments", "environment changes to prevent dry skin", "adjustments for dry skin",
            "daily routines", "routines to prevent dry skin", "what daily routines can help", "habits for dry skin prevention"
        ]) and "dry skin" in q else 0,
    ],
    "what is dry skin": [
        lambda q: 1 if any(w in q for w in [
            "what is dry skin", "define dry skin", "explain dry skin", "meaning of dry skin", "info about dry skin",
            "describe dry skin", "dry skin info", "skin dryness", "dryness of skin"
        ]) else 0,
    ],
    

    "oily skin causes": [
        lambda q: 2 if any(w in q for w in [
            "oily skin causes","why is my skin oily all the time","what causes excess oil production",
            "can diet affect oily skin","does stress make oily skin worse","causes of oily skin",
            "reasons for oily skin","why skin is oily","factors oily skin","excess oil production cause"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin symptoms": [
        lambda q: 2 if any(w in q for w in [
            "oily skin symptoms","how to tell if i have oily skin","what are the common signs of oily skin",
            "why do i get oily skin mostly on my forehead and nose","signs of oily skin",
            "oily skin signs","how to identify oily skin","oily forehead",
            "oily nose","common signs of oily skin"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin related problems": [
        lambda q: 2 if any(w in q for w in [
            "oily skin related problems", "why do i get acne with oily skin","does oily skin cause blackheads or whiteheads",
            "can oily skin lead to enlarged pores","why is my oily skin shiny all day",
            "oily skin acne","blackheads oily skin","whiteheads oily skin","enlarged pores oily skin",
            "shiny oily skin","skin problems oily skin","acne and oily skin"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin diagnosis": [
        lambda q: 2 if any(w in q for w in [
            "oily skin diagnosis","how can i check if my skin is oily or combination","can you help diagnose if my skin is oily",
            "what skin type do i have if it’s oily and prone to breakouts","diagnose oily skin",
            "check oily skin","skin type oily","oily combination skin","identify oily skin","skin type identification"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin home remedies": [
        lambda q: 2 if any(w in q for w in [
            "oily skin home remedies","what are some natural remedies to control oily skin","how to reduce oiliness using kitchen ingredients",
            "can i use lemon or honey for oily skin","are there any homemade face masks for oily skin",
            "natural remedies oily skin","homemade remedies oily skin","kitchen ingredients oily skin",
            "lemon for oily skin","honey for oily skin","face masks for oily skin","control oiliness naturally"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin routine": [
        lambda q: 2 if any(w in q for w in [
            "oily skin routine","what’s the best daily routine for oily skin","which ingredients should i avoid if i have oily skin",
            "how often should i wash my face if my skin is oily","can moisturizer help oily skin",
            "skincare routine oily skin","daily routine oily skin",
            "ingredients to avoid oily skin","washing face oily skin","moisturizer for oily skin",
            "best routine oily skin"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin lifestyle diet": [
        lambda q: 2 if any(w in q for w in [
            "oily skin lifestyle diet","does drinking water help oily skin","what foods should i avoid to reduce oily skin",
            "can exercise affect oil production on my skin","lifestyle tips oily skin","diet tips oily skin",
            "water for oily skin","foods to avoid oily skin","exercise and oily skin","oil production lifestyle"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin mistakes": [
        lambda q: 2 if any(w in q for w in [
            "oily skin mistakes", "what skincare mistakes worsen oily skin", "can over-washing make oily skin worse",
            "is using oily skin products bad for oily skin","common mistakes oily skin","skincare mistakes oily skin",
            "over-washing oily skin", "bad products for oily skin", "worsen oily skin"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin treatments products": [
        lambda q: 2 if any(w in q for w in [
            "oily skin treatments products","what kind of products work best for oily skin","are oil-free moisturizers good for oily skin",
            "can natural oils help oily skin","should i use toner for oily skin",
            "oily skin treatments","best products for oily skin","oil-free moisturizers oily skin",
            "natural oils for oily skin", "toner for oily skin", "products for oily skin"
        ]) and "oily skin" in q else 0,
    ],

    "oily skin seasonal factors": [
        lambda q: 2 if any(w in q for w in [
            "oily skin seasonal factors","why does my skin get oilier in summer",
            "can weather changes affect oily skin","seasonal oily skin",
            "summer oily skin","weather oily skin","environmental factors oily skin",
            "oily skin in summer"
        ]) and "oily skin" in q else 0,
    ],
    "what is oily skin": [
        lambda q: 1 if any(w in q for w in [
            "oily skin","what is oily skin","define oily skin","oily skin meaning",
            "explain oily skin","tell me about oily skin","info on oily skin","what's oily skin"
        ]) else 0,
    ],
    "sensitive skin symptoms": [
        lambda q: 2 if any(w in q for w in ["symptom", "sign", "recognize", "look like", "feel like", "red", "itchy", "stinging", "burning"]) and "sensitive skin" in q else 0,
        lambda q: 2 if "do i have sensitive skin" in q else 0,
    ],

    "sensitive skin triggers": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["trigger", "irritate", "weather", "food", "drink", "stress"]) else 0,
        lambda q: 2 if any(w in q for w in ["what irritates", "common triggers", "does stress"]) and "sensitive skin" in q else 0,
    ],

    "sensitive skin related issues": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["rash", "redness", "eczema", "dermatitis", "flare"]) else 0,
        lambda q: 2 if "why" in q and "sensitive skin" in q and any(w in q for w in ["flare", "rash", "redness"]) else 0,
    ],

    "sensitive skin diagnosis": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["diagnose", "test", "allergy"]) else 0,
        lambda q: 2 if "how can i test if my skin is sensitive" in q else 0,
    ],

    "sensitive skin home remedies": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["home remedy", "natural remedy", "diy", "oatmeal", "aloe", "soothe"]) else 0,
        lambda q: 2 if "can i use" in q and any(w in q for w in ["oatmeal", "aloe", "natural"]) and "sensitive skin" in q else 0,
    ],

    "sensitive skin routine": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["routine", "skincare", "moisturizer", "sunscreen", "patch test", "daily", "avoid ingredients"]) else 0,
        lambda q: 2 if "how to patch test" in q or "patch test new products" in q else 0,
    ],

    "sensitive skin lifestyle environmental": [
        lambda q: 2 if "sensitive skin" in q and any(w in q for w in ["pollution", "diet", "lifestyle", "seasonal", "environment"]) else 0,
        lambda q: 2 if "protect sensitive skin from pollution" in q or "manage sensitive skin seasonal" in q else 0,
    ],

    "what is sensitive skin": [
        lambda q: 2 if any(w in q for w in ["what is", "define", "explain", "meaning of", "info on"]) and "sensitive skin" in q else 0,
        lambda q: 2 if "what causes sensitive skin" in q or "is sensitive skin the same as allergic" in q else 0,
        lambda q: 1 if "sensitive skin" in q else 0,
    ],
     "activated charcoal mask": [
        lambda q: 2 if any(w in q for w in ["charcoal mask", "activated charcoal mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea tree charcoal mask", "charcoal and tea tree oil mask", "tea tree oil and charcoal mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use charcoal mask", "how do i use charcoal mask", "applying charcoal mask", "how often to use charcoal mask", "charcoal mask directions", "charcoal mask usage"]) else 0,
        lambda q: 2 if any(w in q for w in ["benefits of charcoal mask", "charcoal mask benefits", "is charcoal mask good for acne", "charcoal mask effects", "charcoal mask skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["charcoal mask for acne", "charcoal face mask for acne", "charcoal acne mask", "charcoal acne remedy", "charcoal pack for acne", "charcoal mask for pimples"]) else 0,
        lambda q: 2 if any(w in q for w in ["diy charcoal mask", "homemade charcoal mask", "charcoal mask home remedy", "natural charcoal mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["charcoal mask for blackheads", "purifying charcoal mask", "deep cleansing charcoal mask"]) else 0,
        lambda q: 2 if "charcoal mask ingredients" in q else 0,
        lambda q: 1 if "charcoal mask" in q else 0,
    ],

    "rice flour brightening paste": [
        lambda q: 2 if any(w in q for w in ["rice flour paste", "rice flour brightening paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["rice flour lemon honey paste", "lemon and honey rice flour paste", "rice flour and honey paste", "rice flour and lemon face pack"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make rice flour paste", "how to apply rice flour mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["diy rice flour mask", "natural rice flour face mask", "homemade rice flour mask", "natural brightening paste", "brightening face pack at home"]) else 0,
        lambda q: 2 if any(w in q for w in ["benefits of rice flour paste", "rice flour mask for glowing skin", "rice flour skin lightening", "glow rice flour mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["rice flour mask for acne", "rice flour mask for scars"]) else 0,
        lambda q: 2 if any(w in q for w in ["rice flour face pack", "rice flour pack for skin", "rice flour paste for skin", "skin brightening rice flour remedy", "rice flour paste remedy", "rice paste for skin care"]) else 0,
        lambda q: 2 if any(w in q for w in ["exfoliating rice flour paste", "rice flour face scrub"]) else 0,
        lambda q: 1 if "rice flour" in q else 0,
    ],

    "alum rose water toner": [
        lambda q: 2 if any(w in q for w in ["alum toner", "rose water toner", "alum and rose water toner"]) else 0,
        lambda q: 2 if any(w in q for w in ["diy alum toner", "how to make alum toner", "homemade toner for acne", "toner with rose water and alum", "diy toner with alum and rose water"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use alum toner"]) else 0,
        lambda q: 2 if any(w in q for w in ["benefits of alum toner", "rose water toner benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["alum for acne", "rose water for acne", "natural toner for acne", "acne toner diy", "acne rose water toner", "alum toner for oily skin", "acne skin toner rose water"]) else 0,
        lambda q: 2 if any(w in q for w in ["rose water for skin", "alum face toner", "rose water skin remedy", "alum toner skin remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["what is alum toner", "alum toner recipe", "alum toner effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["soothing alum toner", "alum toner for sensitive skin"]) else 0,
        lambda q: 1 if "alum rose water" in q or "alum toner" in q or "rose water toner" in q else 0,
    ],
    "baking soda with water paste": [
        lambda q: 2 if any(w in q for w in ["baking soda with water paste", "baking soda paste", "baking soda mask", "baking soda face pack"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make baking soda paste", "how to use baking soda for blackheads", "diy baking soda mask", "homemade baking soda paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["baking soda for blackheads", "blackhead remedy baking soda", "baking soda scrub for blackheads", "baking soda treatment for blackheads", "baking soda pore cleanser"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural remedy baking soda", "baking soda exfoliant", "baking soda for oily skin"]) else 0,
        lambda q: 1 if "baking soda" in q and "paste" in q else 0,
    ],

    "baking soda paste benefits": [
        lambda q: 2 if any(w in q for w in ["baking soda paste benefits", "benefits of baking soda paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does baking soda help blackheads", "baking soda for blackheads", "why use baking soda for skin", "baking soda clears pores"]) else 0,
        lambda q: 2 if any(w in q for w in ["baking soda mask uses", "advantages of baking soda face mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["baking soda anti-inflammatory benefits", "baking soda acne benefits", "baking soda skin brightening"]) else 0,
        lambda q: 1 if "baking soda" in q and "benefits" in q else 0,
    ],

    "baking soda paste side effects": [
        lambda q: 2 if any(w in q for w in ["baking soda paste side effects", "side effects of baking soda paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["is baking soda paste safe for skin", "can baking soda damage skin", "harms of using baking soda", "is baking soda safe"]) else 0,
        lambda q: 2 if any(w in q for w in ["baking soda skin irritation", "baking soda skin burning", "baking soda reaction on skin", "dryness from baking soda"]) else 0,
        lambda q: 2 if any(w in q for w in ["baking soda pH imbalance", "overuse of baking soda risks"]) else 0,
        lambda q: 1 if "baking soda" in q and "side effects" in q else 0,
    ],

    "tomato rub / pulp mask": [
        lambda q: 2 if any(w in q for w in ["tomato rub", "tomato pulp mask", "tomato mask", "tomato facial", "tomato face pack"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply tomato mask", "tomato skincare remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["tomato for blackheads", "blackhead removal with tomato", "tomato juice for blackheads"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural tomato mask", "diy tomato face pack", "tomato and lemon mask", "tomato and honey mask"]) else 0,
        lambda q: 2 if "tomato mask for oily skin" in q else 0,
        lambda q: 1 if "tomato" in q and ("mask" in q or "rub" in q or "pulp" in q) else 0,
    ],

    "tomato rub benefits": [
        lambda q: 2 if any(w in q for w in ["tomato rub benefits", "tomato pulp mask benefits", "benefits of tomato mask for blackheads"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does tomato help blackheads", "tomato clears skin", "why use tomato on face", "tomato for oily skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["tomato vitamin c benefits", "tomato mask brightening effects", "tomato anti-inflammatory", "tomato skin soothing", "tomato antioxidant benefits"]) else 0,
        lambda q: 1 if "tomato" in q and "benefits" in q else 0,
    ],

    "tomato rub side effects": [
        lambda q: 2 if any(w in q for w in ["tomato rub side effects", "tomato pulp mask side effects", "side effects of tomato mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["is tomato good for sensitive skin", "can tomato cause skin irritation", "negative effects of tomato on face", "is tomato safe for facial skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["tomato allergy symptoms", "tomato redness reaction", "tomato photosensitivity", "tomato mask stinging"]) else 0,
        lambda q: 1 if "tomato" in q and "side effects" in q else 0,
    ],

    "facial steaming": [
        lambda q: 2 if any(w in q for w in ["facial steaming", "steam face", "steam mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to do facial steaming", "steam with herbs"]) else 0,
        lambda q: 2 if any(w in q for w in ["steaming for blackheads", "face steam for blackheads", "steam for clear skin", "blackhead steam treatment", "open pores steaming", "natural blackhead removal steam"]) else 0,
        lambda q: 2 if any(w in q for w in ["steam for oily skin", "steam for deep cleansing"]) else 0,
        lambda q: 1 if "facial steaming" in q or "steam face" in q else 0,
    ],

    "facial steaming benefits": [
        lambda q: 2 if any(w in q for w in ["facial steaming benefits", "benefits of steam for blackheads"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does facial steaming help blackheads", "steam face for blackheads", "why steam face for acne", "open pores with steam"]) else 0,
        lambda q: 2 if any(w in q for w in ["facial steaming for oily skin", "steam skin detox", "steam to soften skin", "steam for better absorption", "steam for improved circulation"]) else 0,
        lambda q: 1 if "facial steaming" in q and "benefits" in q else 0,
    ],

    "facial steaming side effects": [
        lambda q: 2 if any(w in q for w in ["facial steaming side effects", "side effects of steam on face"]) else 0,
        lambda q: 2 if any(w in q for w in ["can steam harm skin", "is facial steaming safe", "steam skin damage", "facial steaming burns"]) else 0,
        lambda q: 2 if any(w in q for w in ["too much steam on face", "steaming face every day", "steam dryness", "steam irritation", "steam aggravating rosacea"]) else 0,
        lambda q: 1 if "facial steaming" in q and "side effects" in q else 0,
    ],

    "honey, crushed grains & multani mitti mask": [
        lambda q: 2 if any(w in q for w in ["honey crushed grains multani mitti mask", "honey mask", "multani mitti mask", "crushed grains mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make multani mitti mask", "homemade multani mitti face pack", "blackhead mask recipe"]) else 0,
        lambda q: 2 if any(w in q for w in ["multani mitti for blackheads", "natural blackhead mask", "honey face pack for acne"]) else 0,
        lambda q: 2 if any(w in q for w in ["honey and multani mitti mask", "grains and multani mitti for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["multani mitti scrub mask", "multani mitti exfoliation"]) else 0,
        lambda q: 1 if "honey" in q and "multani mitti" in q and "mask" in q else 0,
    ],

    "honey multani mitti mask benefits": [
        lambda q: 2 if any(w in q for w in ["honey multani mitti mask benefits", "benefits of multani mitti for blackheads", "crushed grains multani mitti mask benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does honey and multani mitti help", "why use honey in face pack"]) else 0,
        lambda q: 2 if any(w in q for w in ["multani mitti for oily skin", "multani mitti oil absorption"]) else 0,
        lambda q: 2 if any(w in q for w in ["honey antibacterial effects", "honey skin moisturizing"]) else 0,
        lambda q: 2 if any(w in q for w in ["advantages of multani mitti mask", "benefits of grain-based face masks", "exfoliating benefits of grains"]) else 0,
        lambda q: 1 if "honey multani mitti" in q and "benefits" in q else 0,
    ],

    "honey multani mitti mask side effects": [
        lambda q: 2 if any(w in q for w in ["honey multani mitti mask side effects", "multani mitti mask side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["can multani mitti cause dryness", "is this mask safe for all skin types", "side effects of honey face mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["multani mitti irritation", "honey mask allergy", "is multani mitti safe for sensitive skin", "mask causing redness", "dry skin after multani mitti", "allergic reaction to honey mask"]) else 0,
        lambda q: 1 if "honey multani mitti" in q and "side effects" in q else 0,
    ],

    "chilled cucumber slices": [
        lambda q: 2 if any(w in q for w in ["chilled cucumber slices", "cucumber slices", "cold cucumber for eyes", "cucumber on eyes for dark circles"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use cucumber slices for dark circles", "put cucumber on eyes", "diy cucumber eye treatment"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber for dark circles", "cucumber eye remedy", "cucumber remedy for under eye bags"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural remedy with cucumber", "cooling cucumber slices", "hydrating cucumber under eyes", "anti-inflammatory cucumber", "refreshing eye treatment"]) else 0,
        lambda q: 1 if "cucumber slices" in q or "cucumber for eyes" in q else 0,
    ],
    "benefits chilled cucumber slices": [
        lambda q: 2 if any(w in q for w in ["benefits of chilled cucumber slices", "chilled cucumber slices benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["why use cucumber slices for eyes", "what does cucumber do for dark circles"]) else 0,
        lambda q: 2 if any(w in q for w in ["cooling effect of cucumber", "cucumber properties for eyes", "cucumber healing dark circles"]) else 0,
        lambda q: 2 if any(w in q for w in ["reduces puffiness cucumber", "cucumber antioxidants for skin", "moisturizing cucumber benefits", "soothing effects of cucumber"]) else 0,
        lambda q: 1 if "cucumber benefits" in q else 0,
    ],
    "side effects chilled cucumber slices": [
        lambda q: 2 if any(w in q for w in ["side effects of chilled cucumber slices", "chilled cucumber slices side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["can cucumber irritate eyes", "is cucumber safe for eyes", "cucumber allergy on skin", "negative effects of cucumber on eyes"]) else 0,
        lambda q: 2 if any(w in q for w in ["eye redness from cucumber", "cucumber sensitivity", "possible irritation from cucumber slices"]) else 0,
        lambda q: 1 if "cucumber side effects" in q else 0,
    ],
    "cold tea bags (green or black)": [
        lambda q: 2 if any(w in q for w in ["cold tea bags", "cold green tea bags", "cold black tea bags"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea bags for dark circles", "green tea bags for eyes", "black tea bags for dark circles", "tea compress for eyes", "tea bag under eye remedy", "cold tea under eye treatment"]) else 0,
        lambda q: 2 if any(w in q for w in ["diy tea bag under eye remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["antioxidant tea bags", "tea bags reduce puffiness"]) else 0,
        lambda q: 1 if "tea bags" in q and ("cold" in q or "eyes" in q or "dark circles" in q) else 0,
    ],
    "benefits of cold tea bags": [
        lambda q: 2 if any(w in q for w in ["benefits of cold tea bags", "cold tea bags benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea bags help dark circles", "green tea bags under eyes benefits", "black tea bag eye remedy benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea bags reduce puffiness", "tea bags tighten skin", "caffeine benefits for eyes", "tea antioxidants for skin"]) else 0,
        lambda q: 1 if "tea bags" in q and "benefits" in q else 0,
    ],
    "side effects cold tea bags": [
        lambda q: 2 if any(w in q for w in ["side effects of cold tea bags", "cold tea bags side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea bag eye irritation", "can tea bags harm eyes", "is tea bag safe for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["tea allergy reaction", "side effects of tea bag compress", "eye redness from tea bags", "possible tea sensitivity"]) else 0,
        lambda q: 1 if "tea bags" in q and "side effects" in q else 0,
    ],

    "aloe vera gel": [
        lambda q: 2 if any(w in q for w in ["aloe vera gel", "aloe vera"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply aloe vera gel", "using aloe vera on undereyes", "aloe for under eye care"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe vera for dark circles", "aloe gel for dark circles", "diy aloe vera eye remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe moisturizing gel", "soothing aloe vera", "healing aloe for skin"]) else 0,
        lambda q: 1 if "aloe vera" in q and "gel" in q else 0,
    ],
    "benefits of aloe vera gel": [
        lambda q: 2 if any(w in q for w in ["benefits of aloe vera gel", "aloe vera gel benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["why use aloe vera under eyes", "healing properties of aloe vera", "aloe gel skin benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["dark circles aloe remedy benefits", "anti-inflammatory aloe vera", "aloe antioxidant benefits", "aloe skin repair"]) else 0,
        lambda q: 1 if "aloe vera" in q and "benefits" in q else 0,
    ],
    "side effects aloe vera gel": [
        lambda q: 2 if any(w in q for w in ["side effects of aloe vera gel", "aloe vera gel side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe skin irritation", "is aloe safe for eyes", "can aloe vera cause redness"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe vera allergy symptoms", "negative aloe vera effects", "possible aloe sensitivity", "eye irritation from aloe"]) else 0,
        lambda q: 1 if "aloe vera" in q and "side effects" in q else 0,
    ],

    "almond oil and vitamin e": [
        lambda q: 2 if any(w in q for w in ["almond oil and vitamin e", "almond oil", "vitamin e oil"]) else 0,
        lambda q: 2 if any(w in q for w in ["almond oil for dark circles", "vitamin e for dark circles", "almond oil and vitamin e for eyes"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use almond oil and vitamin e", "dark circles remedy almond oil", "vitamin e and almond oil eye treatment"]) else 0,
        lambda q: 2 if any(w in q for w in ["nourishing oils for skin", "moisturizing almond oil", "antioxidant vitamin e"]) else 0,
        lambda q: 1 if "almond oil" in q and "vitamin e" in q else 0,
    ],
    "benefits of almond oil and vitamin e": [
        lambda q: 2 if any(w in q for w in ["benefits of almond oil and vitamin e", "almond oil and vitamin e benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["almond oil for under eyes", "how does almond oil help", "vitamin e for skin health"]) else 0,
        lambda q: 2 if any(w in q for w in ["nourishing benefits of vitamin e", "under eye benefits almond oil", "skin repair oils", "vitamin e antioxidant effects"]) else 0,
        lambda q: 1 if "almond oil" in q and "vitamin e" in q and "benefits" in q else 0,
    ],
    "side effects almond oil and vitamin e": [
        lambda q: 2 if any(w in q for w in ["side effects of almond oil and vitamin e", "almond oil and vitamin e side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["can almond oil cause irritation", "vitamin e oil allergy", "is almond oil safe for sensitive skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["almond oil eye reaction", "side effects of vitamin e near eyes", "possible almond oil sensitivity", "eye redness from oils"]) else 0,
        lambda q: 1 if "almond oil" in q and "vitamin e" in q and "side effects" in q else 0,
    ],

    "chilled spoons or cold compress": [
        lambda q: 2 if any(w in q for w in ["chilled spoons", "cold compress", "ice compress for eye bags", "cold compress for puffy eyes", "cold spoons for puffy eyes"]) else 0,
        lambda q: 2 if any(w in q for w in ["spoons for eye bags", "how to use cold compress for eye bags", "how to use chilled spoons under eyes", "cold therapy for eye puffiness"]) else 0,
        lambda q: 2 if any(w in q for w in ["under eye swelling cold remedy", "cooling treatment for eye bags", "eye bags compress", "metal spoons for puffy eyes", "diy cold spoon remedy", "freeze spoons for eyes", "chill compress eyes", "cold pack for under eye bags"]) else 0,
        lambda q: 2 if any(w in q for w in ["reduces eye puffiness", "soothes tired eyes", "anti-inflammatory cold treatment"]) else 0,
        lambda q: 1 if "chilled spoons" in q or "cold compress" in q else 0,
    ],
    "chilled spoons benefits": [
        lambda q: 2 if any(w in q for w in ["benefits of chilled spoons for eyebags", "how does cold compress help with eyebags", "advantages of using chilled spoons on eyebags", "why use cold compress for eyebags", "cold compress eyebags benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["chilled spoons for puffiness", "reduce swelling with cold spoon", "cold therapy eye bag remedy", "improve eye bags with cold compress"]) else 0,
        lambda q: 2 if any(w in q for w in ["effectiveness of cold compress for eyes", "relieves under eye fluid retention", "refreshing eye care", "reduces redness and puffiness"]) else 0,
        lambda q: 1 if "chilled spoons" in q and "benefits" in q else 0,
    ],
    "chilled spoons side effects": [
        lambda q: 2 if any(w in q for w in ["side effects of chilled spoons for eyebags", "any risks using cold compress for eyebags", "negative effects of chilled spoons on eyebags", "problems with cold compress for eyebags", "cold compress eyebags side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["eye irritation from cold compress", "cold spoon damage under eyes", "discomfort from chilled spoons"]) else 0,
        lambda q: 2 if any(w in q for w in ["overuse of cold compress eyes", "safety of cold compress under eyes", "possible cold-induced skin sensitivity", "frostbite risk with extreme cold", "temporary redness from cold therapy"]) else 0,
        lambda q: 1 if "chilled spoons" in q and "side effects" in q else 0,
    ],

    "cucumber slices": [ # Specific for eye bags, as opposed to general skin
        lambda q: 2 if any(w in q for w in ["cucumber slices", "how to use cucumber slices for eye bags", "cucumber for eye bags"]) else 0,
        lambda q: 2 if any(w in q for w in ["puffy eye remedy cucumber", "cold cucumber slices for eyes", "cucumber under eyes for puffiness"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural remedy cucumber for eye bags", "diy cucumber eye remedy", "cucumber treatment for eye swelling"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber discs for eyes", "hydrating cucumber eye pads", "anti-inflammatory cucumber slices", "cooling effect cucumber eyes"]) else 0,
        lambda q: 1 if "cucumber slices" in q and ("eye bags" in q or "puffy eyes" in q) else 0,
    ],
    "cucumber slices benefits": [
        lambda q: 2 if any(w in q for w in ["benefits of cucumber slices for eyebags", "how do cucumber slices help with eyebags", "why use cucumber slices for eyebags", "advantages of cucumber slices on eyebags", "cucumber slices eyebags benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber soothes puffy eyes", "natural cooling from cucumber", "reduce swelling with cucumber", "cucumber calms under eye area"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin benefits of cucumber under eyes", "rejuvenates tired eyes", "antioxidant properties cucumber", "refreshes under eye skin"]) else 0,
        lambda q: 1 if "cucumber slices" in q and "benefits" in q and ("eye bags" in q or "eyes" in q) else 0,
    ],
    "cucumber slices side effects": [
        lambda q: 2 if any(w in q for w in ["side effects of cucumber slices for eyebags", "any risks using cucumber slices for eyebags", "negative effects of cucumber slices on eyebags", "problems with cucumber slices for eyebags", "cucumber slices eyebags side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber allergy under eyes", "skin reaction from cucumber", "cucumber causes irritation"]) else 0,
        lambda q: 2 if any(w in q for w in ["eye sensitivity to cucumber", "cucumber slices eye area issues", "possible redness from cucumber", "skin dryness from cucumber use", "eye irritation due to prolonged contact"]) else 0,
        lambda q: 1 if "cucumber slices" in q and "side effects" in q and ("eye bags" in q or "eyes" in q) else 0,
    ],

    "turmeric and lemon juice paste": [
        lambda q: 2 if any(w in q for w in ["turmeric and lemon juice paste", "turmeric lemon paste", "lemon turmeric face paste", "turmeric lemon brightening paste", "homemade turmeric lemon mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make turmeric lemon juice paste", "turmeric lemon diy mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric and lemon for freckles", "freckle remedy turmeric lemon"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin lightening with turmeric and lemon", "turmeric lemon mask for pigmentation", "turmeric lemon skin remedy", "turmeric lemon spot treatment"]) else 0,
        lambda q: 1 if "turmeric" in q and "lemon" in q and "paste" in q else 0,
    ],
    "turmeric and lemon juice paste benefits": [
        lambda q: 2 if any(w in q for w in ["benefits of turmeric and lemon juice paste", "turmeric lemon paste benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["why use turmeric and lemon juice on skin", "advantages of turmeric lemon paste", "turmeric and lemon juice effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["what are the benefits of turmeric and lemon paste", "is turmeric lemon paste good for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["brightening skin with turmeric and lemon", "freckle fading turmeric lemon", "natural glow turmeric lemon paste", "anti-pigmentation turmeric lemon benefits", "skin tone evening turmeric lemon"]) else 0,
        lambda q: 1 if "turmeric lemon" in q and "benefits" in q else 0,
    ],
    "turmeric and lemon juice paste side effects": [
        lambda q: 2 if any(w in q for w in ["side effects of turmeric and lemon juice paste", "turmeric lemon paste side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["any risks with turmeric lemon juice on skin", "is turmeric lemon paste safe", "can turmeric lemon paste harm skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric lemon juice allergic reactions", "what are the side effects of turmeric and lemon paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["lemon irritation with turmeric", "turmeric lemon burning sensation", "turmeric lemon skin sensitivity", "photosensitivity turmeric lemon", "skin redness from turmeric lemon paste"]) else 0,
        lambda q: 1 if "turmeric lemon" in q and "side effects" in q else 0,
    ],
    "honey and water": [
        lambda q: 2 if any(w in q for w in ["honey and water", "honey water"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use honey and water for freckles", "honey and water for freckles", "freckle remedy honey water"]) else 0,
        lambda q: 2 if any(w in q for w in ["honey water for skin", "diy honey water remedy", "natural honey water for pigmentation", "honey diluted for face"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin remedy honey with water", "apply honey water mix", "honey water brightening solution", "honey water moisturizing mix"]) else 0,
        lambda q: 1 if "honey" in q and "water" in q else 0,
    ],
    "honey and water benefits": [
        lambda q: 2 if any(w in q for w in ["benefits of honey and water", "why use honey and water for skin", "advantages of honey water mix"]) else 0,
        lambda q: 2 if any(w in q for w in ["honey and water effects on skin", "what are the benefits of honey and water for skin", "is honey water good for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin softening honey water", "natural hydration honey water", "clear skin with honey water", "moisturizing with honey water"]) else 0,
        lambda q: 2 if any(w in q for w in ["antibacterial honey water benefits", "glowing skin with honey water"]) else 0,
        lambda q: 1 if "honey water" in q and "benefits" in q else 0,
    ],
    "honey and water side effects": [
        lambda q: 2 if any(w in q for w in ["side effects of honey and water", "is honey and water safe for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["honey water allergic reactions", "any risks with honey and water on skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["what are the side effects of honey and water paste", "can honey water harm skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["stickiness issues honey water", "honey residue on skin problems", "honey water skin irritation", "freckle treatment risks honey water"]) else 0,
        lambda q: 2 if any(w in q for w in ["pore clogging risk honey water", "skin breakouts from honey water"]) else 0,
        lambda q: 1 if "honey water" in q and "side effects" in q else 0,
    ],

    "papaya juice": [
        lambda q: 2 if any(w in q for w in ["papaya juice", "papaya for skin brightening"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply papaya juice", "papaya juice face application", "raw papaya juice on skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya juice for freckles", "freckle remedy papaya", "how to use papaya on freckles", "homemade papaya freckle treatment"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural papaya skin remedy", "papaya skin lightening remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya enzyme exfoliation", "papaya juice natural bleaching"]) else 0,
        lambda q: 1 if "papaya juice" in q else 0,
    ],
    "papaya juice benefits": [
        lambda q: 2 if any(w in q for w in ["benefits of papaya juice", "why use papaya juice on skin", "advantages of papaya juice for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya juice effects", "what are the benefits of papaya juice for skin", "is papaya juice good for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya enzymes skin benefits", "freckle lightening with papaya", "glowing skin with papaya"]) else 0,
        lambda q: 2 if any(w in q for w in ["exfoliating properties papaya juice", "skin renewal papaya juice", "papaya juice anti-aging benefits"]) else 0,
        lambda q: 1 if "papaya juice" in q and "benefits" in q else 0,
    ],
    "papaya juice side effects": [
        lambda q: 2 if any(w in q for w in ["side effects of papaya juice", "is papaya juice safe for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya juice allergic reactions", "any risks with papaya juice on skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["what are the side effects of papaya juice on skin", "can papaya juice harm skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["papaya juice irritation", "rash from papaya juice", "papaya juice skin sensitivity", "freckle remedy papaya risks"]) else 0,
        lambda q: 2 if any(w in q for w in ["photosensitivity papaya juice", "skin redness from papaya"]) else 0,
        lambda q: 1 if "papaya juice" in q and "side effects" in q else 0,
    ],

    "sandalwood and rice flour remedy": [
        lambda q: 2 if any(w in q for w in ["sandalwood and rice flour remedy", "sandalwood rice flour remedy", "sandalwood and rice flour face mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make sandalwood rice flour remedy", "diy sandalwood rice flour paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["sandalwood and rice flour for pigmentation", "pigmentation remedy sandalwood rice flour", "rice flour for pigmentation"]) else 0,
        lambda q: 2 if any(w in q for w in ["sandalwood for skin lightening", "natural pigmentation remedy with rice flour"]) else 0,
        lambda q: 2 if any(w in q for w in ["sandalwood pigmentation treatment", "sandalwood rice flour spot remover", "sandalwood rice flour brightening mask"]) else 0,
        lambda q: 1 if "sandalwood" in q and "rice flour" in q else 0,
    ],
    "sandalwood and rice flour remedy benefits": [
        lambda q: 2 if any(w in q for w in ["sandalwood and rice flour remedy benefits", "benefits of sandalwood and rice flour remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["what are the benefits of sandalwood and rice flour remedy", "why use sandalwood and rice flour for skin", "advantages of sandalwood and rice flour"]) else 0,
        lambda q: 2 if any(w in q for w in ["brightening skin with sandalwood and rice flour", "natural glow from sandalwood and rice flour"]) else 0,
        lambda q: 2 if any(w in q for w in ["sandalwood rice flour face pack benefits", "even skin tone remedy rice flour", "remedy for dark spots sandalwood rice flour"]) else 0,
        lambda q: 2 if any(w in q for w in ["anti-pigmentation sandalwood benefits", "skin tone evening with rice flour and sandalwood"]) else 0,
        lambda q: 1 if "sandalwood rice flour" in q and "benefits" in q else 0,
    ],
    "sandalwood and rice flour remedy side effects": [
        lambda q: 2 if any(w in q for w in ["sandalwood and rice flour remedy side effects", "side effects of sandalwood and rice flour remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["any side effects of sandalwood and rice flour remedy", "is sandalwood and rice flour remedy safe"]) else 0,
        lambda q: 2 if any(w in q for w in ["can sandalwood and rice flour remedy cause allergies", "issues with sandalwood on skin", "rice flour skin irritation"]) else 0,
        lambda q: 2 if any(w in q for w in ["redness from sandalwood mask", "pigmentation remedy safety", "diy mask sandalwood rice flour risks", "skin sensitivity sandalwood rice flour"]) else 0,
        lambda q: 1 if "sandalwood rice flour" in q and "side effects" in q else 0,
    ],

    "nutmeg and milk remedy": [
        lambda q: 2 if any(w in q for w in ["nutmeg and milk remedy", "jaifil and milk remedy", "nutmeg milk remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make nutmeg milk remedy", "nutmeg milk face pack", "natural nutmeg milk remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["nutmeg and milk for pigmentation", "pigmentation remedy nutmeg milk"]) else 0,
        lambda q: 2 if any(w in q for w in ["milk for skin pigmentation", "jaifil for skin", "nutmeg pigmentation paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["nutmeg milk brightening mask", "nutmeg milk spot remover"]) else 0,
        lambda q: 1 if "nutmeg" in q and "milk" in q else 0,
    ],
    "nutmeg and milk remedy benefits": [
        lambda q: 2 if any(w in q for w in ["nutmeg and milk remedy benefits", "benefits of nutmeg and milk remedy", "what are the benefits of nutmeg and milk remedy", "jaifil and milk remedy benefits"]) else 0,
        lambda q: 2 if any(w in q for w in ["why use nutmeg and milk for skin", "nutmeg skin glow benefits", "brightening with nutmeg milk"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural nutmeg remedy for pigmentation", "reduce spots with nutmeg and milk"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin lightening jaifil milk benefits", "pigmentation fading nutmeg milk", "milk and nutmeg skin tone improvement"]) else 0,
        lambda q: 1 if "nutmeg milk" in q and "benefits" in q else 0,
    ],
    "nutmeg and milk remedy side effects": [
        lambda q: 2 if any(w in q for w in ["nutmeg and milk remedy side effects", "side effects of nutmeg and milk remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["any side effects of nutmeg and milk remedy", "is nutmeg and milk remedy safe", "can nutmeg and milk remedy cause allergies", "jaifil and milk remedy side effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["nutmeg milk skin irritation", "freckle remedy nutmeg risk", "acne from nutmeg milk remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["nutmeg face mask reaction", "skin redness from nutmeg milk", "milk nutmeg allergic response"]) else 0,
        lambda q: 1 if "nutmeg milk" in q and "side effects" in q else 0,
    ],

    "turmeric and milk paste": [
        lambda q: 2 if any(w in q for w in ["turmeric and milk paste", "turmeric milk paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make turmeric milk paste", "diy turmeric milk mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric and milk for pigmentation", "pigmentation remedy turmeric milk"]) else 0,
        lambda q: 2 if any(w in q for w in ["milk and turmeric for skin", "turmeric paste for dark spots", "turmeric milk for blemishes"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric milk remedy for hyperpigmentation", "turmeric milk skin lightening"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric milk anti-blemish paste", "turmeric milk glow treatment"]) else 0,
        lambda q: 1 if "turmeric" in q and "milk" in q and "paste" in q else 0,
    ],
    "turmeric and milk paste benefits": [
        lambda q: 2 if any(w in q for w in ["turmeric and milk paste benefits", "benefits of turmeric and milk paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["what are the benefits of turmeric and milk paste", "why use turmeric and milk for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric milk glow benefits", "turmeric milk to reduce pigmentation", "natural lightening with turmeric and milk"]) else 0,
        lambda q: 2 if any(w in q for w in ["milk turmeric paste for glowing skin", "turmeric and milk acne spot remedy"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric milk anti-inflammatory properties", "pigmentation reduction turmeric milk", "milk turmeric brightening effects"]) else 0,
        lambda q: 1 if "turmeric milk" in q and "benefits" in q else 0,
    ],
    "turmeric and milk paste side effects": [
        lambda q: 2 if any(w in q for w in ["turmeric and milk paste side effects", "side effects of turmeric and milk paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["any side effects of turmeric and milk paste", "is turmeric and milk paste safe", "can turmeric and milk paste cause allergies"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric skin staining", "milk and turmeric skin reaction", "turmeric paste burns or tingling"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric face mask sensitivity", "turmeric milk rash or redness", "skin irritation turmeric milk", "milk turmeric allergic reactions"]) else 0,
        lambda q: 1 if "turmeric milk" in q and "side effects" in q else 0,
    ],

    "cooling cucumber and yogurt body mask": [
        lambda q: 2 if any(w in q for w in ["cooling cucumber yogurt mask", "cucumber and yogurt mask", "cucumber yogurt mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["cooling cucumber mask", "yogurt cucumber mask", "how to make cucumber yogurt mask", "diy cucumber yogurt face mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber and yogurt for sunspots", "sunspot mask cucumber yogurt", "natural sunspot remedy cucumber yogurt"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin cooling remedy", "hydrating cucumber yogurt treatment"]) else 0,
        lambda q: 1 if "cucumber" in q and "yogurt" in q and "mask" in q else 0,
    ],
    "cooling cucumber and yogurt body mask benefits": [
        lambda q: 2 if any(w in q for w in ["cooling cucumber and yogurt body mask benefits", "benefits of cucumber and yogurt mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how cucumber and yogurt mask helps skin", "cooling effects of cucumber yogurt"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber yogurt benefits for sunspots", "hydration from cucumber yogurt mask", "natural skin cooling mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["cucumber yogurt for sun-damaged skin", "anti-inflammatory cucumber yogurt benefits", "brightening cucumber yogurt effects"]) else 0,
        lambda q: 1 if "cucumber yogurt mask" in q and "benefits" in q else 0,
    ],
    "cooling cucumber and yogurt body mask side effects": [
        lambda q: 2 if any(w in q for w in ["cooling cucumber and yogurt body mask side effects", "any side effects of cucumber and yogurt mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["can cucumber and yogurt mask cause irritation", "cucumber and yogurt mask harmful effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["yogurt skin allergy", "cucumber reaction on sensitive skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["risks of using cucumber yogurt mask", "possible redness from cucumber yogurt"]) else 0,
        lambda q: 1 if "cucumber yogurt mask" in q and "side effects" in q else 0,
    ],
    "potato slices": [
        lambda q: 2 if any(w in q for w in ["potato slices", "potato slice remedy", "slice of potato"]) else 0,
        lambda q: 2 if any(w in q for w in ["potato for skin", "potato skin remedy", "raw potato slices for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use potato slices for sunspots", "potato for sunspots", "sunspot remedy potato", "potato application for pigmentation"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural bleach potato slices", "potato juice for skin brightening"]) else 0,
        lambda q: 1 if "potato" in q and "skin" in q else 0,
    ],
    "potato slices benefits": [
        lambda q: 2 if any(w in q for w in ["potato slices benefits", "benefits of potato slices on skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["how potato slices help skin", "potato juice skin brightening"]) else 0,
        lambda q: 2 if any(w in q for w in ["potato for reducing sunspots", "potato as a natural bleach"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin lightening potato benefits", "potato antioxidant effects on skin"]) else 0,
        lambda q: 1 if "potato" in q and "benefits" in q else 0,
    ],
    "potato slices side effects": [
        lambda q: 2 if any(w in q for w in ["potato slices side effects", "any side effects of potato slices"]) else 0,
        lambda q: 2 if any(w in q for w in ["can potato slices irritate skin", "potato slices harmful effects"]) else 0,
        lambda q: 2 if any(w in q for w in ["potato allergy on skin", "raw potato skin sensitivity", "possible skin dryness from potato slices"]) else 0,
        lambda q: 1 if "potato" in q and "side effects" in q else 0,
    ],

    "turmeric and sandalwood paste": [
        lambda q: 2 if any(w in q for w in ["turmeric sandalwood paste", "turmeric and sandalwood mask", "sandalwood turmeric paste", "sandalwood turmeric mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make turmeric sandalwood paste", "natural brightening turmeric sandalwood mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric and sandalwood for sunspots", "sunspot remedy turmeric sandalwood", "sandalwood for dark spots", "turmeric for skin discoloration", "anti-pigmentation turmeric sandalwood"]) else 0,
        lambda q: 1 if "turmeric" in q and "sandalwood" in q and "paste" in q else 0,
    ],
    "turmeric and sandalwood paste benefits": [
        lambda q: 2 if any(w in q for w in ["turmeric and sandalwood paste benefits", "benefits of turmeric and sandalwood paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["how turmeric and sandalwood paste helps skin", "turmeric and sandalwood for glowing skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["anti-inflammatory benefits of turmeric", "sandalwood for calming skin", "turmeric and sandalwood skin healing properties"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin tone evening turmeric sandalwood", "pigmentation fading turmeric sandalwood"]) else 0,
        lambda q: 1 if "turmeric sandalwood" in q and "benefits" in q else 0,
    ],
    "turmeric and sandalwood paste side effects": [
        lambda q: 2 if any(w in q for w in ["turmeric and sandalwood paste side effects", "any side effects of turmeric and sandalwood paste"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric and sandalwood paste harmful effects", "can turmeric and sandalwood paste cause irritation"]) else 0,
        lambda q: 2 if any(w in q for w in ["turmeric skin allergy", "sandalwood reaction on sensitive skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["risks of using turmeric sandalwood paste", "skin redness turmeric sandalwood"]) else 0,
        lambda q: 1 if "turmeric sandalwood" in q and "side effects" in q else 0,
    ],

    "aloe vera gel for wrinkles": [
        lambda q: 2 if any(w in q for w in ["aloe vera gel", "aloe vera", "aloe gel", "aloe vera skin gel", "aloe vera leaf gel"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply aloe vera gel for wrinkles", "aloe vera for wrinkles", "wrinkle remedy aloe vera"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural wrinkle treatment aloe vera", "anti-aging aloe vera gel"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe vera moisturizing for wrinkles", "aloe vera skin rejuvenation"]) else 0,
        lambda q: 1 if "aloe vera" in q and "wrinkles" in q else 0,
    ],
    "aloe vera gel benefits (wrinkles)": [ # Clarified this is for wrinkles context
        lambda q: 2 if any(w in q for w in ["aloe vera gel benefits", "benefits of aloe vera gel", "what are the benefits of aloe vera gel"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does aloe vera gel help wrinkles", "aloe vera collagen boost"]) else 0,
        lambda q: 2 if any(w in q for w in ["hydration benefits of aloe vera", "anti-aging aloe vera properties", "skin soothing aloe vera benefits"]) else 0,
        lambda q: 1 if "aloe vera" in q and "benefits" in q and "wrinkles" in q else 0,
    ],
    "aloe vera gel side effects (wrinkles)": [ # Clarified this is for wrinkles context
        lambda q: 2 if any(w in q for w in ["aloe vera gel side effects", "side effects of aloe vera gel", "any side effects of aloe vera gel"]) else 0,
        lambda q: 2 if any(w in q for w in ["is aloe vera gel safe", "can aloe vera gel cause allergies"]) else 0,
        lambda q: 2 if any(w in q for w in ["aloe vera skin irritation", "allergic reaction aloe vera", "aloe vera gel sensitivity"]) else 0,
        lambda q: 1 if "aloe vera" in q and "side effects" in q and "wrinkles" in q else 0,
    ],

    "coconut oil massage": [
        lambda q: 2 if any(w in q for w in ["coconut oil massage", "coconut massage", "massage with coconut oil", "coconut oil body massage", "massage coconut oil"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to do coconut oil massage", "coconut oil for wrinkles", "wrinkle remedy coconut oil"]) else 0,
        lambda q: 2 if any(w in q for w in ["anti-aging coconut oil", "deep skin moisturizing coconut oil", "coconut oil elasticity boost"]) else 0,
        lambda q: 1 if "coconut oil" in q and "massage" in q else 0,
    ],
    "coconut oil massage benefits": [
        lambda q: 2 if any(w in q for w in ["coconut oil massage benefits", "benefits of coconut oil massage", "what are the benefits of coconut oil massage"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does coconut oil massage help wrinkles", "coconut oil for skin elasticity"]) else 0,
        lambda q: 2 if any(w in q for w in ["deep moisturizing coconut oil", "coconut oil antioxidant benefits", "coconut oil skin nourishment"]) else 0,
        lambda q: 1 if "coconut oil massage" in q and "benefits" in q else 0,
    ],
    "coconut oil massage side effects": [
        lambda q: 2 if any(w in q for w in ["coconut oil massage side effects", "side effects of coconut oil massage", "any side effects of coconut oil massage"]) else 0,
        lambda q: 2 if any(w in q for w in ["is coconut oil massage safe", "can coconut oil massage cause allergies"]) else 0,
        lambda q: 2 if any(w in q for w in ["coconut oil pore clogging", "skin breakouts from coconut oil", "coconut oil acne risk"]) else 0,
        lambda q: 1 if "coconut oil massage" in q and "side effects" in q else 0,
    ],

    "egg white mask": [
        lambda q: 2 if any(w in q for w in ["egg white mask", "egg white face mask", "egg white facial mask", "egg mask", "egg white pack"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply egg white mask", "egg white for wrinkles", "wrinkle remedy egg white"]) else 0,
        lambda q: 2 if any(w in q for w in ["egg white tightening mask", "egg white skin firming"]) else 0,
        lambda q: 1 if "egg white mask" in q else 0,
    ],
    "egg white mask benefits": [
        lambda q: 2 if any(w in q for w in ["egg white mask benefits", "benefits of egg white mask", "what are the benefits of egg white mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does egg white mask help wrinkles", "egg white for firming skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["natural skin tightening remedy", "egg white pore tightening", "egg white skin texture improvement"]) else 0,
        lambda q: 1 if "egg white mask" in q and "benefits" in q else 0,
    ],
    "egg white mask side effects": [
        lambda q: 2 if any(w in q for w in ["egg white mask side effects", "side effects of egg white mask", "any side effects of egg white mask"]) else 0,
        lambda q: 2 if any(w in q for w in ["is egg white mask safe", "can egg white mask cause allergies"]) else 0,
        lambda q: 2 if any(w in q for w in ["egg white skin sensitivity", "egg white allergic reaction", "egg white irritation risks"]) else 0,
        lambda q: 1 if "egg white mask" in q and "side effects" in q else 0,
    ],

    # --------------------------------------------GENERAL QUESTIONS---------------------------------------------------------

    "diet and nutrition for skin": [
        lambda q: 2 if any(w in q for w in ["diet for skin health", "nutrition for skin", "what diet is good for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["best diet for healthy skin", "skin health diet", "eating for skin health"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does diet affect skin", "skin nutrition", "what nutrition is good for skin", "how does nutrition impact skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["foods for glowing skin", "vitamins for skin", "healthy diet skin benefits", "nutrients good for skin", "skin friendly diet"]) else 0,
        lambda q: 1 if "diet" in q and "skin" in q else 0,
    ],

    "stress impact on skin": [
        lambda q: 2 if any(w in q for w in ["stress and skin", "how does stress affect skin", "skin problems from stress"]) else 0,
        lambda q: 2 if any(w in q for w in ["stress impact on skin health", "can stress cause skin issues", "effect of stress on skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["stress acne", "stress skin flare ups", "stress causing skin inflammation"]) else 0,
        lambda q: 2 if any(w in q for w in ["stress and breakouts", "managing stress for better skin"]) else 0,
        lambda q: 1 if "stress" in q and "skin" in q else 0,
    ],

    "sleep impact on skin": [
        lambda q: 2 if any(w in q for w in ["sleep and skin", "how does sleep affect skin", "sleep impact on skin health"]) else 0,
        lambda q: 2 if any(w in q for w in ["beauty sleep", "lack of sleep and skin", "good sleep for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["sleep deprivation skin damage", "sleep and skin regeneration"]) else 0,
        lambda q: 2 if any(w in q for w in ["importance of sleep for skin", "nighttime skin repair"]) else 0,
        lambda q: 1 if "sleep" in q and "skin" in q else 0,
    ],

    "using non-comedogenic products": [
        lambda q: 2 if any(w in q for w in ["skincare products", "what skincare products to use", "best skincare products", "product recommendations for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["non-comedogenic products", "what are non-comedogenic products", "using non-comedogenic skincare"]) else 0,
        lambda q: 2 if any(w in q for w in ["makeup for acne", "best makeup for acne prone skin", "acne friendly makeup", "non-comedogenic makeup", "can makeup cause acne"]) else 0,
        lambda q: 2 if any(w in q for w in ["oil free skincare", "non pore clogging products", "sensitive skin makeup"]) else 0,
        lambda q: 1 if "non-comedogenic" in q or ("skincare" in q and "products" in q) else 0,
    ],

    "protecting skin from sun": [
        lambda q: 2 if any(w in q for w in ["sunscreen for acne", "best sunscreen for acne prone skin", "sun protection for acne"]) else 0,
        lambda q: 2 if any(w in q for w in ["is sunscreen good for acne", "importance of sunscreen for skin", "how to protect skin from sun"]) else 0,
        lambda q: 2 if any(w in q for w in ["sun protection", "broad spectrum sunscreen", "spf for sensitive skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["daily sunscreen use", "sun damage prevention"]) else 0,
        lambda q: 1 if "sun" in q and "skin" in q else 0,
    ],
     "washing face tips": [
        lambda q: 2 if any(w in q for w in ["washing face", "how to wash face", "face washing tips", "proper way to wash face", "best way to wash face for clear skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["face cleansing routine", "face wash frequency", "gentle face washing", "avoid over washing face"]) else 0,
        lambda q: 2 if any(w in q for w in ["cleanse skin properly", "morning and night face washing"]) else 0,
        lambda q: 1 if "wash face" in q or "face cleansing" in q else 0,
    ],

    "avoid picking or popping pimples": [
        lambda q: 2 if any(w in q for w in ["picking pimples", "should i pick pimples", "popping pimples", "is it bad to pick pimples"]) else 0,
        lambda q: 2 if any(w in q for w in ["don't pick pimples", "harm of picking pimples", "pimple picking scars", "pimple popping risks"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to avoid picking pimples", "pimple healing tips", "reduce pimple inflammation"]) else 0,
        lambda q: 1 if "picking pimples" in q or "popping pimples" in q else 0,
    ],

    "eating well for skin": [
        lambda q: 2 if any(w in q for w in ["eating well for skin", "food for healthy skin", "good food for skin", "healthy eating for skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["nutrition for healthy skin", "diet for healthy complexion", "skin boosting foods"]) else 0,
        lambda q: 2 if any(w in q for w in ["antioxidants for skin", "hydrating foods for skin", "vitamins and minerals for skin", "balanced diet for skin health"]) else 0,
        lambda q: 1 if "eating" in q and "skin" in q else 0,
    ],

    "aloe vera": [ # Distinguished from specific uses like for dark circles or wrinkles
        lambda q: 2 if any(w in q for w in ["is aloe vera good for acne?", "what are the benefits of aloe vera for skin?", "how do you use aloe vera on your face?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can aloe vera cause allergies?", "is aloe vera effective for wrinkles?", "does aloe vera help with skin hydration?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should i apply aloe vera gel?", "is aloe vera safe for sensitive skin?", "is aloe vera good for pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can aloe vera soothe irritated skin?", "does aloe vera reduce redness?", "can aloe vera cause skin irritation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is aloe vera gel good for sunburn?", "can aloe vera help with dry skin?"]) else 0,
        lambda q: 1 if "aloe vera" in q else 0,
    ],
    "lemon juice": [ # Distinguished from specific uses like in turmeric paste
        lambda q: 2 if any(w in q for w in ["is lemon juice good for skin whitening?", "can lemon juice help with acne scars?", "how do you apply lemon juice on the skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is lemon juice safe for sensitive skin?", "does lemon juice cause skin irritation?", "can lemon juice make skin dry?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should lemon juice be used on skin?", "can lemon juice cause photosensitivity?", "is lemon juice good for pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can lemon juice lighten dark spots?", "does lemon juice help reduce redness?", "is lemon juice effective for oily skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is lemon juice safe to use daily?", "can lemon juice cause allergic reactions?"]) else 0,
        lambda q: 1 if "lemon juice" in q else 0,
    ],
    "honey": [ # Distinguished from specific uses like with water
        lambda q: 2 if any(w in q for w in ["is honey good for acne?", "what are the skin benefits of honey?", "how do you use honey for glowing skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can honey help with dry skin?", "is raw honey better than processed honey for skin?", "does honey have antibacterial properties for skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can honey cause allergic reactions?", "how often should honey be applied to skin?", "is honey good for pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can honey reduce inflammation on skin?", "does honey moisturize skin effectively?", "is honey safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can honey soothe irritated skin?", "does honey help with wound healing?"]) else 0,
        lambda q: 1 if "honey" in q and "skin" in q else 0,
    ],
    "turmeric": [ # Distinguished from specific uses like in pastes
        lambda q: 2 if any(w in q for w in ["is turmeric good for reducing pigmentation?", "how does turmeric help with acne?", "can turmeric be used daily on skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does turmeric cause staining on skin?", "what are the anti-inflammatory benefits of turmeric?", "can turmeric lighten dark spots?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is turmeric safe for sensitive skin?", "how to make a turmeric face mask?", "can turmeric cause allergic reactions?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is turmeric effective for wrinkles?", "does turmeric help reduce redness?", "is turmeric good for oily skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can turmeric soothe irritated skin?"]) else 0,
        lambda q: 1 if "turmeric" in q and "skin" in q else 0,
    ],
    "oatmeal": [
        lambda q: 2 if any(w in q for w in ["is oatmeal good for sensitive skin?", "how does oatmeal soothe irritated skin?", "can oatmeal help with rosacea?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to prepare an oatmeal face mask?", "does oatmeal exfoliate the skin?", "is oatmeal safe for daily use on skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can oatmeal reduce redness on the face?", "does oatmeal moisturize the skin?", "is oatmeal effective for dry skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can oatmeal calm inflammation?", "does oatmeal help with acne?", "is oatmeal good for pigmentation?", "can oatmeal cause allergic reactions?"]) else 0,
        lambda q: 1 if "oatmeal" in q and "skin" in q else 0,
    ],
    "yogurt": [ # Distinguished from specific uses like in masks
        lambda q: 2 if any(w in q for w in ["is yogurt good for skin hydration?", "how does yogurt help with acne?", "can yogurt lighten dark spots?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply yogurt on the face?", "is yogurt safe for all skin types?", "does yogurt exfoliate dead skin cells?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can yogurt cause allergic reactions?", "how often should yogurt be used on skin?", "is yogurt good for pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does yogurt soothe irritated skin?", "is yogurt effective for dry skin?", "can yogurt help reduce redness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is yogurt safe for sensitive skin?"]) else 0,
        lambda q: 1 if "yogurt" in q and "skin" in q else 0,
    ],
    "cucumber": [ # Distinguished from specific uses like for eye bags
        lambda q: 2 if any(w in q for w in ["is cucumber good for reducing puffiness?", "how does cucumber help with sunburn?", "can cucumber lighten dark circles?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use cucumber slices on face?", "does cucumber hydrate the skin?", "is cucumber safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can cucumber reduce redness on skin?", "how often can cucumber be applied on skin?", "is cucumber effective for pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can cucumber soothe irritated skin?", "does cucumber help with oily skin?", "is cucumber good for dry skin?", "can cucumber cause allergic reactions?"]) else 0,
        lambda q: 1 if "cucumber" in q and "skin" in q else 0,
    ],
    "potato": [ # Distinguished from specific "potato slices" remedy
        lambda q: 2 if any(w in q for w in ["is potato good for lightening dark spots?", "how does potato juice help skin?", "can potato reduce pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply potato slices on face?", "does potato have exfoliating properties?", "is potato safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can potato cause skin irritation?", "how often should potato be used for skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is potato effective for acne scars?", "can potato soothe irritated skin?", "does potato brighten skin tone?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is potato good for dry skin?", "can potato lighten freckles?"]) else 0,
        lambda q: 1 if "potato" in q and "skin" in q else 0,
    ],
    "sandalwood": [ # Distinguished from specific "turmeric and sandalwood paste" remedy
        lambda q: 2 if any(w in q for w in ["is sandalwood good for skin lightening?", "how does sandalwood help with pigmentation?", "can sandalwood soothe irritated skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to make a sandalwood face mask?", "does sandalwood reduce acne scars?", "is sandalwood safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can sandalwood cause allergies?", "how often should sandalwood be applied?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is sandalwood effective for oily skin?", "does sandalwood have anti-inflammatory properties?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can sandalwood help with wrinkles?", "is sandalwood good for dry skin?"]) else 0,
        lambda q: 1 if "sandalwood" in q and "skin" in q else 0,
    ],
    "rose water": [
        lambda q: 2 if any(w in q for w in ["is rose water good for skin hydration?", "how does rose water help with acne?", "can rose water reduce redness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use rose water as a toner?", "is rose water safe for sensitive skin?", "does rose water have anti-inflammatory properties?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can rose water help with oily skin?", "how often should rose water be applied?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is rose water effective for pigmentation?", "can rose water soothe irritated skin?", "does rose water tighten pores?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is rose water good for dry skin?"]) else 0,
        lambda q: 1 if "rose water" in q and "skin" in q else 0,
    ],
    "baking soda": [
        lambda q: 2 if any(w in q for w in ["is baking soda good for exfoliation?", "can baking soda help with acne?", "does baking soda cause skin irritation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use baking soda safely on skin?", "is baking soda suitable for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can baking soda lighten dark spots?", "how often can baking soda be used on skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["are there risks of using baking soda on skin?", "is baking soda effective for oily skin?", "can baking soda cause dryness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does baking soda disrupt skin ph?", "is baking soda good for blackheads?"]) else 0,
        lambda q: 1 if "baking soda" in q and "skin" in q else 0,
    ],
    "lavender oil": [
        lambda q: 2 if any(w in q for w in ["is lavender oil good for rosacea?", "how does lavender oil help with acne?", "can lavender oil soothe irritated skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use lavender oil on skin safely?", "is lavender oil safe for sensitive skin?", "can lavender oil cause allergic reactions?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does lavender oil have antibacterial properties?", "how often should lavender oil be applied?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is lavender oil good for scars?", "can lavender oil reduce redness?", "is lavender oil effective for wrinkles?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can lavender oil help with oily skin?"]) else 0,
        lambda q: 1 if "lavender oil" in q and "skin" in q else 0,
    ],
    "activated charcoal": [
        lambda q: 2 if any(w in q for w in ["is activated charcoal good for deep cleansing?", "can activated charcoal help with acne?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how does activated charcoal detoxify skin?", "is activated charcoal safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should activated charcoal masks be used?", "does activated charcoal dry out the skin?", "can activated charcoal cause irritation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply activated charcoal on face?", "is activated charcoal effective for blackheads?", "can activated charcoal reduce oiliness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does activated charcoal exfoliate skin?", "is activated charcoal good for pigmentation?"]) else 0,
        lambda q: 1 if "activated charcoal" in q and "skin" in q else 0,
    ],
    "tea tree oil": [
        lambda q: 2 if any(w in q for w in ["is tea tree oil effective for acne?", "how to dilute tea tree oil for skin use?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can tea tree oil cause skin irritation?", "does tea tree oil have antibacterial properties?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is tea tree oil safe for sensitive skin?", "how often can tea tree oil be applied?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can tea tree oil reduce redness?", "is tea tree oil good for oily skin?", "is tea tree oil effective for blemishes?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can tea tree oil help with scars?", "does tea tree oil soothe inflamed skin?"]) else 0,
        lambda q: 1 if "tea tree oil" in q and "skin" in q else 0,
    ],
    "rice flour": [ # Distinguished from specific "sandalwood and rice flour remedy"
        lambda q: 2 if any(w in q for w in ["is rice flour good for exfoliating skin?", "how does rice flour help with pigmentation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can rice flour lighten skin tone?", "how to use rice flour in face masks?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is rice flour safe for sensitive skin?", "does rice flour absorb excess oil?", "can rice flour cause dryness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should rice flour be used on skin?", "is rice flour effective for acne?", "can rice flour reduce redness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["does rice flour exfoliate gently?"]) else 0,
        lambda q: 1 if "rice flour" in q and "skin" in q else 0,
    ],
    "alum": [
        lambda q: 2 if any(w in q for w in ["is alum good for acne treatment?", "how does alum help with skin tightening?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can alum cause skin irritation?", "is alum safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply alum on skin?", "does alum reduce oiliness?", "can alum lighten dark spots?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should alum be used on skin?", "is alum effective for pores?", "can alum soothe irritated skin?"]) else 0,
        lambda q: 1 if "alum" in q and "skin" in q else 0,
    ],
    "multani mitti": [
        lambda q: 2 if any(w in q for w in ["is multani mitti good for oily skin?", "how does multani mitti help with acne?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can multani mitti reduce pigmentation?", "how to prepare multani mitti face pack?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is multani mitti safe for sensitive skin?", "does multani mitti absorb excess oil?", "can multani mitti cause dryness?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how often should multani mitti be used?", "is multani mitti effective for exfoliation?", "can multani mitti soothe irritated skin?"]) else 0,
        lambda q: 1 if "multani mitti" in q and "skin" in q else 0,
    ],
    "almonds": [ # Distinguished from specific "almond oil and vitamin e"
        lambda q: 2 if any(w in q for w in ["are almonds good for skin health?", "how does almond oil help with skin hydration?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can almonds reduce wrinkles?", "is almond oil safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to use almonds in skincare?", "do almonds help with acne scars?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can almond oil cause allergies?", "how often should almond oil be applied?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is almond oil good for pigmentation?", "can almonds improve skin elasticity?"]) else 0,
        lambda q: 1 if "almonds" in q and "skin" in q else 0,
    ],
    "milk": [ # Distinguished from specific "nutmeg and milk remedy" and "turmeric and milk paste"
        lambda q: 2 if any(w in q for w in ["is milk good for skin moisturizing?", "how does milk help with exfoliation?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can milk lighten dark spots?", "is milk safe for sensitive skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to apply milk on skin?", "does milk cause acne?"]) else 0,
        lambda q: 2 if any(w in q for w in ["can milk soothe irritated skin?", "how often should milk be used on skin?"]) else 0,
        lambda q: 2 if any(w in q for w in ["is milk effective for dry skin?", "can milk help with pigmentation?"]) else 0,
        lambda q: 1 if "milk" in q and "skin" in q else 0,
    ],
      # --------------------------------------------WEBSITE INFO---------------------------------------------------------

    "skin diagnosis info": [
        lambda q: 2 if any(w in q for w in ["how does skin diagnosis work", "what is skin diagnosis", "how does the ai predict skin issues"]) else 0,
        lambda q: 2 if any(w in q for w in ["how can i diagnose my skin", "what is skin analysis", "how accurate is skin diagnosis"]) else 0,
        lambda q: 2 if any(w in q for w in ["can i upload my photo for skin diagnosis", "how do i check my skin condition"]) else 0,
        lambda q: 1 if "skin diagnosis" in q or "skin analysis" in q else 0,
    ],
    "developer info": [
        lambda q: 2 if any(w in q for w in ["who develop this website", "who made this website", "who created this website", "who is the developer"]) else 0,
        lambda q: 2 if any(w in q for w in ["developer of this website", "site developer", "about the developer", "who built this site"]) else 0,
        lambda q: 2 if any(w in q for w in ["website creator", "who coded this website", "developer info", "info about developer"]) else 0,
        lambda q: 2 if any(w in q for w in ["website developed by", "creator of website", "website designer", "who designed this website"]) else 0,
        lambda q: 2 if any(w in q for w in ["who is behind this website", "contact developer", "website development team", "who is the founder of this website"]) else 0,
        lambda q: 2 if any(w in q for w in ["developer information", "who is the creator", "website maintainer", "site owner", "owner", "founder"]) else 0,
        lambda q: 1 if "developer" in q or "creator" in q or "owner" in q else 0,
    ],
    "gallery info": [
        lambda q: 2 if any(w in q for w in ["what skin issues are covered", "show me the skin problems", "what conditions can this site detect"]) else 0,
        lambda q: 2 if any(w in q for w in ["what skin conditions are included in gallery", "can i learn more about acne", "does this site include rosacea"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin problem examples", "show skin conditions photos"]) else 0,
        lambda q: 1 if "gallery" in q or "skin issues" in q else 0,
    ],
    "article info": [
        lambda q: 2 if any(w in q for w in ["do you have skin care articles", "where can i read articles", "home remedy blogs"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin tips and articles", "skincare advice articles", "latest skincare blog"]) else 0,
        lambda q: 2 if any(w in q for w in ["best skincare articles", "skin health articles"]) else 0,
        lambda q: 1 if "articles" in q or "blog" in q else 0,
    ],
    "contact info": [
        lambda q: 2 if any(w in q for w in ["how can i contact you", "how to email you", "contact support"]) else 0,
        lambda q: 2 if any(w in q for w in ["customer support email", "how to reach out", "contact information"]) else 0,
        lambda q: 1 if "contact" in q or "email" in q else 0,
    ],
    "review info": [
        lambda q: 2 if any(w in q for w in ["where can i leave a review", "can i share feedback", "submit a review"]) else 0,
        lambda q: 2 if any(w in q for w in ["leave feedback", "rate this website", "write a testimonial"]) else 0,
        lambda q: 1 if "review" in q or "feedback" in q else 0,
    ],
    "faq info": [
        lambda q: 2 if any(w in q for w in ["frequently asked questions", "where is the faq section", "common questions"]) else 0,
        lambda q: 2 if any(w in q for w in ["help section", "faq page", "questions and answers", "most asked questions"]) else 0,
        lambda q: 1 if "faq" in q or "questions" in q else 0,
    ],
    "login required info": [
        lambda q: 2 if any(w in q for w in ["do i need to login", "can i use this without signing in", "do i need to sign up to diagnose my skin"]) else 0,
        lambda q: 2 if any(w in q for w in ["can i use diagnosis without signing in", "is account required", "do i need an account"]) else 0,
        lambda q: 1 if "login" in q or "account" in q else 0,
    ],
    "signup help": [
        lambda q: 2 if any(w in q for w in ["how to sign up", "how do i create an account", "register account", "signup process"]) else 0,
        lambda q: 1 if "sign up" in q or "create account" in q else 0,
    ],
    "login help": [
        lambda q: 2 if any(w in q for w in ["how to login", "sign in help", "forgot password", "login issues"]) else 0,
        lambda q: 1 if "login" in q or "sign in" in q else 0,
    ],
    "general help": [
        lambda q: 2 if any(w in q for w in ["can you help me", "help me", "i need help", "what can i ask you"]) else 0,
        lambda q: 2 if any(w in q for w in ["how can you assist me", "support help", "help"]) else 0,
        lambda q: 1 if "help" in q else 0,
    ],
    "website usage help": [
        lambda q: 2 if any(w in q for w in ["how do i get started", "guide me through the site", "how to use this website"]) else 0,
        lambda q: 2 if any(w in q for w in ["website walkthrough", "site tutorial", "getting started guide"]) else 0,
        lambda q: 1 if "how to use" in q or "website usage" in q else 0,
    ],
    "common concerns": [
        lambda q: 2 if any(w in q for w in ["common concerns", "frequent problems", "typical skin issues", "most common skin problems"]) else 0,
        lambda q: 1 if "common concerns" in q or "skin problems" in q else 0,
    ],
    "skincare basics": [
        lambda q: 2 if any(w in q for w in ["how to take care of my skin", "skin care routine", "daily skincare routine"]) else 0,
        lambda q: 2 if any(w in q for w in ["how to maintain healthy skin", "skin care for beginners", "basic skincare tips"]) else 0,
        lambda q: 2 if any(w in q for w in ["morning skincare routine", "night skincare routine", "simple skincare steps"]) else 0,
        lambda q: 1 if "skincare routine" in q or "skincare tips" in q else 0,
    ],
    "personalized skincare": [
        lambda q: 2 if any(w in q for w in ["best skincare routine", "what skincare should i use", "recommend skincare for me"]) else 0,
        lambda q: 2 if any(w in q for w in ["what products should i use", "custom skincare advice", "personalized skin care tips"]) else 0,
        lambda q: 2 if any(w in q for w in ["skin care for my skin type", "which skincare suits me"]) else 0,
        lambda q: 1 if "personalized skincare" in q or "custom skincare" in q else 0,
    ],
    "skincare products difference": [
        lambda q: 2 if any(w in q for w in ["difference between moisturizer and serum", "moisturizer vs serum"]) else 0,
        lambda q: 2 if any(w in q for w in ["toner vs astringent", "difference between sunscreen and sunblock"]) else 0,
        lambda q: 2 if any(w in q for w in ["what is serum", "what does toner do", "when to use moisturizer", "what is astringent"]) else 0,
        lambda q: 1 if "difference between" in q and "skincare" in q else 0,
    ],
    "greeting": [
        lambda q: 2 if any(w in q for w in ["hi", "hello", "hey", "hola", "salam", "asalam u alaikum"]) else 0,
        lambda q: 2 if any(w in q for w in ["hey glowgenie", "hey glow.genie", "hey genie", "hi genie"]) else 0,
        lambda q: 1 if "hi" in q or "hello" in q else 0,
    ],
    "good morning": [
        lambda q: 2 if any(w in q for w in ["good morning", "morning", "morning greetings"]) else 0,
        lambda q: 1 if "morning" in q else 0,
    ],
    "good afternoon": [
        lambda q: 2 if any(w in q for w in ["good afternoon", "afternoon", "afternoon greetings"]) else 0,
        lambda q: 1 if "afternoon" in q else 0,
    ],
    "good evening": [
        lambda q: 2 if any(w in q for w in ["good evening", "night", "good night", "evening greetings"]) else 0,
        lambda q: 1 if "evening" in q or "night" in q else 0,
    ],
    "how are you": [
        lambda q: 2 if any(w in q for w in ["how are you", "what's up", "how is it going", "how do you do"]) else 0,
        lambda q: 1 if "how are you" in q else 0,
    ],
    "thanks": [
        lambda q: 2 if any(w in q for w in ["thanks", "thank you", "thanks for helping", "appreciate it"]) else 0,
        lambda q: 2 if any(w in q for w in ["thank you so much", "thanks a lot"]) else 0,
        lambda q: 1 if "thanks" in q or "thank you" in q else 0,
    ],
    "how it works": [
        lambda q: 2 if any(w in q for w in ["how does this work", "how to use this site", "what can you do"]) else 0,
        lambda q: 2 if any(w in q for w in ["how can you help me", "what is this website for", "explain this website", "how does the diagnosis work"]) else 0,
        lambda q: 1 if "how it works" in q or "how to use" in q else 0,
    ],


    }

    scores = {tag: sum(rule(text) for rule in rules) for tag, rules in scoring_rules.items()}
    best_tag = max(scores, key=scores.get)
    return best_tag if scores[best_tag] > 0 else None


@csrf_exempt
@require_POST
def chatbot(request):
    ensure_loaded()  # ensure data is cached

    try:
        body = json.loads(request.body)
        user_input = body.get("message", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"response": "Invalid input."}, status=400)

    if not user_input:
        return JsonResponse({"response": "Please enter a message."})

    user_input_norm = normalize(user_input)

    # 1. Exact match from DB
    best_question = next((q for q in cached_questions if normalize(q.text) == user_input_norm), None)

    # 2. From normalized mapping
    if not best_question:
        normalized_input = NORMALIZED_INPUTS.get(user_input_norm)
        if normalized_input:
            best_question = next((q for q in cached_questions if normalize(q.text) == normalize(normalized_input)), None)

    # 3. Keyword/topic-based
    if not best_question:
        topic = detect_topic(user_input)
        if topic:
            # You need a question in your DB that maps to the topic to get an answer.
            # Example: A question with text "Topic: {topic_name}" could hold the answer.
            best_question = next((q for q in cached_questions if normalize(q.text) == topic), None)

    # 4. Semantic match using spaCy
    # Note: `score_tag` and `topic` logic seem redundant/unclear in the original code.
    # The `detect_topic` handles keyword-based intent.
    if not best_question:
        best_question = get_best_match_spacy(user_input, cached_questions, cached_spacy_docs)

    # 5. Return found answer
    if best_question and best_question.answer:
        return JsonResponse({"response": best_question.answer.content})

    # 6. Fallback
    log_unmatched_query(user_input)
    topic = detect_topic(user_input)
    fallback_buttons = topic_buttons.get(topic, topic_buttons.get("default", []))
    
    return JsonResponse({
        "response": [
            {"type": "paragraph", "text": "Sorry, I didn't find an answer for that."},
            {"type": "button_group", "buttons": fallback_buttons}
        ]
    })
# @require_GET
# def initial_greeting(request):
#     try:
#         initial_question = Question.objects.get(text__iexact="initial_greeting_payload")
#         content = json.loads(initial_question.answer.content)
#     except:
#         content = [
#             {"type": "paragraph", "text": "Hi! How can I help you today?"},
#             {"type": "button_group", "buttons": topic_buttons["default"]}
#         ]
#     return JsonResponse({"response": content})