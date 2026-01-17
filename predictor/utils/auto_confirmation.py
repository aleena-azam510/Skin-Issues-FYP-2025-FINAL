import json
import os

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/questions.json"
)

def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_confirm_conditions(user_answers: dict):
    """
    user_answers example:
    {
        "dark_crescents": true,
        "hollows_under_eyes": true,
        "puffiness_under_eyes": false
    }

    Returns:
    confirmed_conditions: list[str]
    condition_confidence: dict[str, float]
    """

    questions_data = load_questions()

    confirmed_conditions = []
    condition_confidence = {}

    for condition, block in questions_data.items():

        confirm_symptoms = block["confirm_if_yes"]
        min_required = block["min_yes_required"]

        yes_count = 0
        total_considered = len(confirm_symptoms)

        for symptom in confirm_symptoms:
            if user_answers.get(symptom) is True:
                yes_count += 1

        # ✅ Condition confirmed
        if yes_count >= min_required:
            confirmed_conditions.append(condition)

            # 📊 Confidence calculation (simple but effective)
            confidence = round((yes_count / total_considered) * 100, 2)
            condition_confidence[condition] = confidence

    return confirmed_conditions, condition_confidence
