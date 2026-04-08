import os
import pickle
import random
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

user_name = None

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("what's up", "greeting"),
    ("yo", "greeting"),

    ("how are you", "status"),
    ("how are you doing", "status"),
    ("are you okay", "status"),
    ("how is it going", "status"),
    ("how do you feel", "status"),

    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("quit", "farewell"),
    ("exit", "farewell"),

    ("help", "help"),
    ("what can you do", "help"),
    ("show me your features", "help"),
    ("what are your commands", "help"),
    ("can you help me", "help"),

    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("thx", "thanks"),
    ("appreciate it", "thanks"),
    ("thanks a lot", "thanks"),

    ("who made you", "creator"),
    ("who created you", "creator"),
    ("who built you", "creator"),
    ("who is your creator", "creator"),
    ("who made this bot", "creator")
]

responses = {
    "greeting": [
        "Hey. How can I help you?",
        "Hello there.",
        "Hi."
    ],
    "status": [
        "I'm running smoothly.",
        "All systems operational.",
        "Doing exactly what I was built for."
    ],
    "farewell": [
        "Goodbye.",
        "See you later.",
        "Take care."
    ],
    "help": [
        "I can respond to greetings, remember your name, tell the time and date, answer status questions, identify my creator, and handle combined messages."
    ],
    "thanks": [
        "You're welcome.",
        "No problem.",
        "Glad to help."
    ],
    "creator": [
        "I was built as a Python chatbot project."
    ]
}

default_responses = [
    "I don't understand that yet.",
    "Can you rephrase that?",
    "I'm not sure what you mean."
]


def train_model():
    texts = [text for text, label in training_data]
    labels = [label for text, label in training_data]

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(x_train, labels)

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_FILE, "wb") as file:
        pickle.dump(vectorizer, file)

    return model, vectorizer


def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)

        with open(VECTORIZER_FILE, "rb") as file:
            vectorizer = pickle.load(file)

        return model, vectorizer

    return train_model()


model, vectorizer = load_model()


def extract_name(user_input):
    if "my name is" in user_input:
        return user_input.split("my name is")[-1].strip().title()
    return None


def get_name_response():
    if user_name:
        return f"Your name is {user_name}."
    return "You haven't told me your name yet."


def get_time_response():
    current_time = datetime.now().strftime("%H:%M")
    return f"The current time is {current_time}."


def get_date_response():
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"Today's date is {current_date}."


def get_time_and_date_response():
    current_time = datetime.now().strftime("%H:%M")
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"The current time is {current_time}, and today's date is {current_date}."


def predict_intent(user_input):
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)[0]
    confidence = model.predict_proba(transformed_input).max()

    if confidence < 0.30:
        return None

    return prediction


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def get_greeting_response():
    response = random.choice(responses["greeting"])

    if user_name:
        return f"{response} {user_name}."

    return response


def get_status_response():
    return random.choice(responses["status"])


def get_help_response():
    return random.choice(responses["help"])


def get_creator_response():
    return random.choice(responses["creator"])


def get_thanks_response():
    return random.choice(responses["thanks"])


def get_farewell_response():
    return random.choice(responses["farewell"])


def get_response(user_input):
    global user_name
    user_input = user_input.lower().strip()

    extracted_name = extract_name(user_input)
    if extracted_name:
        user_name = extracted_name
        return f"Nice to meet you, {user_name}."

    asks_for_name = "what is my name" in user_input
    asks_for_time = "time" in user_input
    asks_for_date = "date" in user_input or "today" in user_input

    has_greeting = contains_any(
        user_input,
        ["hello", "hi", "hey", "good morning", "good evening", "what's up", "yo"]
    )

    has_status = contains_any(
        user_input,
        ["how are you", "how are you doing", "are you okay", "how is it going", "how do you feel"]
    )

    has_help = contains_any(
        user_input,
        ["help", "what can you do", "show me your features", "what are your commands", "can you help me"]
    )

    has_thanks = contains_any(
        user_input,
        ["thanks", "thank you", "thx", "appreciate it", "thanks a lot"]
    )

    has_creator = contains_any(
        user_input,
        ["who made you", "who created you", "who built you", "who is your creator", "who made this bot"]
    )

    has_farewell = contains_any(
        user_input,
        ["bye", "goodbye", "see you", "exit", "quit"]
    )

    response_parts = []

    if has_greeting:
        response_parts.append(get_greeting_response())

    if has_status:
        response_parts.append(get_status_response())

    if asks_for_name:
        response_parts.append(get_name_response())

    if asks_for_time and asks_for_date:
        response_parts.append(get_time_and_date_response())
    elif asks_for_date:
        response_parts.append(get_date_response())
    elif asks_for_time:
        response_parts.append(get_time_response())

    if has_help:
        response_parts.append(get_help_response())

    if has_creator:
        response_parts.append(get_creator_response())

    if has_thanks:
        response_parts.append(get_thanks_response())

    if has_farewell:
        response_parts.append(get_farewell_response())

    if response_parts:
        return " ".join(response_parts)

    intent = predict_intent(user_input)

    if intent == "greeting":
        return get_greeting_response()

    if intent == "status":
        return get_status_response()

    if intent == "help":
        return get_help_response()

    if intent == "creator":
        return get_creator_response()

    if intent == "thanks":
        return get_thanks_response()

    if intent == "farewell":
        return get_farewell_response()

    return random.choice(default_responses)