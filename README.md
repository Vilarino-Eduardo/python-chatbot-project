# AI Chatbot (Flask + Machine Learning)

A smart chatbot built with Python and Flask, featuring machine learning-based intent detection, conversation memory, and support for composed messages.

## Features

- ML-based intent classification (scikit-learn)
- TF-IDF vectorization
- Logistic Regression model
- Model persistence (saved and loaded with pickle)
- Remembers user's name
- Handles composed messages (multiple intents in one input)
- Responds to:
  - greetings
  - status questions
  - time & date
  - help commands
  - creator questions
  - thanks & farewells
- Web interface (Flask)
- Typing indicator (UX improvement)

## Model Details

- Algorithm: Logistic Regression
- Vectorization: TF-IDF
- Confidence threshold: 0.30

## Example Inputs

- `hello`
- `my name is Eduardo`
- `what is my name`
- `what time is it`
- `what is the date`
- `hello, what time is it?`
- `thanks, bye`
- `who made you`

## Tech Stack

- Python
- Flask
- scikit-learn
- HTML / CSS / JavaScript

## How to Run

```bash
git clone https://github.com/your-username/python-chatbot-project.git
cd python-chatbot-project

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

## Project Structure

```
python-chatbot-project/
│── app.py
│── chatbot.py
│── model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
│── .gitignore
│── templates/
    └── index.html
```

## Future Improvements

- Better NLP (NLTK / spaCy)
- Deep learning model (transformers)
- Chat history storage (database)
- User authentication
- API deployment

## Author

Built as part of a Python + AI portfolio project.