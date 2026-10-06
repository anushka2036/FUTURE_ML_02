import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Support Ticket Classification",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"


CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.pkl"
CATEGORY_VECTORIZER_PATH = MODELS_DIR / "category_vectorizer.pkl"

PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.pkl"
PRIORITY_VECTORIZER_PATH = MODELS_DIR / "priority_vectorizer.pkl"

STOP_WORDS_PATH = MODELS_DIR / "stop_words.pkl"
PREPROCESSING_CONFIG_PATH = MODELS_DIR / "preprocessing_config.pkl"


@st.cache_resource
def load_models():

    category_model = joblib.load(
        CATEGORY_MODEL_PATH
    )

    category_vectorizer = joblib.load(
        CATEGORY_VECTORIZER_PATH
    )

    priority_model = joblib.load(
        PRIORITY_MODEL_PATH
    )

    priority_vectorizer = joblib.load(
        PRIORITY_VECTORIZER_PATH
    )

    try:
        stop_words = joblib.load(
            STOP_WORDS_PATH
        )

    except Exception:

        try:
            with open(STOP_WORDS_PATH, "rb") as f:
                stop_words = pickle.load(f)

        except Exception:
            stop_words = set()

    try:

        preprocessing_config = joblib.load(
            PREPROCESSING_CONFIG_PATH
        )

    except Exception:

        preprocessing_config = {}

    return (
        category_model,
        category_vectorizer,
        priority_model,
        priority_vectorizer,
        stop_words,
        preprocessing_config
    )


def clean_ticket_text(text, stop_words):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def predict_ticket(
    subject,
    description,
    category_model,
    category_vectorizer,
    priority_model,
    priority_vectorizer,
    stop_words
):

    raw_text = f"{subject} {description}"

    clean_text = clean_ticket_text(
        raw_text,
        stop_words
    )

    category_vector = category_vectorizer.transform(
        [clean_text]
    )

    category_prediction = category_model.predict(
        category_vector
    )[0]

    if hasattr(category_model, "predict_proba"):

        category_probabilities = (
            category_model.predict_proba(
                category_vector
            )[0]
        )

        category_confidence = (
            np.max(category_probabilities) * 100
        )

    else:

        category_confidence = None

    priority_vector = priority_vectorizer.transform(
        [clean_text]
    )

    priority_prediction = priority_model.predict(
        priority_vector
    )[0]

    if hasattr(priority_model, "predict_proba"):

        priority_probabilities = (
            priority_model.predict_proba(
                priority_vector
            )[0]
        )

        priority_confidence = (
            np.max(priority_probabilities) * 100
        )

    else:

        priority_confidence = None

    return {
        "category": category_prediction,
        "category_confidence": category_confidence,
        "priority": priority_prediction,
        "priority_confidence": priority_confidence,
        "clean_text": clean_text
    }


def generate_explanation(
    subject,
    description,
    category,
    priority
):

    text = f"{subject} {description}".lower()

    category_keywords = {

        "Technical issue": [
            "error",
            "crash",
            "bug",
            "login",
            "not working",
            "failed",
            "problem",
            "issue",
            "technical"
        ],

        "Billing inquiry": [
            "billing",
            "payment",
            "charge",
            "charged",
            "invoice",
            "price",
            "cost",
            "bill"
        ],

        "Refund request": [
            "refund",
            "money back",
            "return",
            "reimbursement",
            "refund request"
        ],

        "Cancellation request": [
            "cancel",
            "cancellation",
            "unsubscribe",
            "stop",
            "terminate"
        ],

        "Product inquiry": [
            "product",
            "feature",
            "available",
            "information",
            "details",
            "specification",
            "buy"
        ]
    }

    detected_keywords = []

    if category in category_keywords:

        for keyword in category_keywords[category]:

            if keyword in text:
                detected_keywords.append(keyword)

    if detected_keywords:

        keyword_text = ", ".join(
            detected_keywords[:3]
        )

        explanation = (
            f"The ticket contains terms such as "
            f"'{keyword_text}', which are associated "
            f"with the predicted {category.lower()} category. "
            f"The model also assigned {priority.lower()} priority."
        )

    else:

        explanation = (
            f"The model classified this ticket as "
            f"{category.lower()} with {priority.lower()} priority "
            f"based on patterns learned from the training data."
        )

    return explanation


try:

    (
        category_model,
        category_vectorizer,
        priority_model,
        priority_vectorizer,
        stop_words,
        preprocessing_config
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Unable to load the trained models."
    )

    st.code(
        str(e)
    )

st.title(
    "Customer Support Ticket Classification"
)

st.divider()

st.subheader(
    "Enter Ticket Details"
)

subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Unable to login to my account"
)

description = st.text_area(
    "Ticket Description",
    placeholder=(
        "Example: I have been trying to login to my account "
        "since yesterday, but the application keeps showing "
        "an error message."
    ),
    height=180
)


st.subheader(
    "Example Tickets"
)

example_options = {

    "Technical Issue": (
        "Unable to login",
        "I cannot access my account because the application "
        "keeps showing an error message."
    ),

    "Billing Inquiry": (
        "Unexpected charge on my account",
        "I noticed an unexpected payment charge on my account "
        "and would like to understand the billing details."
    ),

    "Refund Request": (
        "Requesting a refund",
        "I purchased the product recently but would like "
        "to request a refund."
    ),

    "Cancellation Request": (
        "Cancel my subscription",
        "I would like to cancel my current subscription "
        "and stop future charges."
    ),

    "Product Inquiry": (
        "Product information",
        "I would like to know more about the product features "
        "and available options."
    )
}

selected_example = st.selectbox(
    "Select an example ticket",
    ["None"] + list(example_options.keys())
)


if selected_example != "None":

    example_subject, example_description = (
        example_options[selected_example]
    )

    st.info(
        f"Subject: {example_subject}\n\n"
        f"Description: {example_description}"
    )

    if st.button(
        "Use This Example",
        use_container_width=True
    ):

        st.session_state.example_subject = (
            example_subject
        )

        st.session_state.example_description = (
            example_description
        )

        st.rerun()

if "example_subject" in st.session_state:

    subject = st.session_state.example_subject


if "example_description" in st.session_state:

    description = st.session_state.example_description


st.divider()


predict_button = st.button(
    "Predict Ticket",
    type="primary",
    use_container_width=True
)


if predict_button:

    if not subject.strip():

        st.warning(
            "Please enter a ticket subject."
        )

    elif not description.strip():

        st.warning(
            "Please enter a ticket description."
        )

    elif not models_loaded:

        st.error(
            "Models could not be loaded. "
            "Please check the models folder."
        )

    else:

        with st.spinner(
            "Analyzing the support ticket..."
        ):

            result = predict_ticket(
                subject,
                description,
                category_model,
                category_vectorizer,
                priority_model,
                priority_vectorizer,
                stop_words
            )


        st.success(
            "Ticket successfully classified."
        )

        st.subheader(
            "Prediction Results"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "### Ticket Category"
            )

            st.metric(
                label="Predicted Category",
                value=result["category"]
            )

            if result["category_confidence"] is not None:

                st.progress(
                    min(
                        result["category_confidence"] / 100,
                        1.0
                    )
                )

                st.write(
                    f"Confidence: "
                    f"**{result['category_confidence']:.2f}%**"
                )



        with col2:

            st.markdown(
                "### Ticket Priority"
            )

            st.metric(
                label="Predicted Priority",
                value=result["priority"]
            )

            if result["priority_confidence"] is not None:

                st.progress(
                    min(
                        result["priority_confidence"] / 100,
                        1.0
                    )
                )

                st.write(
                    f"Confidence: "
                    f"**{result['priority_confidence']:.2f}%**"
                )


        st.divider()

        st.subheader(
            "Prediction Explanation"
        )

        explanation = generate_explanation(
            subject,
            description,
            result["category"],
            result["priority"]
        )

        st.info(
            explanation
        )

        st.caption(
            "Predictions are generated using the trained "
            "machine learning models. Confidence scores indicate "
            "model probability and do not guarantee correctness."
        )