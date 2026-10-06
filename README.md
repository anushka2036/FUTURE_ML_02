# Customer Support Ticket Classification & Prioritization

A machine learning project that automatically classifies customer support tickets and assigns an appropriate priority level based on the ticket subject and description.

The project is designed to help support teams organize incoming tickets, identify their type, and prioritize them for faster handling.

## Project Overview

Customer support teams receive a large number of tickets every day. Manually categorizing and prioritizing these tickets can be time-consuming.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically predict:

* **Ticket Category**

  * Technical Issue
  * Billing Inquiry
  * Refund Request
  * Cancellation Request
  * Product Inquiry

* **Ticket Priority**

  * High
  * Medium
  * Low

The trained models analyze the textual information provided in the ticket and generate predictions through a simple Streamlit interface.

## Key Features

* Text preprocessing and cleaning
* TF-IDF based text representation
* Machine learning classification
* Automatic ticket category prediction
* Automatic priority prediction
* Prediction confidence scores
* Simple prediction explanation
* Interactive Streamlit web application
* Reusable trained model files

## Technology Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **TF-IDF**
* **Logistic Regression**
* **Joblib / Pickle**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

## Project Workflow

```text
Customer Support Ticket
          |
          v
   Text Preprocessing
          |
          v
    TF-IDF Vectorization
          |
          v
   Machine Learning Model
       /          \
      v            v
  Category      Priority
 Prediction     Prediction
       \          /
        \        /
         v      v
       Streamlit
       Interface
```

## Project Structure

```text
support-ticket-classification/
│
├── data/
│   └── customer_support_tickets.csv
│
├── models/
│   ├── category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── priority_model.pkl
│   ├── priority_vectorizer.pkl
│   ├── stop_words.pkl
│   └── preprocessing_config.pkl
│
├── notebooks/
│   └── Support_Ticket_Classification.ipynb
│
├── webapp/
│   └── main.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/support-ticket-classification.git
cd support-ticket-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
cd streamlit_app
streamlit run app.py
```

The application will open in your browser.

## Example

**Input**

```text
Subject: Unable to login to my account

Description:
I cannot access my account because the application
keeps showing an error message.
```

**Output**

```text
Predicted Category: Technical Issue
Predicted Priority: High
```

## Machine Learning Approach

The project follows a standard NLP classification workflow:

1. Load and explore the customer support ticket dataset.
2. Clean and preprocess ticket text.
3. Combine relevant textual information from the ticket.
4. Convert text into numerical features using **TF-IDF**.
5. Train classification models for ticket category and priority.
6. Evaluate the trained models using classification metrics.
7. Save the trained models and vectorizers.
8. Integrate the models into a Streamlit application.

## Future Improvements

The project can be further enhanced with:

* Advanced transformer-based NLP models
* Better handling of ambiguous or noisy tickets
* Explainable AI techniques such as SHAP
* Real-time ticket management integration
* Automatic routing to support teams
* Continuous model retraining using newly resolved tickets

## Conclusion

This project demonstrates how **NLP and machine learning can be applied to customer support automation**. It provides a practical workflow from data preprocessing and model development to deployment through an interactive Streamlit application.

The project was developed as part of the **Future Interns Machine Learning Internship**.
