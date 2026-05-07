# Starbucks Offer Completion Predictor

**Live app:** [starbucks-offer-intelligence-gleg.vercel.app](https://starbucks-offer-intelligence-gleg.vercel.app/)

![Starbucks](pictures/starbucks-8.webp)

A machine learning project that predicts whether a Starbucks customer will complete a promotional offer, built on real transaction and customer data from Starbucks.

The live app lets you plug in a customer profile and offer details and get an instant prediction from a trained Random Forest model.

---

## The Problem

Starbucks sends promotional offers through its rewards app. Not every customer acts on them. Some complete the offer, others ignore it, and some spend money anyway without the offer having any effect. The goal here is to figure out, given what we know about a customer and an offer, whether they will actually follow through.

This is a binary classification problem. The target variable is constructed from the raw event log: for each offer a customer received, did a matching "offer completed" event show up for that same customer? That gives us a clean 0/1 label to train on.

---

## The Data

Three JSON files from Starbucks, all in newline-delimited format.

**portfolio.json** describes 10 promotional offers. Each has an offer type (BOGO, discount, or informational), a spend difficulty, a reward, a duration, and the channels it was sent through.

**profile.json** covers about 17,000 customers with age, income, gender, and membership start date. Some records used 118 as a placeholder for missing age and had null income. Those were removed.

**transcript.json** is the full event log with 300,000 rows. Every offer received, viewed, and completed event is here alongside every transaction. The offer ID is stored inside a dict column, and the key name differs between event types, which required a cleaning step before any joins could happen.

---

## Approach

**Data cleaning** removed the placeholder demographics, converted the membership date from an integer to a proper datetime, and standardized the inconsistent key names in the event log.

**Feature engineering** built a flat (customer, offer) table by joining all three files. The final features are age, income, gender, offer type, difficulty, duration, reward, channels (one-hot encoded), and member tenure in days.

**Modelling** compared three approaches on an 80/20 train/test split. Logistic Regression was the baseline, followed by Gradient Boosting and Random Forest. Results on the held-out test set:

| Model | Accuracy |
|---|---|
| Logistic Regression | 77% |
| Gradient Boosting | 79% |
| Random Forest | **82%** |

A PyTorch LSTM was also built to model each customer's event history as a sequence. It scored 69%, lower than Random Forest. Most offers in this dataset were sent at time zero, so most customers had no prior history for the sequence model to work with. The LSTM was reading mostly zeros while Random Forest was using richer static features.

---

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model is not committed to this repo because it exceeds GitHub's file size limit. Run the notebook first to generate `model.pkl`, then start the app.

---

## Tech stack

Python, pandas, scikit-learn, PyTorch, Streamlit, Jupyter
