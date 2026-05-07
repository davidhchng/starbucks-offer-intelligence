import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

st.set_page_config(page_title='Starbucks Offer Completion Predictor', page_icon='☕', layout='wide')

DARK_GREEN = '#1A2F20'
MID_GREEN  = '#243B2C'
ACCENT     = '#CBA258'
GOLD       = '#F5ECD7'
LIGHT_GOLD = '#FAF5EE'

st.markdown(f"""
    <style>
        .stApp {{ background-color: {DARK_GREEN}; }}
        h1, h2, h3, h4, p, li, label, .stMarkdown, .stText {{ color: {GOLD} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ background-color: {MID_GREEN}; border-radius: 8px; }}
        .stTabs [data-baseweb="tab"] {{ color: {GOLD}; }}
        .stTabs [aria-selected="true"] {{ background-color: {ACCENT}; color: white !important; border-radius: 6px; }}
        .stSlider > div > div {{ background-color: {ACCENT}; }}
        .stButton > button {{
            background-color: {ACCENT};
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-size: 1rem;
            border-radius: 20px;
        }}
        .stButton > button:hover {{ background-color: {GOLD}; color: {DARK_GREEN}; }}
        .stSelectbox > div, .stNumberInput > div {{ background-color: {MID_GREEN}; }}
        .metric-box {{
            background-color: {MID_GREEN};
            border-left: 4px solid {ACCENT};
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    portfolio = pd.read_json('data/portfolio.json', lines=True)
    profile   = pd.read_json('data/profile.json',   lines=True)
    transcript = pd.read_json('data/transcript.json', lines=True)

    profile = profile[profile['age'] != 118]
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')

    def standardize_value(d):
        if not isinstance(d, dict):
            return d
        if 'offer id' in d:
            d['offer_id'] = d.pop('offer id')
        return d
    transcript['value'] = transcript['value'].apply(standardize_value)

    offer_events = transcript[transcript['event'] != 'transaction'].copy()
    offer_events['offer_id'] = offer_events['value'].apply(lambda d: d.get('offer_id'))

    completed_set = offer_events[offer_events['event'] == 'offer completed'][['person', 'offer_id']].drop_duplicates()
    completed_set['completed'] = 1

    received = offer_events[offer_events['event'] == 'offer received'][['person', 'offer_id', 'time']].copy()
    df = received.merge(completed_set, on=['person', 'offer_id'], how='left')
    df['completed'] = df['completed'].fillna(0).astype(int)

    df = df.merge(profile, left_on='person', right_on='id', how='left')
    df = df.merge(portfolio, left_on='offer_id', right_on='id', how='left')
    df = df.drop(columns=['id_x', 'id_y'])
    df = df.dropna(subset=['age', 'income', 'gender'])

    df['member_tenure_days'] = (pd.Timestamp('2018-08-01') - df['became_member_on']).dt.days
    df['income_bracket'] = pd.cut(df['income'],
        bins=[0, 40000, 60000, 80000, 120001],
        labels=['Under 40k', '40k to 60k', '60k to 80k', 'Over 80k'])

    return df, portfolio


def styled_fig():
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(MID_GREEN)
    ax.set_facecolor(MID_GREEN)
    ax.tick_params(colors=GOLD)
    ax.xaxis.label.set_color(GOLD)
    ax.yaxis.label.set_color(GOLD)
    ax.title.set_color(GOLD)
    for spine in ax.spines.values():
        spine.set_edgecolor(GOLD)
        spine.set_alpha(0.3)
    return fig, ax


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df, portfolio = load_data()

st.image('pictures/starbucks-8.webp', use_container_width=True)
st.title('☕ Starbucks Offer Completion Predictor')
st.markdown('*A machine learning project that predicts whether a Starbucks customer will complete a promotional offer.*')

tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Data Insights', 'Model', 'Predictor'])


with tab1:
    st.header('What this project does')
    st.markdown("""
    Starbucks runs promotional offers through its rewards app. Some customers complete them, many don't.
    This project uses real customer and transaction data to figure out which customers are most likely
    to follow through on an offer, based on who they are and what kind of offer it is.

    The goal was to build a full machine learning pipeline from raw data to a working prediction tool.
    That means cleaning messy data, engineering useful features, comparing several models, and wrapping
    the best one in an app you can actually use.
    """)

    st.header('The data')
    st.markdown("""
    The dataset comes from Starbucks and contains three files covering about 17,000 customers and
    300,000 transaction events over a test period.

    **Customers** (profile.json) includes age, income, gender, and how long they have been a member.
    Some records had placeholder values like age 118 and null income, which we identified and removed.

    **Offers** (portfolio.json) describes 10 promotional offers across three types. BOGO (buy one get one)
    and discount offers have a spend target and a reward. Informational offers are just notifications with
    no reward and no completion target.

    **Events** (transcript.json) is the full event log. Every time a customer received, viewed, or
    completed an offer, or made a transaction, it was recorded here with a timestamp. This is the most
    complex file because the offer ID is buried inside a dict column and the key name is inconsistent
    depending on the event type.
    """)

    st.header('How we built the target variable')
    st.markdown("""
    The raw data has no "completed" column. We had to construct it ourselves.

    For each offer a customer received, we checked whether a matching "offer completed" event existed
    for that same customer and offer ID. If it did, the label is 1. If not, it is 0. This gave us a
    binary classification target across 66,501 (customer, offer) pairs, with roughly a 50/50 class split
    after filtering out customers with missing demographics.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-box"><h3>66,501</h3><p>labeled (customer, offer) pairs</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h3>14,825</h3><p>customers with valid demographics</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h3>82%</h3><p>accuracy on held-out test data</p></div>', unsafe_allow_html=True)


with tab2:
    st.header('What the data tells us')
    st.markdown('These charts show completion patterns across different customer segments and offer types.')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Completion rate by offer type')
        rate_by_type = df.groupby('offer_type')['completed'].mean().sort_values()
        fig, ax = styled_fig()
        bars = ax.barh(rate_by_type.index, rate_by_type.values, color=ACCENT)
        ax.bar_label(bars, labels=[f'{v:.0%}' for v in rate_by_type.values], label_type='edge', padding=4, color=GOLD)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel('Completion rate')
        ax.set_title('Completion Rate by Offer Type')
        st.pyplot(fig)
        st.markdown('Informational offers show no completions because they have no spend target. BOGO and discount offers perform similarly.')

    with col2:
        st.subheader('Completion rate by gender')
        rate_by_gender = df.groupby('gender')['completed'].mean()
        fig, ax = styled_fig()
        bars = ax.bar(rate_by_gender.index, rate_by_gender.values, color=[ACCENT, GOLD, MID_GREEN], edgecolor=GOLD, linewidth=0.5)
        ax.bar_label(bars, labels=[f'{v:.0%}' for v in rate_by_gender.values], padding=4, color=GOLD)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel('Completion rate')
        ax.set_title('Completion Rate by Gender')
        st.pyplot(fig)
        st.markdown('Female customers complete offers at a noticeably higher rate than male customers.')

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Completion rate by income bracket')
        rate_by_income = df.groupby('income_bracket', observed=True)['completed'].mean()
        fig, ax = styled_fig()
        bars = ax.bar(rate_by_income.index, rate_by_income.values, color=ACCENT, edgecolor=GOLD, linewidth=0.5)
        ax.bar_label(bars, labels=[f'{v:.0%}' for v in rate_by_income.values], padding=4, color=GOLD)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel('Completion rate')
        ax.set_title('Completion Rate by Income')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        st.markdown('Higher income customers complete offers more often, but the difference is modest.')

    with col4:
        st.subheader('Completion rate by member tenure')
        df['tenure_bin'] = pd.cut(df['member_tenure_days'], bins=5)
        rate_by_tenure = df.groupby('tenure_bin', observed=True)['completed'].mean()
        fig, ax = styled_fig()
        ax.plot(range(len(rate_by_tenure)), rate_by_tenure.values, color=GOLD, marker='o', linewidth=2)
        ax.set_xticks(range(len(rate_by_tenure)))
        ax.set_xticklabels(['Newest', '', '', '', 'Longest'], color=GOLD)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylabel('Completion rate')
        ax.set_title('Completion Rate by Member Tenure')
        st.pyplot(fig)
        st.markdown('Longer-standing members are more likely to complete offers, which makes intuitive sense.')


with tab3:
    st.header('How the model works')
    st.markdown("""
    We trained three different models and compared them on accuracy and F1 score on a held-out 20% test set.
    F1 is reported for the "completed" class since that is the outcome we care most about predicting correctly.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Model comparison')
        results = {
            'Logistic Regression': 0.77,
            'Gradient Boosting':   0.79,
            'Random Forest':       0.82,
        }
        fig, ax = styled_fig()
        colors = [ACCENT, ACCENT, GOLD]
        bars = ax.barh(list(results.keys()), list(results.values()), color=colors)
        ax.bar_label(bars, labels=[f'{v:.0%}' for v in results.values()], label_type='edge', padding=4, color=GOLD)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel('Accuracy')
        ax.set_xlim(0, 1)
        ax.set_title('Model Accuracy on Test Set')
        st.pyplot(fig)

    with col2:
        st.subheader('Why Random Forest won')
        st.markdown("""
        Logistic Regression draws a straight boundary through the feature space to separate completions
        from non-completions. That is fast and interpretable but misses complex interactions between
        features.

        Random Forest builds 100 decision trees, each trained on a random slice of the data, then takes
        a majority vote. Because each tree learns slightly different patterns, the ensemble is much harder
        to fool by noise and captures non-linear relationships that a single model would miss.

        Gradient Boosting builds trees sequentially, with each new tree correcting the mistakes of
        the previous ones. It scored between the two, which is common when the dataset is clean and
        moderately sized.
        """)

    st.subheader('Feature importance')
    st.markdown('These are the features the Random Forest relied on most when making predictions.')
    importances = model.feature_importances_
    feature_names = ['gender', 'age', 'income', 'reward', 'difficulty', 'duration',
                     'offer_type', 'member_tenure_days', 'channel_email',
                     'channel_web', 'channel_mobile', 'channel_social']
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance')
    fig, ax = styled_fig()
    ax.barh(imp_df['feature'], imp_df['importance'], color=ACCENT)
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance')
    fig.set_size_inches(9, 5)
    st.pyplot(fig)

    st.subheader('We also tried a PyTorch LSTM')
    st.markdown("""
    Beyond the scikit-learn models, we built an LSTM (Long Short-Term Memory) neural network in PyTorch.
    The idea was to model each customer's sequence of events before receiving an offer, rather than
    treating every row independently. An LSTM reads a sequence one step at a time and maintains a
    running memory of what it has seen, which lets earlier events influence the final prediction.

    It scored 69% accuracy, lower than Random Forest. The main reason is that most offers in this
    dataset were sent at time zero, meaning most customers had no prior event history for the model
    to learn from. The LSTM was reading sequences of zeros. Random Forest won by leaning on richer
    static features like income, age, and offer type, which carry more signal here.
    """)


with tab4:
    st.header('Try the predictor')
    st.markdown('Fill in a customer profile and offer details to get a completion prediction from the Random Forest model.')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Customer')
        age = st.slider('Age', 18, 100, 40)
        income = st.number_input('Annual Income', min_value=30000, max_value=120000, value=60000, step=1000)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        member_tenure_days = st.slider('Member Tenure (days)', 0, 2000, 500)

    with col2:
        st.subheader('Offer')
        offer_type = st.selectbox('Offer Type', ['BOGO', 'Discount', 'Informational'])
        difficulty = st.slider('Difficulty (min spend)', 0, 20, 5)
        duration = st.slider('Duration (days)', 3, 10, 7)
        reward = st.slider('Reward', 0, 10, 5)

    with col3:
        st.subheader('Channels')
        channel_email  = st.checkbox('Email')
        channel_web    = st.checkbox('Web')
        channel_mobile = st.checkbox('Mobile')
        channel_social = st.checkbox('Social')

    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    offer_map  = {'BOGO': 0, 'Discount': 1, 'Informational': 2}

    features = np.array([[
        gender_map[gender], age, income, reward, difficulty, duration,
        offer_map[offer_type], member_tenure_days,
        int(channel_email), int(channel_web), int(channel_mobile), int(channel_social)
    ]])

    st.markdown('')
    if st.button('Predict'):
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        if pred == 1:
            st.success(f'This customer is likely to complete the offer. Predicted probability: {prob:.0%}')
        else:
            st.error(f'This customer is unlikely to complete the offer. Predicted probability: {prob:.0%}')
