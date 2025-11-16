# Emotional Weather Prediction AI - Kaggle-Capstone Project
# Author:Saksham Gupta
# Description: Predict a city's collective emotional state for the next day using weather, events,
# trends and social sentiment signals. This notebook includes synthetic-data generator, EDA,
# baseline models (XGBoost), sequence models (LSTM), evaluation, visualization, and
# instructions to plug-in real-world data sources.

# %% [markdown]
# # Emotional Weather Prediction AI
# This code builds an end-to-end pipeline to forecast the *collective emotional state* of a city
# for the next day using weather data, local events, search-trends, and social sentiment.
# The goal is to produce a Kaggle-quality submission: reproducible, well-documented and robust.

# %%
# --- 0) ENV & IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import os
import json
from datetime import datetime, timedelta

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Gradient boosting
try:
    import xgboost as xgb
except Exception:
    xgb = None

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducible
np.random.seed(42)
if tf.__version__:
    tf.random.set_seed(42)

# %% [markdown]
# ## 1) Problem framing
# Predict a city's "collective mood" for tomorrow into 5 classes: ['Very Negative','Negative','Neutral','Positive','Very Positive'].
# This is a multi-class classification problem using tabular + time-series signals.

# %% [markdown]
# ## 2) Synthetic dataset generator (replace with real inputs on Kaggle)
# The generator simulates day-by-day values for: temperature, humidity, precipitation, wind_speed,
# event_strength (how big events are that day), google_trend_score (0-100), twitter_sentiment (-1 to 1),
# holiday_flag, previous_mood (lag features), and target_mood (categorical label for next day).

# %%
def generate_emotional_weather_data(days=720, start_date='2024-01-01', city_name='Metropolis'):
    dates = pd.date_range(start=start_date, periods=days)
    rows = []

    # seasonal baseline for temperature (sinusoidal)
    for i, d in enumerate(dates):
        day_of_year = d.timetuple().tm_yday
        # temp follows a yearly sine wave plus noise
        temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year / 365.0)) + np.random.normal(0, 3)
        humidity = np.clip(50 + 20 * np.cos(2 * np.pi * (day_of_year / 365.0)) + np.random.normal(0, 6), 10, 100)
        precipitation = max(0, np.random.exponential(0.5) - (np.cos(day_of_year/20) * 0.2))
        wind_speed = np.clip(np.random.normal(8, 3), 0, 30)

        # events: higher on weekends and some random spikes for festivals
        weekday = d.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        # simulate occasional big events
        festival = 1 if np.random.rand() < 0.03 else 0
        event_strength = is_weekend * np.random.uniform(0, 0.6) + festival * np.random.uniform(0.6, 1.0)

        # trends: smooth signal with occasional spikes
        base_trend = 40 + 10 * np.sin(day_of_year/30.0) + np.random.normal(0,4)
        trend_spike = np.random.choice([0, 1], p=[0.97, 0.03]) * np.random.uniform(20,60)
        google_trend = np.clip(base_trend + trend_spike, 0, 100)

        # twitter sentiment: follow trends and events slightly
        twitter_sentiment = np.clip((google_trend - 50)/60 + event_strength*0.2 + np.random.normal(0,0.2), -1, 1)

        # previous day's mood placeholder (we will compute target using heuristics below)
        rows.append({
            'date': d,
            'city': city_name,
            'temp_c': round(temp,2),
            'humidity': round(humidity,2),
            'precip_mm': round(precipitation,2),
            'wind_kmh': round(wind_speed,2),
            'event_strength': round(event_strength,3),
            'google_trend': round(google_trend,1),
            'twitter_sentiment': round(twitter_sentiment,3),
            'is_weekend': is_weekend,
            'festival': festival,
            'holiday': 1 if np.random.rand() < 0.02 else 0
        })

    df = pd.DataFrame(rows)

    # compute a continuous "mood_score" for each day using a heuristic mixing signals
    # higher temp (comfortable), low precipitation, low humidity, positive twitter sentiment,
    # strong events (positive or negative) and high google_trend spikes influence mood.
    def compute_mood_score(row):
        score = 50.0
        # temperature comfort: 18-26 ideal
        temp = row['temp_c']
        temp_effect = -abs(temp - 22) + 6  # peak at 22
        # precipitation negative effect
        precip_effect = -row['precip_mm'] * 2.0
        # humidity mild negative if too high
        humidity_effect = -max(0, (row['humidity'] - 70)/5)
        # twitter sentiment scaled
        tw = row['twitter_sentiment'] * 12.0
        # events: large events can be positive (celebrations) or negative (stress), add noise
        ev = row['event_strength'] * np.random.uniform(-6, 6)
        # trends: big spikes often create mixed but energetic feelings
        tr = (row['google_trend'] - 50)/6.0
        base = score + temp_effect + precip_effect + humidity_effect + tw + ev + tr
        # clamp
        return np.clip(base + np.random.normal(0,3), 0, 100)

    df['mood_score'] = df.apply(compute_mood_score, axis=1)

    # Convert mood_score to categorical next-day label (we will predict next day's mood)
    # Bins: 0-20 Very Negative, 20-40 Negative, 40-60 Neutral, 60-80 Positive, 80-100 Very Positive
    bins = [0,20,40,60,80,100]
    labels = ['Very Negative','Negative','Neutral','Positive','Very Positive']
    df['mood_category_today'] = pd.cut(df['mood_score'], bins=bins, labels=labels, include_lowest=True)

    # target: tomorrow's mood category (shifted)
    df['target_mood'] = df['mood_category_today'].shift(-1)

    # drop last row with NaN target
    df = df[:-1].reset_index(drop=True)

    return df

# generate data
print('Generating synthetic dataset...')
df = generate_emotional_weather_data(days=900, start_date='2023-01-01', city_name='MetroCity')
print('Shape:', df.shape)

# quick peek
print(df.head())

# %% [markdown]
# ## 3) Quick EDA

# %%
plt.figure(figsize=(12,4))
plt.plot(df['date'], df['mood_score'], label='mood_score')
plt.title('Synthetic mood score over time')
plt.xlabel('Date')
plt.ylabel('Mood Score')
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

# distribution of target
plt.figure(figsize=(6,4))
sns.countplot(y=df['target_mood'], order=['Very Positive','Positive','Neutral','Negative','Very Negative'])
plt.title('Target mood distribution')
plt.show()

# correlations
plt.figure(figsize=(10,6))
sns.heatmap(df[['temp_c','humidity','precip_mm','wind_kmh','google_trend','twitter_sentiment','event_strength','mood_score']].corr(), annot=True, fmt='.2f')
plt.title('Feature Correlations')
plt.show()

# %% [markdown]
# ## 4) Feature engineering
# Create lag features, rolling statistics, weekday features and encode target.

# %%
work = df.copy()
# lags for mood_score
for lag in [1,2,3,7]:
    work[f'mood_lag_{lag}'] = work['mood_score'].shift(lag)
# rolling means
work['trend_roll_7'] = work['google_trend'].rolling(7, min_periods=1).mean()
work['sent_roll_7'] = work['twitter_sentiment'].rolling(7, min_periods=1).mean()
work['precip_roll_7'] = work['precip_mm'].rolling(7, min_periods=1).mean()

# weekday
work['weekday'] = work['date'].dt.weekday

# fill na from shifts
work.fillna(method='bfill', inplace=True)

# encode target
le = LabelEncoder()
work['target_label'] = le.fit_transform(work['target_mood'])
print('Classes:', list(le.classes_))

# select features
feature_cols = ['temp_c','humidity','precip_mm','wind_kmh','google_trend','twitter_sentiment','event_strength','is_weekend','holiday',
                'mood_lag_1','mood_lag_2','mood_lag_3','mood_lag_7','trend_roll_7','sent_roll_7','precip_roll_7','weekday']
X = work[feature_cols].copy()
y = work['target_label'].copy()

# train-test split time-aware: last 20% as test
split_idx = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print('Train/Test sizes:', X_train.shape, X_test.shape)

# %% [markdown]
# ## 5) Baseline model: XGBoost (tabular)
# Use XGBoost as a strong baseline for tabular classification. If xgboost is unavailable, sklearn's RandomForest could be used instead.

# %%
if xgb is None:
    print('XGBoost not installed. Skipping XGBoost baseline. Install xgboost via pip for better results.')
else:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'eval_metric': 'mlogloss',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=500, early_stopping_rounds=25, evals=watchlist, verbose_eval=50)

    # predict
    probs = bst.predict(dtest)
    preds = np.argmax(probs, axis=1)
    print('XGBoost Test Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - XGBoost')
    plt.show()

# %% [markdown]
# ## 6) Sequence model (LSTM) using recent windows
# We'll create sequences of length SEQ_LEN days and predict next-day mood category as classification.

# %%
SEQ_LEN = 14  # number of days in input window
# prepare scaled features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# helper to build sequences
def make_seq_dataset(X_arr, y_arr, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i:i+seq_len])
        ys.append(y_arr[i+seq_len])  # label of day after the window
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_seq_dataset(X_scaled, y.values, seq_len=SEQ_LEN)
print('Seq shapes:', X_seq.shape, y_seq.shape)

# train-test split consistent with earlier split index
split_seq = int(0.8 * len(X_seq))
X_seq_tr, X_seq_te = X_seq[:split_seq], X_seq[split_seq:]
y_seq_tr, y_seq_te = y_seq[:split_seq], y_seq[split_seq:]

# build model
num_classes = len(le.classes_)
input_shape = X_seq_tr.shape[1:]
model = Sequential([
    LSTM(128, input_shape=input_shape, return_sequences=True),
    Dropout(0.25),
    LSTM(64, return_sequences=False),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
if not os.path.exists('models'):
    os.makedirs('models')
es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
mc = ModelCheckpoint('models/best_emotion_lstm.h5', save_best_only=True, monitor='val_loss')

history = model.fit(X_seq_tr, y_seq_tr, validation_split=0.15, epochs=80, batch_size=32, callbacks=[es,mc])

# evaluate
loss, acc = model.evaluate(X_seq_te, y_seq_te)
print('LSTM Test Acc:', acc)

# predictions
y_pred = np.argmax(model.predict(X_seq_te), axis=1)
print(classification_report(y_seq_te, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_seq_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - LSTM')
plt.show()

# %% [markdown]
# ## 7) Feature importance (from XGBoost) & Interpretation

# %%
if xgb is not None:
    fig, ax = plt.subplots(figsize=(10,6))
    xgb.plot_importance(bst, max_num_features=20, ax=ax)
    plt.title('Feature importance - XGBoost')
    plt.show()
else:
    print('XGBoost not available to show feature importances. Use sklearn or shap for interpretability.')

# %% [markdown]
# ## 8) Inference wrapper and simple dashboard visualizations

# %%
# inference: given last SEQ_LEN days, predict next-day mood category

def predict_next_day_from_df(window_df, model_type='lstm'):
    """
    window_df: dataframe of last SEQ_LEN days containing feature_cols
    model_type: 'lstm' or 'xgb'
    returns: predicted label (string) and probabilities
    """
    assert len(window_df) >= SEQ_LEN, f"Need at least {SEQ_LEN} rows"
    w = window_df.tail(SEQ_LEN)[feature_cols].copy()
    if model_type == 'lstm':
        arr = scaler.transform(w)
        arr = arr.reshape(1, SEQ_LEN, arr.shape[1])
        probs = model.predict(arr)[0]
        pred = np.argmax(probs)
        return le.inverse_transform([pred])[0], probs
    elif model_type == 'xgb' and xgb is not None:
        d = xgb.DMatrix(w)
        probs = bst.predict(d)[-1]
        pred = np.argmax(probs)
        return le.inverse_transform([pred])[0], probs
    else:
        raise ValueError('Invalid model_type or missing model')

# example
example_window = work.tail(SEQ_LEN)
label, probs = predict_next_day_from_df(example_window, model_type='lstm')
print('Predicted next-day mood:', label)

# %% [markdown]
# ## 9) How to replace synthetic inputs with real data (instructions)
# 1. Weather: Use APIs like Open-Meteo, NOAA, or local meteorological datasets to fetch historical weather (temp, humidity, precip, wind).
# 2. Events: Create an events calendar for the city — scrape municipal event pages, festival dates, and public calendars. Convert to an `event_strength` metric.
# 3. Google Trends: Use PyTrends to obtain relative search interest for top mood-related keywords in the city/region.
# 4. Twitter/Reddit sentiment: Use Twitter API (v2) or Reddit pushshift to pull local posts and compute daily sentiment averages using a sentiment model (e.g., VADER for English or transformer-based sentiment model for better quality).
# 5. Holidays: Use country-specific holiday libraries (e.g., `holidays` in Python) to flag holidays.
# 6. Merge all sources by date and create similar engineered features and lags.

# %% [markdown]
# ## 10) Kaggle submission & packaging tips
# - Add a clear README explaining the signals and assumptions.
# - Include `requirements.txt` with pinned versions: pandas, numpy, sklearn, xgboost, tensorflow, seaborn, matplotlib.
# - Keep synthetic generator as fallback but show how to replace with real data.
# - Provide an inference notebook that only contains loading models and predicting for a new city/date.

# %%
# Save example dataset and models
work.to_csv('outputs/emotional_weather_synthetic.csv', index=False)
print('Saved outputs/emotional_weather_synthetic.csv')
if xgb is not None:
    bst.save_model('models/xgb_emotion.model')
    print('Saved XGBoost model')
model.save('models/emotion_lstm.h5')
print('Saved LSTM model')
