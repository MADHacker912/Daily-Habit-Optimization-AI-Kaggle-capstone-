import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HabitOptimizerAI:
    def __init__(self, db_path='habits.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()
        self.model = None
        self.train_model()

    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_name TEXT NOT NULL,
            date TEXT NOT NULL,
            completed INTEGER NOT NULL,  -- 1 for yes, 0 for no
            rating INTEGER,  -- 1-5 scale
            notes TEXT
        )
        '''
        self.conn.execute(query)
        self.conn.commit()

    def log_habit(self, habit_name, completed, rating=None, notes=''):
        date = datetime.now().strftime('%Y-%m-%d')
        query = 'INSERT INTO habits (habit_name, date, completed, rating, notes) VALUES (?, ?, ?, ?, ?)'
        self.conn.execute(query, (habit_name, date, completed, rating, notes))
        self.conn.commit()
        logging.info(f"Logged habit: {habit_name}")

    def get_habits_data(self):
        query = 'SELECT * FROM habits'
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def analyze_habits(self):
        df = self.get_habits_data()
        if df.empty:
            print("No habits data available.")
            return

        # Calculate streaks
        df['streak'] = df.groupby('habit_name')['completed'].cumsum() - df.groupby('habit_name')['completed'].cumsum().shift(1).fillna(0)
        streaks = df.groupby('habit_name')['streak'].max()

        # Visualize completion rates
        completion_rate = df.groupby('habit_name')['completed'].mean()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=completion_rate.index, y=completion_rate.values)
        plt.title('Habit Completion Rates')
        plt.ylabel('Completion Rate')
        plt.show()

        # Correlation analysis (if multiple habits)
        if len(df['habit_name'].unique()) > 1:
            pivot = df.pivot_table(index='date', columns='habit_name', values='completed', fill_value=0)
            corr = pivot.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Habit Correlations')
            plt.show()

        print("Streaks:", streaks.to_dict())

    def train_model(self):
        df = self.get_habits_data()
        if df.empty or len(df) < 10:
            logging.warning("Not enough data to train model.")
            return

        # Prepare features: day of week, previous completions
        df['day_of_week'] = df['date'].dt.dayofweek
        df['prev_completed'] = df.groupby('habit_name')['completed'].shift(1).fillna(0)
        features = ['day_of_week', 'prev_completed']
        target = 'completed'

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Model trained with accuracy: {accuracy:.2f}")

    def predict_success(self, habit_name, day_of_week):
        if not self.model:
            print("Model not trained yet.")
            return None
        prev_completed = self.get_habits_data().query(f"habit_name == '{habit_name}'")['completed'].iloc[-1] if not self.get_habits_data().empty else 0
        prediction = self.model.predict([[day_of_week, prev_completed]])
        return prediction[0]

    def suggest_optimization(self, habit_name):
        df = self.get_habits_data().query(f"habit_name == '{habit_name}'")
        if df.empty:
            print("No data for this habit.")
            return

        avg_rating = df['rating'].mean()
        completion_rate = df['completed'].mean()
        suggestions = []

        if completion_rate < 0.5:
            suggestions.append("Consider setting reminders or pairing with an enjoyable activity.")
        if avg_rating < 3:
            suggestions.append("Reflect on why the habit feels low-rated and adjust accordingly.")
        if self.model:
            today = datetime.now().weekday()
            pred = self.predict_success(habit_name, today)
            if pred == 0:
                suggestions.append("Based on AI prediction, today might be challenging; prepare extra motivation.")

        print("Optimization Suggestions:")
        for s in suggestions:
            print(f"- {s}")

    def close(self):
        self.conn.close()

# CLI Interface
def main():
    ai = HabitOptimizerAI()
    while True:
        print("\nDaily Habit Optimizer AI")
        print("1. Log a habit")
        print("2. Analyze habits")
        print("3. Get optimization suggestions")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            habit_name = input("Habit name: ")
            completed = int(input("Completed (1 for yes, 0 for no): "))
            rating = int(input("Rating (1-5, optional): ") or 0)
            notes = input("Notes: ")
            ai.log_habit(habit_name, completed, rating if rating else None, notes)
        elif choice == '2':
            ai.analyze_habits()
        elif choice == '3':
            habit_name = input("Habit name: ")
            ai.suggest_optimization(habit_name)
        elif choice == '4':
            ai.close()
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
