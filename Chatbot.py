import nltk
import pandas as pd
import numpy as np
import requests
import json
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance
import sqlite3
import re
from datetime import datetime, timedelta
import dateparser 
from sample_flights import flights

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('maxent_ne_chunker_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# User informations
user_name = None


class FlightBookingSystem:
    def __init__(self,user_name):  
        self.user_name = user_name
        self.step = 0 if not user_name else 1
        self.origin = None
        self.destination = None
        self.date = None
        self.flight_selected = False

    def handle_input(self, user_input):
        try:
            if self.step == 0:
                return self.ask_name(user_input)
            elif self.step == 1:
                return self.ask_origin(user_input)
            elif self.step == 2:
                return self.ask_destination(user_input)
            elif self.step == 3:
                return self.ask_date(user_input)
            elif self.step == 4:
                return self.confirm_booking(user_input)
            elif self.step == 5:
                return self.select_flight(user_input)
            else:
                return "Booking completed."
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def ask_name(self, user_input):
        
        # if user_name already exsits, then stop asking
        if not self.user_name:
            self.user_name = user_input.strip()
            
        self.step = 1
        return "Hi, " + self.user_name + "! Where are you flying from?"

    def ask_origin(self, user_input):
        people, locations, dates_times = extract_entities(user_input.strip())
        self.origin = locations[0]
        self.step = 2
        return "Great! Now, where are you flying to?"

    def ask_destination(self, user_input):
        people, locations, dates_times = extract_entities(user_input.strip())
        self.destination = locations[0]
        self.step = 3
        return f"Got it! When would you like to fly from {self.origin} to {self.destination}? (Please enter a date in YYYY-MM-DD format)\nYou can type 'skip' if you're unsure."

    def ask_date(self, user_input):
        
        if user_input.lower() == 'skip':
            self.date = None
            self.step = 4
            return f"You're flying from {self.origin} to {self.destination}, No date selected. Is everything correct?(yes/no)"
        
        parsed_date = self.parse_date(user_input)
        if parsed_date:
            self.date = parsed_date
            self.step = 4
            return f"You're flying from {self.origin} to {self.destination} on {self.date}.  Is everything correct? (yes/no)"
        else:
            return "Invalid date format. Please enter a date in YYYY-MM-DD format."

    def confirm_booking(self, user_input):
        if user_input.lower() == 'yes':
            self.step = 5
            flights = search_flights(self.origin, self.destination, self.date)
            # Ensure flights is a valid list and not empty
            if isinstance(flights, list) and flights:  
                flight_options = "\n".join(
                    [f"Flight {i + 1}: {flight[1]} to {flight[2]} on {flight[3]} - ${flight[4]}" for i, flight in enumerate(flights)]
                )
                return f"Booking confirmed for {self.user_name} from {self.origin} to {self.destination} on {self.date}.\nAvailable flights:\n{flight_options} \n Please enter the flight number to book."
            else:
                self.step = 0
                return "I'm sorry, but it seems there aren't any available flights for your requested search. Let's start fresh - (you can type 'reset' to reset the search or 'exit' to exit the system.)"
        else:
            self.step = 0
            return "Let's start over. (you can type 'reset' to reset the search or 'exit' to exit the system.)"
        
    def select_flight(self, user_input):
        try:
            flight_choice = int(user_input)
            flights = search_flights(self.origin, self.destination, self.date)
            if 1 <= flight_choice <= len(flights):
                selected_flight = flights[flight_choice - 1]
                #selected_flight[0] flight id
                booking_message = book_flight(selected_flight[0], self.user_name)
                self.flight_selected = True
                return booking_message + "\n" + self.view_bookings()
            else:
                return "Invalid flight number. Please try again."
        except ValueError:
            return "Please enter a valid number."   
    
    def view_bookings(self):
        bookings = view_bookings(self.user_name)
        if isinstance(bookings, str):
            return bookings
        else:
            booking_details = "\n".join([f"Booking ID {b[0]}: {b[1]} to {b[2]} on {b[3]} - ${b[4]}" for b in bookings])
            return f"Your bookings:\n{booking_details}"
        
    def parse_date(self, user_input):
        parsed_date = dateparser.parse(user_input)
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        else:
            return None
    def extract_locations_and_date(self,user_input):
        
        """Extract the departure and destination locations and date from the user's input."""
        try:
             # Step 1: Extract locations and date using extract_entities
            people, locations, dates_times = extract_entities(user_input)

            # Extract origin and destination (assume at least two locations)
            origin = locations[0] if len(locations) > 0 else None
            destination = locations[1] if len(locations) > 1 else None

            # Extract date (use the first recognized date or None if unavailable)
            date = None
            parsed_date = None
            if dates_times:
                parsed_date = dateparser.parse(dates_times[0])
                date = parsed_date.strftime('%Y-%m-%d') if parsed_date else None

            return origin, destination, date
        except Exception as e:
            print(f"Error while extracting locations and date: {str(e)}")
            return None, None, None


class FlightSearchSystem:
    def __init__(self):
        self.step = 0
        self.origin = None
        self.destination = None
        self.date = None

    def reset_search(self):
        self.step = 0
        self.origin = None
        self.destination = None
        self.date = None
        return "Search has been reset. Let's start over!"

    def handle_input(self, user_input):
        if user_input.lower() == "reset":
            return self.reset_search()

        if self.step == 0:
            return self.ask_origin(user_input)
        elif self.step == 1:
            return self.ask_destination(user_input)
        elif self.step == 2:
            return self.ask_date(user_input)
        elif self.step == 3:
            return self.confirm_search(user_input)
        else:
            return "Search completed."

    def ask_origin(self, user_input):
        people, locations, dates_times = extract_entities(user_input.strip())
        self.origin = locations[0]
        self.step = 1
        return "Where are you flying to?"

    def ask_destination(self, user_input):
        people, locations, dates_times = extract_entities(user_input.strip())
        self.destination = locations[0]
        self.step = 2
        return f"Great! When do you want to fly from {self.origin} to {self.destination}? (Please enter a date in YYYY-MM-DD format)\nYou can type 'skip' if you're unsure."

    def ask_date(self, user_input):
        
        if user_input.lower() == 'skip':
            self.date = None
            self.step = 3
            return f"You're searching for flights from {self.origin} to {self.destination}, No date selected. Is everything correct?(yes/no)"
        
        parsed_date = self.parse_date(user_input)
        if parsed_date:
            self.date = parsed_date
            self.step = 3
            return f"You're searching for flights from {self.origin} to {self.destination} on {self.date}. Is everything correct?(yes/no)"
        else:
            return "Invalid date format. Please enter a date in YYYY-MM-DD format."

    def confirm_search(self, user_input):
        if user_input.lower() == 'yes':
           return self.show_flights(user_input)
        else:
            self.step = 0
            return "Let's start over. where are you flying from?"
    def show_flights(self, user_input):
        flights = search_flights(self.origin, self.destination, self.date)
        if flights and isinstance(flights, list):
            best_flight = flights[0]  # Give the cheapset flight
            flight_details = "\n".join([f"Flight {i+1}: {flight[1]} to {flight[2]} on {flight[3]} - ${flight[4]}" 
                                        for i, flight in enumerate(flights)])
            response = (f"Available flights:\n{flight_details}\n"
                        f"The cheapest flight: {best_flight[1]} to {best_flight[2]} on {best_flight[3]} - ${best_flight[4]}")
        else:
            response = "Sorry! No flights found."
        self.step = 4
        return response
    
    # Add a date parsing feature that allows users to input vague dates, such as "tomorrow" or "next Friday.。
    def parse_date(self,user_input):
        parsed_date = dateparser.parse(user_input)
        if parsed_date:
            return parsed_date.strftime('%Y-%m-%d')
        else:
            return None
        
    def extract_locations_and_date(self,user_input):
        
        """Extract the departure and destination locations and date from the user's input."""
        try:
             # Step 1: Extract locations and date using extract_entities
            people, locations, dates_times = extract_entities(user_input)

            # Extract origin and destination (assume at least two locations)
            origin = locations[0] if len(locations) > 0 else None
            destination = locations[1] if len(locations) > 1 else None

            # Extract date (use the first recognized date or None if unavailable)
            date = None
            if dates_times:
                parsed_date = dateparser.parse(dates_times[0])
                date = parsed_date.strftime('%Y-%m-%d') if parsed_date else None

            return origin, destination, date
        except Exception as e:
            print(f"Error while extracting locations and date: {str(e)}")
            return None, None, None

# Load QA dataset
def load_qa_dataset(file_path):
    qa_data = pd.read_csv(file_path)
    questions = qa_data['Question'].tolist()
    answers = qa_data['Answer'].tolist()
    return questions, answers

# Load Smalltalk dataset
def load_st_dataset(file_path):
    qa_data = pd.read_csv(file_path)
    questions = qa_data['Question'].tolist()
    answers = qa_data['Answer'].tolist()
    return questions, answers

# train QA model using TfidfVectorizer
def train_qa_model():
    questions, answers = load_qa_dataset('./COMP3074-CW1-Dataset.csv')

    # Preprocess the training data
    texts = [preprocess_text(question) for question in questions]
    qa_vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words=None)
    qa_X_tfidf = qa_vectorizer.fit_transform(texts)

    # Return the vectorizer, TF-IDF matrix, and list of answers
    return qa_vectorizer, qa_X_tfidf, answers


# Load Smalltalk dataset
def train_st_model():
    questions, answers = load_st_dataset('./smalltalk_responses.csv')

    # Preprocess the training data
    texts = [preprocess_text(question) for question in questions]
    st_vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words=None)
    st_X_tfidf = st_vectorizer.fit_transform(texts)

    # Return the vectorizer, TF-IDF matrix, and list of answers
    return st_vectorizer, st_X_tfidf, answers


def question_answering(user_input, qa_vectorizer, qa_X_tfidf, answers):

    processed_input = preprocess_text(user_input)
    user_tfidf = qa_vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_tfidf, qa_X_tfidf)
    max_index = np.argmax(similarities)
    # Add debug information
    # print(f"Processed input: {processed_input}")
    # print(f"Similarities shape: {similarities.shape}")
    # print(f"Similarities: {similarities}")
    # print(f"Max similarity index: {max_index}")
    # print(f"Length of answers: {len(answers)}")
    # print("similarities[0][max_index]",similarities[0][max_index])
    # Return the answer to the most similar question
    
    if similarities[0][max_index] > 0.2:  # set threshold
        return answers[max_index]
    else:
        return "I'm not sure about that. Can you ask something else?"

# Load QA dataset
def load_intent_dataset(file_path):
    intent_data = pd.read_csv(file_path)
    sentences = intent_data['Sentence'].tolist()
    intents = intent_data['Intent'].tolist()
    return sentences, intents

# Build an intent recognition model
def train_intent_model():
    
    """
    Train and evaluate the intent classification model
    Returns:
        model: trained LogisticRegression model
        vectorizer: fitted TfidfVectorizer
        evaluation_results: dict containing evaluation metrics
    """
    
    # Load and preprocess data
    sentences, intents = load_intent_dataset('./intent_training_data.csv')
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=None,ngram_range=(1, 2))
    X = vectorizer.fit_transform(preprocessed_sentences)
    y = intents

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
    # Define parameter grid
    # param_grid = {
    #     'C': [0.1, 1.0, 10.0],
    #     'solver': ['liblinear', 'saga','lbfgs '],
    #     'max_iter': [1000, 2000]
    # }
    
    # Instantiate Logistic Regression
    # lr = LogisticRegression(class_weight='balanced', multi_class='multinomial', random_state=42)

    # # Using GridSearchCV
    # grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    # # Output the best parameters
    # print("Best parameters:", grid_search.best_params_)

  
    
    # Train the model using the best parameters
    # model = grid_search.best_estimator_

    # Initialize model with best parameters from grid search
    model = LogisticRegression(
        class_weight='balanced', 
        C=10.0, 
        solver='saga', 
        random_state=42, 
        max_iter=1000
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate model using accuracy, classification report, and confusion matrix
    evaluate_intent_model( model, vectorizer, X_test, y_test)
    
    
    return model, vectorizer

# Evaluate the intent classification model
def evaluate_intent_model( model, vectorizer, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        evaluation_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'intent_distribution': pd.Series(y_test).value_counts().to_dict(),
            'class_probabilities': {
                'mean': np.mean(np.max(y_pred_proba, axis=1)),
                'std': np.std(np.max(y_pred_proba, axis=1))
            }
        }
        
        print("\nIntent Classification Model Evaluation")
        print("=====================================")
        print(f"\nAccuracy: {evaluation_results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(evaluation_results['classification_report'])
        print("\nConfusion Matrix:")
        print(evaluation_results['confusion_matrix'])
        print("\nIntent Distribution:")
        for intent, count in evaluation_results['intent_distribution'].items():
            print(f"{intent}: {count} samples")
        print("\nPrediction Confidence:")
        print(f"Mean: {evaluation_results['class_probabilities']['mean']:.4f}")
        print(f"Std: {evaluation_results['class_probabilities']['std']:.4f}")
        
        example_sentences = [
            "I want to book a flight",
            "show me my bookings",
            "what is the weather like",
            "hello there"
        ]
        print("\nExample Predictions:")
        for sentence in example_sentences:
            processed = preprocess_text(sentence)
            vec = vectorizer.transform([processed])
            pred = model.predict(vec)[0]
            prob = np.max(model.predict_proba(vec))
            print(f"\nInput: {sentence}")
            print(f"Predicted Intent: {pred}")
            print(f"Confidence: {prob:.4f}")

# create database and table of flights
def init_db():
    conn = sqlite3.connect("flights.db")
    cursor = conn.cursor()
    # flights table 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY,
            origin TEXT,
            destination TEXT,
            date TEXT,
            price REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_id INTEGER,
            user_name TEXT,
            booking_date TEXT,
            FOREIGN KEY (flight_id) REFERENCES flights (id)
        )
    ''')
    conn.commit()
    conn.close()

# insert some sample data
def insert_sample_data():
    conn = sqlite3.connect('flights.db')
    cursor = conn.cursor()

    # check whether there are data
    cursor.execute("SELECT COUNT(*) FROM flights")
    if cursor.fetchone()[0] > 0:
        print("Sample data already exists. Skipping insertion.")
        conn.close()
        return

    sample_flights = flights
    cursor.executemany("INSERT INTO flights (origin, destination, date, price) VALUES (?, ?, ?, ?)", sample_flights)
    conn.commit()
    conn.close()


# accept searching without date
def search_flights(origin, destination, date = None):
    conn = sqlite3.connect('flights.db')
    cursor = conn.cursor()
    
    if date:
        cursor.execute("SELECT * FROM flights WHERE LOWER(origin) LIKE ? AND LOWER(destination) LIKE? AND date=?", 
                       (f"%{origin}%", f"%{destination}%", date))
    else:
        cursor.execute("SELECT * FROM flights WHERE LOWER(origin) LIKE ? AND LOWER(destination) LIKE?", 
                       (f"%{origin}%", f"%{destination}%"))
    
    flights = cursor.fetchall()
    conn.close()
    
    if flights:
        flights = sorted(flights, key=lambda x: x[4])  # Sort by price
        return flights
    else:
        return "No flights found for your query. Try changing the date or destination."


def book_flight(flight_id, user_name):
    conn = sqlite3.connect('flights.db')
    cursor = conn.cursor()
    booking_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO bookings (flight_id, user_name, booking_date) VALUES (?, ?, ?)", 
                   (flight_id, user_name, booking_date))
    conn.commit()
    conn.close()
    return f"Your flight (Flight Number:{flight_id}) , has been successfully booked!"


def view_bookings(user_name):
    conn = sqlite3.connect('flights.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT b.booking_id, f.origin, f.destination, f.date, f.price 
                      FROM bookings b JOIN flights f ON b.flight_id = f.id WHERE b.user_name = ?''', (user_name,))
    bookings = cursor.fetchall()
    conn.close()
    return bookings if bookings else "Your booking record was not found."

def cancel_booking(booking_id, user_name):
    conn = sqlite3.connect('flights.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM bookings WHERE booking_id=? AND user_name=?", (booking_id, user_name))
    conn.commit()
    conn.close()
    return "booking has been cancelled。" if cursor.rowcount > 0 else "Booking not found, unable to cancel."



def detect_intent(user_input: str, model, vectorizer):
   
    # Load intent keywords from JSON file
    with open('intent_keywords.json', 'r') as file:
        intent_keywords = json.load(file)

    # using the model to predict the intent
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    probabilities = model.predict_proba(input_vector)
    max_probability = max(probabilities[0])
    predicted_intent = model.predict(input_vector)[0]
    # print(f"Maximum probability: {max_probability:.2f} The predicted intent is: {predicted_intent}")

    # If the model confidence is high, return directly
    if max_probability >= 0.6:
        return predicted_intent, max_probability

    # Otherwise, use keyword matching as a fallback mechanism
    input_lower = user_input.lower()
    tokens = word_tokenize(input_lower)
    
    # Check keywords for each intent
    for intent, config in intent_keywords.items():
        # Check for special smalltalk cases
        if intent == 'smalltalk':
            for phrase in config['phrases']:
                if edit_distance(input_lower, phrase) <= config['threshold']:
                    return intent, 1.0
        
        # Check for question answering
        elif intent == 'question_answering' and config.get('position') == 'start':
            if tokens and tokens[0] in config['phrases']:
                return intent, 1.0
        
        # Check for other intents
        else:
            # Check main keywords
            has_main_keyword = any(keyword in tokens for keyword in config['phrases'])
            # Check required keywords
            has_required = any(req in tokens for req in config.get('requires', []))
            
            if has_main_keyword and has_required:
                return intent, 0.8

    # If no intent is matched, return the original prediction but with lower confidence
    return predicted_intent, max_probability


def handle_intent(user_input, model, vectorizer):
    # Detect intent and check confidence
    intent, confidence = detect_intent(user_input, model, vectorizer)
    
    # Check if the intent requires confirmation
    if confidence >= 0.6:
        return intent, None
    else:
        confirmation_message = (
            f"I think you want to {intent.replace('_', ' ')}. "
            f"Is that correct? (yes/no)"
        )
        return intent, confirmation_message


# preprocess text
def preprocess_text(text):
    
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return " ".join(tokens)


def get_current_weather(city_name):
    lat, lon = get_coordinates(city_name)
    if lat is None or lon is None:
        return "I'm sorry, but I couldn't find the coordinates for the city you mentioned. Could you check the name and try again?"

    api_key = 'xxxxxxxxxxxxxxxxxxxxxxx'
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=en'

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        feels_like = data['main']['feels_like']
        pressure = data['main']['pressure']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        wind_deg = data['wind']['deg']

        return (f"Hey there! Here's the current weather for {city_name}:\n"
                f"Weather: {weather_description.capitalize()}\n"
                f"Temperature: {temperature}°C (Feels like {feels_like}°C)\n"
                f"Wind: {wind_speed} m/s, blowing at {wind_deg}°\n"
                f"Humidity: {humidity}%\n")
    else:
        return "Sorry, I couldn't retrieve the weather information at the moment. Please try again later."


def get_coordinates(city_name):
    api_key = 'xxxxxxxxxxxxxxxxxxxxxx'
    url = f'http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return lat, lon
        else:
            return None, None
    else:
        return None, None

def handle_weather_intent(user_input):
    
    """
    Handles user input related to weather queries.
    
    Args:
        user_input (str): The user's input message.
    
    Returns:
        str: Weather information for the detected city, or a detailed prompt for proper input.
    """
    
    # Attempt to extract city using regex
    city_pattern = re.compile(r'weather (?:in|at|for)?\s+([a-zA-Z\s]+)')
    match = city_pattern.search(user_input.lower())
    
    if match:
        city = match.group(1).strip()
        return get_current_weather(city)
    
    # if regex fails, use NER to extract city
    people, locations, dates_times = extract_entities(user_input)
    if locations:
        city = locations[0] 
        return get_current_weather(city)
    
    # If no city is detected, provide a helpful error prompt
    return ("Sorry! I couldn't detect a city name in your input. "
            "Please provide the city name in a format like: "
            "'What is the weather in London?' or 'Tell me the weather at Tokyo'.")

def handle_greet():
    global user_name
    if user_name:
        # If there is already a username, then greet directly.
        return f"Hello again, {user_name}! How can I assist you today?"
    else:
        # Otherwise, ask the user to provide a name.
        while True:
            user_input = input("AI: Hello! What's your name?\nYou: ")
            
            # use regular expression to match the name
            name_patterns = [
                r"my name is ([a-zA-Z\s]+)",  # Match "my name is John Doe"
                r"([a-zA-Z\s]+) is my name",  # Match "John Doe is my name"
            ]

            # Attempt to match different patterns
            for pattern in name_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    user_name = match.group(1).strip()  # Extract the name and remove extra spaces
                    return f"Nice to meet you, {user_name}! How can I assist you today?"
            
            # Extract entities using NER
            people, locations, dates_times = extract_entities(user_input)
            if people:
                user_name = people[0]
                return f"Nice to meet you, {user_name}! How can I assist you today?"
            # If no pattern matched, ask for the name again with a clearer prompt
            print("I didn't catch your name clearly. Could you please provide your name in a format like 'My name is John Doe'?")

def handle_get_name():
    if user_name:
        return f"Your name is {user_name}."
    else:
        return "I don't know your name yet. Please tell me your name."
    
def extract_entities(user_input):
    # Tokenize the input and extract POS tags
    tokens = word_tokenize(user_input)
    pos_tags = pos_tag(tokens)
    
    # Perform named entity recognition (NER) using NLTK's ne_chunk
    named_entities = ne_chunk(pos_tags)
    
    # Initialize lists to store entities
    people = []
    locations = []
    dates_times = []
    
    for tree in named_entities:
        if isinstance(tree, Tree):  # Check if it's a named entity
            label = tree.label()
            entity = " ".join([child[0] for child in tree])  # Extract entity text
            
            if label == 'PERSON':
                people.append(entity)
            elif label in ['GPE', 'LOCATION']:  # GPE: Geo-Political Entity
                locations.append(entity)
            # elif label in ['DATE', 'TIME']:
            #     dates_times.append(entity)
    
    
     # If NER doesn't find any locations, use regex to extract from/to locations
    if len(locations) < 2:  # Ensure we don't overwrite existing valid locations
        from_to_pattern = r'from\s+([\w\s]+?)\s+to\s+([\w\s]+)'
        match = re.search(from_to_pattern, user_input, re.IGNORECASE)
        if match:
            if len(locations) == 0:  # No locations found by NER
                locations.append(match.group(1).strip())
            if len(locations) == 1:  # Only one location found by NER
                locations.append(match.group(2).strip())
    
    parsed_date = None  # Initialize parsed_date to None
    date_pattern = (
        r'\b\d{4}-\d{2}-\d{2}\b|'        # 2023-12-20
        r'\b\d{4}/\d{2}/\d{2}\b|'        # 2023/12/20
        r'\b\d{2}-\d{2}-\d{4}\b|'        # 20-12-2023
        r'\b\d{2}/\d{2}/\d{4}\b|'        # 20/12/2023
        r'\b\d{1,2}/\d{1,2}/\d{4}\b'     # 12/20/2023
    )
    matches = re.findall(date_pattern, user_input)
    if matches:
        start_date = matches[0] 
        parsed_date = dateparser.parse(start_date)
    
    if parsed_date:
        formatted_date = parsed_date.strftime('%Y-%m-%d')
        if formatted_date not in dates_times:
            dates_times.append(formatted_date)

    
    # Return the lists
    return people, locations, dates_times
    
def handle_set_name(user_input):
    global user_name
    
    
    # use regular expression to match the name
    name_patterns = [
        r"my name is ([a-zA-Z\s]+)",  # match "my name is Chen Zhendi"
        r"([a-zA-Z\s]+) is my name",  # match "Chen Zhendi is my name"
    ]

    # Attempt to match different patterns
    for pattern in name_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match: 
            # extract the name and remove extra spaces
            user_name = match.group(1).strip()  
            return f"Nice to meet you, {user_name}!"
    
    people = []
   
    people, locations, dates_times = extract_entities(user_input)
    if people:
        user_name = people[0]   
        
        return f"Nice to meet you, {user_name}!"
    
    
    return "Sorry! I didn't catch your name clearly. Could you please provide your name in a format like 'My name is Xxx Xxx'?"

def handle_discoverability():
    return """I can help you with the following:
    
            1. Flight Booking Services:
                - Book a new flight (Example: "I want to book a flight")
                - Search available flights (Example: "Search flights from London to Paris")
                - View your bookings (Example: "Show my bookings")
                - Cancel bookings (Example: "Cancel booking 123")
            2. Weather Information:
                - Check weather for your departure/destination city
                - Just ask: "How's the weather in [city name]?"
            3. General Assistance:
               - Help with general inquiries
               - Smalltalk
               - Type 'exit' to end our conversation
            
            What would you like help with?"""

def handle_smalltalk(user_input, st_vectorizer, st_X_tfidf, answers):
    processed_input = preprocess_text(user_input)
    user_tfidf = st_vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_tfidf, st_X_tfidf)
    max_index = np.argmax(similarities)
    
    # Return the answer to the most similar question.
    if similarities[0][max_index] > 0.2:  # set threshold
        return answers[max_index]
    else:
        return "I'm not sure about that. Can you ask something else?"


def handle_search_flight(user_input):
    try:
        system = FlightSearchSystem()
        print("Welcome to the flight search system!")

        # Try to directly extract locations from the input.
        origin, destination, date = system.extract_locations_and_date(user_input)

        if origin and destination:
            system.origin = origin
            system.destination = destination
            if date:
                system.date = date
                system.step = 3 # Jump to confirmation
                print(f"System: You're searching for flights from {system.origin} to {system.destination} on {system.date}. Is everything correct? (yes/no)")
            else:
                system.step = 2 # Jump to date input
                print(f"System: Great! You want to search for flights from {system.origin} to {system.destination}.(Please enter a date in YYYY-MM-DD format)\n You can type 'skip' if you're unsure.")
        else:
            print("System: Please enter the city you are flying from.")
            
        while system.step != 4:
            # if(system.step == 0):
            #     print("System: Please enter the city you are flying from.")
            user_input = input("You: ")
            if user_input.lower() == 'exit':
               return "search cancelled. Goodbye!"
            response = system.handle_input(user_input)
            print("System: " + response)

        return "Search completed. Thank you for using the flight search system!"
    except Exception as e:
        return f"Sorry, an error occurred during the search process: {str(e)}"

def handle_book_flight(user_input):
    try:
        global user_name
        system = FlightBookingSystem(user_name)
        print("Welcome to the flight booking system!")
        # first extract locations and date
        origin, destination, date = system.extract_locations_and_date(user_input)
        # print(f"Extracted locations: Origin - {origin}, Destination - {destination}, Date - {date}")
                
        if not user_name:  # If the user's name is not set
            system.step = 1  # Set the step to 1 to ask for the user's name
            while not system.user_name:  # Keep asking until a valid name is provided
                user_input = input("System: Please enter your name: ")
                if user_input.strip():  # Check if the input is not empty
                    people, locations, dates_times = extract_entities(user_input.strip())
                    if people and people[0] is not None:
                        system.user_name = people[0]
                        user_name = people[0]
                        print(f"System: Thank you, {system.user_name}! Let's continue.")
                    else:
                        print("System: Sorry, I didn't catch your name clearly. Could you please provide your name in a format like 'My name is Xxx Xxx'?")
                        
                else:
                    print("System: Your name cannot be empty. Please enter a valid name.")    
        else:
            system.step = 1  # Set the step to 1 to ask for the user's name
            print(f"System: Hi {user_name}! Let's continue.")
        
      
        # After handling the name, process the flight information
        if origin and destination:
            system.origin = origin
            system.destination = destination
            
            if date:
                system.date = date
                system.step = 4  # Jump to confirmation
                print(f"System: You're flying from {system.origin} to {system.destination} on {system.date}. Is everything correct? (yes/no)")
            else:
                system.step = 3  # Jump to date input
                print(f"System: Got it! When would you like to fly from {system.origin} to {system.destination}? (Please enter a date in YYYY-MM-DD format)\n You can type 'skip' if you're unsure.")
        else:
            print("System: Please enter the city you are flying from.")
        
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                return "Booking cancelled. Goodbye!"
            
            response = system.handle_input(user_input)
            print("System: " + response)

            if system.flight_selected:
                return "Booking completed. Thank you for using the flight booking system!"
    except Exception as e:
        return f"Sorry, an error occurred during the booking process: {str(e)}"
        

def handle_view_bookings(user_input):
    global user_name
    
    if not user_name:
        return "I'm sorry, but I don't know your name yet. Please tell me your name first."
    
    bookings = view_bookings(user_name)
    if isinstance(bookings, str):
        response = bookings
    else:
        booking_details = "\n".join([f"Booking ID {b[0]}: {b[1]} to {b[2]} on {b[3]} - ${b[4]}" for b in bookings])
        response = f"Your bookings:\n{booking_details}"
    return response

def handle_cancel_booking(user_input):
    # View the user's current bookings
    bookings = view_bookings(user_name)
    if isinstance(bookings, str):
        return bookings  # If there are no bookings, return the information directly
    
    # Display current booking details
    booking_details = "\n".join([f"Booking ID {b[0]}: {b[1]} to {b[2]} on {b[3]} - ${b[4]}" for b in bookings])
    response = f"Here are your current bookings:\n{booking_details}\n you can cancel a booking by typing 'cancel booking [booking_id]'"
    
    
    
    # Extract booking_id from user input, e.g., user inputs "cancel booking 123" to extract 123
    booking_id_match = re.search(r'\d+', user_input)
    if booking_id_match:
        booking_id = int(booking_id_match.group())
        cancel_message = cancel_booking(booking_id, user_name)
        
        # Display the current bookings after cancellation
        bookings_after_cancel = view_bookings(user_name)
        if isinstance(bookings_after_cancel, str):
            return f"{cancel_message}\n{bookings_after_cancel}"
        else:
            booking_details_after_cancel = "\n".join([f"Booking ID {b[0]}: {b[1]} to {b[2]} on {b[3]} - ${b[4]}" for b in bookings_after_cancel])
            return f"{cancel_message}\nHere are your current bookings:\n{booking_details_after_cancel}"
    else:
        return response



# Main dialogue loop
def main():
    try:
        model, vectorizer = train_intent_model()
        qa_vectorizer, qa_X_tfidf, qa_answers = train_qa_model()
        st_vectorizer, st_X_tfidf, st_answers = train_st_model()
        init_db()
        insert_sample_data()
        print("Hello! Welcome to the flight booking assistant. Type 'exit' to quit.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            intent, confirmation = handle_intent(user_input, model, vectorizer)
            # to show the intent selected
            # print("intent is：" + intent)
            if intent == "greet":
                response = handle_greet()
            elif intent == "get_name":
                response = handle_get_name()
            elif intent == "set_name":
                response = handle_set_name(user_input)
            elif intent == "smalltalk":
                response = handle_smalltalk(user_input, st_vectorizer, st_X_tfidf, st_answers)
            elif intent == "discoverability":
                response = handle_discoverability()
            elif intent == "question_answering": 
                response = question_answering(user_input, qa_vectorizer, qa_X_tfidf, qa_answers)
            elif intent == "search_flight":
                response = handle_search_flight(user_input)
            elif intent == "book_flight":
                response = handle_book_flight(user_input)
            elif intent == "cancel_booking":
                response = handle_cancel_booking(user_input)
            elif intent == "view_bookings":
                response = handle_view_bookings(user_input)
            elif intent == "weather":
                response = handle_weather_intent(user_input)
            else:
                response = "I'm not sure what you mean. Here are some things I can help you with: booking flights, checking FAQs, or general small talk."

            print(f"AI: {response}")
    except Exception as e:
        print(f"An error occurred in the main loop: {str(e)}")

if __name__ == "__main__":
    main()


