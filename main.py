import speech_recognition as sr
from gtts import gTTS
from joblib import load
import pyttsx3
import pandas as pd
import os

from Chatbotfinal import prompt_user_input, ask_about_pain_location, ask_about_primary_symptom, \
    ask_about_symptom_response

model_map = {
    'ذكر': {
        'متزوج': {
            'model': 'Male_Married_model.joblib',
            'data': 'cleaned_dataset88.csv'
        },
        'اعزب': {
            'model': 'Male_Single_model.joblib',
            'data': 'cleaned_dataset77.csv'
        }
    },
    'انثى': {
        'متزوج': {
            'model': 'FeMale_Married_model.joblib',
            'data': 'cleaned_dataset102.csv'
        },
        'اعزب': {
            'model': 'FeMale_Single_model.joblib',
            'data': 'cleaned_dataset102.csv'
        }
    }
}

def listen_to_user():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language="ar-SA").lower()
        except:
            return ""

def speak_to_user(text):
    tts = gTTS(text=text, lang='ar')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

def get_input_voice_or_text():
    choice = input("Do you prefer text or voice interaction? (text/voice): ")
    return choice

def select_resources(gender, marital_status):
    try:
        resource = model_map[gender][marital_status]
        model = load(resource['model'])
        data = pd.read_csv(resource['data'])
        return model, data
    except KeyError:
        return None, None

def predict_disease_voice(model, data):
    # First, ask the user to speak the name of 'جزء الجسم'
    speak_to_user("يرجى قول مكان الألم الذي تشعر به")
    body_parts = data['جزء الجسم'].unique()

    for index, body_part in enumerate(body_parts, 1):
        print(f"{index}. {body_part}")

    selected_body_part_name = listen_to_user()
    print(selected_body_part_name)

    if selected_body_part_name not in body_parts:
        speak_to_user("إدخال خاطئ!")
        return

    # Now, ask the user to speak the name of a symptom (العرض 1) related to the selected body part
    speak_to_user("يرجى قول العرض الأساسي")
    body_part_df = data[data['جزء الجسم'] == selected_body_part_name]
    symptoms_for_selected_body_part = body_part_df['1 العرض'].unique()

    for index, symptom in enumerate(symptoms_for_selected_body_part, 1):
        print(f"{index}. {symptom}")

    selected_symptom_name = listen_to_user()
    print(selected_symptom_name)

    if selected_symptom_name not in symptoms_for_selected_body_part:
        speak_to_user("إدخال خاطئ!")
        return

    # Now, ask the user to select 'العرض 2' related to 'العرض 1'
    symptoms_for_selected_symptom1 = body_part_df[body_part_df['1 العرض'] == selected_symptom_name]['2 العرض'].unique()

    for symptom in symptoms_for_selected_symptom1:
        print(symptom)
        speak_to_user(f":{symptom}?")
        response = listen_to_user().lower()
        print(response)

        if response == "نعم":
            selected_symptom2_name = symptom
            break
        elif response == "لا":
            continue  # Move to the next 'العرض 2' question
        else:
            speak_to_user("إدخال غير صالح. يرجى الرد بـ نعم أو لا.")
            return

    # Filter the dataset for the selected 'جزء الجسم', 'العرض 1', and 'العرض 2'
    filtered_df = body_part_df[(body_part_df['1 العرض'] == selected_symptom_name) & (body_part_df['2 العرض'] == selected_symptom2_name)]

    # Extract relevant columns (symptoms) for prediction based on 'العرض 1' and 'العرض 2'
    relevant_columns = [col for col in filtered_df.columns[3:] if filtered_df[col].values[0] == 1]

    # Create a dictionary to store the user input for symptoms
    user_input = {}

    # Ask the user for each symptom relevant to the selected symptoms
    for symptom in relevant_columns:
        print(symptom)
        speak_to_user(f": {symptom}؟")
        response = listen_to_user()
        print(response)

        if response == "نعم":
            user_input[symptom] = 1
        elif response == "لا":
            user_input[symptom] = 0
        elif response == "لا أعلم":
            continue
        else:
            speak_to_user("إدخال غير صالح. يرجى الرد بـ نعم أو لا أو لا أعلم.")

    # Create a DataFrame with all features except the target variable and set values based on user input
    all_columns = data.columns
    user_data = pd.DataFrame(0, columns=all_columns.drop('المرض'), index=[0])

    for symptom, value in user_input.items():
        user_data[symptom] = value

    # Get predicted probabilities for each disease
    prediction_probs = model.predict_proba(user_data)

    # Create a DataFrame for clearer display
    prob_df = pd.DataFrame(prediction_probs, columns=model.classes_)

    # Get the top 3 disease probabilities
    top_3_probs = prob_df.iloc[0].nlargest(3)

    # Display the top 3 disease probabilities using voice
    speak_to_user("بناءا على العملية التحليلية الذي تمت يوجد لديك اشتباه اعراض لهذه الامراض :")
    for disease, probability in top_3_probs.items():
        percentage = probability * 100
        speak_to_user(f"{disease}: {percentage:.1f}%")
    print("\nبناءا على العملية التحليلية الذي تمت يوجد لديك اشتباه اعراض لهذه الامراض :")
    for disease, probability in top_3_probs.items():
        percentage = probability * 100
        print(f"{disease}: %{percentage:.1f}")

def predict_disease_text(model, data):
    # Load the appropriate pre-trained model and dataset based on user's answers

    # First, ask the user to type the name of 'جزء الجسم'
    print("\n:  يرجى ادخال مكان الألم الذي تشعر به ")
    body_parts = data['جزء الجسم'].unique()
    for index, body_part in enumerate(body_parts, 1):
        print(f"{index}. {body_part}")

    selected_body_part_name = input("\n: مكان الألم ")

    if selected_body_part_name not in body_parts:
        print("إدخال خاطئ!")
        return

    # Now, ask the user to type the name of a symptom (العرض 1) related to the selected body part
    print("\n: يرجى ادخال العرض الأساسي")
    body_part_df = data[data['جزء الجسم'] == selected_body_part_name]
    symptoms_for_selected_body_part = body_part_df['1 العرض'].unique()

    for index, symptom in enumerate(symptoms_for_selected_body_part, 1):
        print(f"{index}. {symptom}")

    selected_symptom_name = input("\n: العرض الأساسي ")

    if selected_symptom_name not in symptoms_for_selected_body_part:
        print(" !ادخال خاطئ ")
        return

    # Now, ask the user to select 'العرض 2' related to 'العرض 1'
    print("\n: يرجى الاجابة على الاسئلة التالية")
    symptoms_for_selected_symptom1 = body_part_df[body_part_df['1 العرض'] == selected_symptom_name]['2 العرض'].unique()

    for symptom in symptoms_for_selected_symptom1:
        response = input(f": {symptom}? (نعم/لا) ").lower()

        if response == "نعم":
            selected_symptom2_name = symptom
            break
        elif response == "لا":
            continue  # Move to the next 'العرض 2' question
        else:
            print("إدخال غير صالح. يرجى الرد بـ نعم أو لا.")
            return

    # Filter the dataset for the selected 'جزء الجسم', 'العرض 1', and 'العرض 2'
    filtered_df = body_part_df[(body_part_df['1 العرض'] == selected_symptom_name) & (body_part_df['2 العرض'] == selected_symptom2_name)]


    # Now, extract relevant columns (symptoms) for prediction based on 'العرض 1' and 'العرض 2'
    relevant_columns = [col for col in filtered_df.columns[3:] if filtered_df[col].values[0] == 1]

    # Create a dictionary to store the user input for symptoms
    user_input = {}

    # Ask the user for each symptom relevant to the selected symptoms
    for symptom in relevant_columns:
        response = input(f": {symptom}؟ (نعم/لا/لا أعلم) ").lower()
        if response == "نعم":
            user_input[symptom] = 1
        elif response == "لا":
            user_input[symptom] = 0
        elif response == "لا أعلم":
            continue  # Simply skip to the next question
        else:
            print("إدخال غير صالح. يرجى الرد بـ نعم أو لا أو لا أعلم.")

    # Create a DataFrame with all features except the target variable and set values based on user input
    all_columns = data.columns
    user_data = pd.DataFrame(0, columns=all_columns.drop('المرض'), index=[0])

    for symptom, value in user_input.items():
        user_data[symptom] = value

    # Get predicted probabilities for each disease
    prediction_probs = model.predict_proba(user_data)

    # Create a DataFrame for clearer display
    prob_df = pd.DataFrame(prediction_probs, columns=model.classes_)

    # Get the top 3 disease probabilities
    top_3_probs = prob_df.iloc[0].nlargest(3)

    # Display the top 3 disease probabilities
    print("\nبناءا على العملية التحليلية الذي تمت يوجد لديك اشتباه اعراض لهذه الامراض ::")
    for disease, probability in top_3_probs.items():
        percentage = probability * 100
        print(f"{disease}: %{percentage:.1f}")

def predict_disease():
    gender = input("الرجاء ادخال الجنس (ذكر/انثى): ").strip().lower()
    marital_status = input("الرجاء ادخال الحالة الاجتماعية (أعزب/متزوج): ").strip().lower()

    model, data = select_resources(gender, marital_status)
    if model is None or data is None:
        print("Could not load resources.")
        return

    interaction_mode = get_input_voice_or_text()
    if interaction_mode == "voice":
        predict_disease_voice(model, data)
    else:
        predict_disease_text(model, data)

def predict_disease_text(model, data, selected_body_part_name, selected_symptom_name):
    print("\n: يرجى الاجابة على الاسئلة التالية")
    user_input = {}

    # Adjust the loop range based on the number of questions you want (here, 7 questions)
    for i in range(1, 8):
        symptom_col_name = f'{i} العرض'

        # Check if the column exists in the dataset
        if symptom_col_name not in data.columns:
            break  # No more symptoms for this level

        symptoms_for_selected_symptom = data[
            (data['جزء الجسم'] == selected_body_part_name) &
            (data['1 العرض'] == selected_symptom_name) &
            (data[symptom_col_name].notnull())
            ][symptom_col_name].unique()

        if not symptoms_for_selected_symptom:
            break  # No more symptoms for this level

        for symptom in symptoms_for_selected_symptom:
            response = ask_about_symptom_response(symptom)

            if response == "نعم":
                user_input[symptom] = 1
            elif response == "لا":
                user_input[symptom] = 0
            else:
                print("إدخال غير صالح. يرجى الرد بـ نعم أو لا.")
                return

    all_columns = data.columns
    user_data = pd.DataFrame(0, columns=all_columns.drop('المرض'), index=[0])

    for symptom, value in user_input.items():
        user_data[symptom] = value

    user_data = user_data[model.feature_names_in_]
    user_data.columns = model.feature_names_in_

    prediction_probs = model.predict_proba(user_data)

    prob_df = pd.DataFrame(prediction_probs, columns=model.classes_)

    top_3_probs = prob_df.iloc[0].nlargest(3)

    print("\nبناءً على العملية التحليلية التي تمت، يوجد لديك اشتباه بالأمراض التالية:")
    for disease, probability in top_3_probs.items():
        percentage = round(probability * 100)  # Round to two decimal places
        print(f"{disease}: {percentage}%")

def predict_disease():
    gender = prompt_user_input("الرجاء ادخال الجنس (ذكر/انثى): ")
    marital_status = prompt_user_input("الرجاء ادخال الحالة الاجتماعية (أعزب/متزوج): ")

    model, data = select_resources(gender, marital_status)
    if model is None or data is None:
        print("الرجاء ادخال معلومات دقيقة ")
        return

    selected_body_part_name = ask_about_pain_location(data)
    if selected_body_part_name is None:
        return

    selected_symptom_name = ask_about_primary_symptom(data, selected_body_part_name)
    if selected_symptom_name is None:
        return

    predict_disease_text(model, data, selected_body_part_name, selected_symptom_name)

if _name_ == '_main_':
    predict_disease()