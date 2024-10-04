from flask import Flask, request, render_template
import pandas as pd
from joblib import load
import random
import json
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.models import load_model

nltk.download('popular')


app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('dataset.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model_map = {
    'ذكر': {
        'متزوج': {
            'model': 'Male_Married_model.joblib',
            'data': 'cleaned_dataset88.csv'
        },
        'أعزب': {
            'model': 'Male_Single_model.joblib',
            'data': '/home/user/AI-Python-Engine/Saleem_Ai/cleaned_dataset77.csv'
        }
    },
    'انثى': {
        'متزوج': {
            'model': '/home/user/AI-Python-Engine/Saleem_Ai/FeMale_Married_model.joblib',
            'data': '/home/user/AI-Python-Engine/Saleem_Ai/cleaned_dataset102.csv'
        },
        'أعزب': {
            'model': '/home/user/AI-Python-Engine/Saleem_Ai/FeMale_Single_model.joblib',
            'data': '/home/user/AI-Python-Engine/Saleem_Ai/cleaned_dataset102.csv'
        }
    }
}

bot_greetings = [
    "اهلا و سهلا، هل تحتاج المساعدة؟",
    "اهلا، هل تحتاج المساعدة؟",
    "سعدت برؤيتك، هل تحتاج المساعدة؟"
]

user_greetings = [
    "مرحبا",
    "كيف حالك",
    "هاي",
    "السلام عليكم",
    "السلام عليكم ورحمة الله وبركاته",
    "شو الأخبار",
    "هلا",
    "كيفك",
    "اخبارك",
    "هللو",
    "هيلو",
    "صباح الخير",
    "صباحك",
    "مساء الخير",
    "مسائو",
    "سليم"
]

age_questions = ['كم عمرك', 'قديه عمرك', 'قديش عمرك', 'شو عمرك',
                 'احسب عمرك', 'عمرك كم', 'شو هو عمرك', 'عمرك', 'ئديه عمرك', 'عمرك؟']
name_questions = ['شو اسمك ', 'ما اسمك', 'اسمك',
                  'اسمك شو', 'اسمك ايه', 'احكيلي اسمك']


def getResponse(intents):
    user_input = input(intents).strip().lower()
    return user_input


def chatbot_response(msg):
    global z
    if any(keyword in msg.lower() for keyword in ["مين انت", "انت مين", "من انت"]):
        return " أنا سليم، مساعد ذكاء صناعي. كيف يمكنني مساعدتك اليوم"

    for greeting in user_greetings:
        if greeting in msg.lower():
            return random.choice(bot_greetings)

    for age_question in age_questions:
        if age_question in msg.lower():
            return " ليس لدي عمر فعلي، فأنا مجرد أداة ذكاء اصطناعي."

    for name_question in name_questions:
        if name_question in msg.lower():
            return " ليس لدي اسم فعلي، فأنا مجرد برنامج ذكاء صناعي."
    if msg.lower() == 'خروج':
        return 'سليم: شكرا لتواصلك معي '
    selected_body_part_name = msg.lower()

    if selected_body_part_name in body_parts and z == 0:
        global symptoms_for_selected_body_part, o, selected_body_part_name_final
        o = 0
        body_part_prompt_1 = "\n: يرجى ادخال العرض الأساسي"

        # Get unique body parts and format them as a numbered list
        body_part_df = data[data['جزء الجسم'] == selected_body_part_name]
        symptoms_for_selected_body_part = body_part_df['1 العرض'].unique()
        formatted_list = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(
            symptoms_for_selected_body_part, 1)])
        selected_body_part_name_final = selected_body_part_name
        z = 1
        return body_part_prompt_1 + "\n" + formatted_list
    if o == 0:
        global selected_symptom_name, body_part_prompt_2
        selected_symptom_name = msg.lower()
        selected_symptom_name = symptoms_for_selected_body_part[int(
            selected_symptom_name) - 1]
        body_part_prompt_2 = "\n: :: يرجى الاجابة على الاسئلة التالية ب نعم او لا "
        # selected_symptom_name = 'اعتصار الصدر '
 
        for i in range(1, len(data) + 1):
            try:
                symptom_col_name = f'{i} العرض'
                symptoms_for_selected_symptom = data[(data['جزء الجسم'] == selected_body_part_name_final) & (
                    data['1 العرض'] == selected_symptom_name) & (data[symptom_col_name].notnull())][symptom_col_name].unique()
                symptoms_for_selected_symptom_list.extend(
                    symptoms_for_selected_symptom)

            except:
                pass

        o = 1
        global len_symptoms_for_selected_symptom_list
        len_symptoms_for_selected_symptom_list = len(
            symptoms_for_selected_symptom_list)

    global user_input, first_item
    user_input = msg.lower()
    try:
        first_item = symptoms_for_selected_symptom_list[0]
        symptoms_for_selected_symptom_list.pop(0)
        return body_part_prompt_2 + "\n" + first_item
    except:
        pass


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/len_questions")
def len_questions():
    return len_symptoms_for_selected_symptom_list

@app.route("/info", methods=["POST", "GET"])
def info_route():
    global model, data, body_parts, z, symptoms_for_selected_symptom_list, user_dict
    user_dict = {}
    symptoms_for_selected_symptom_list = []
    data = {'Gender': 'ذكر', 'Marital_Status': 'متزوج'}
    # data = request.json
    Gender = data.get('Gender')
    Marital_Status = data.get('Marital_Status')

    def select_resources(gender, marital_status):
        try:
            resource = model_map[gender][marital_status]
            model = load(resource['model'])
            data = pd.read_csv(resource['data'])
            return model, data
        except KeyError:
            return None, None
    z = 0
    resource = model_map[Gender][Marital_Status]
    model = load(resource['model'])

    data = pd.read_csv(resource['data'])
    model, data = select_resources(Gender, Marital_Status)
    body_part_prompt = "\n: يرجى ادخال اسم الجزء الذي تشعر به الألم"

    # Get unique body parts and format them as a numbered list
    body_parts = data['جزء الجسم'].unique()
    formatted_list = '\n'.join(
        [f"{index}. \"{item}\"" for index, item in enumerate(body_parts, 1)])

    return body_part_prompt + "\n" + formatted_list


@app.route("/get", methods=["GET"])
def get_route():
    user_text = request.args.get('msg')
    try:
        user_input = request.args.get('msg')
        if user_input == "نعم":
            # Update the user_input dictionary with the symptom set to 1
            user_dict[first_item] = 1
        elif user_input == "لا":
            # Update the user_input dictionary with the symptom set to 0
            user_dict[first_item] = 0

        if len(user_dict) == len_symptoms_for_selected_symptom_list:
            all_columns = data.columns
            user_data = pd.DataFrame(
                0, columns=all_columns.drop('المرض'), index=[0])
            for symptom, value in user_dict.items():
                user_data[symptom] = value
            user_data = user_data[model.feature_names_in_]
            user_data.columns = model.feature_names_in_

            prediction_probs = model.predict_proba(user_data)

            prob_df = pd.DataFrame(prediction_probs, columns=model.classes_)

            top_3_probs = prob_df.iloc[0].nlargest(3)

            output = " بناءً على العملية التحليلية التي تمت، يوجد لديك اشتباه بالأمراض التالية: "
            for disease, probability in top_3_probs.items():
                # Round to two decimal places
                percentage = round(probability * 100)
                output += f"{disease}: {percentage}%"
            return output
    except:
        pass

    return chatbot_response(user_text)


if __name__ == "__main__":
    app.run()
