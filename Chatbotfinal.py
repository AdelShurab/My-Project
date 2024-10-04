import random
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
from gtts import gTTS
import pandas as pd
from joblib import load

warnings.filterwarnings('ignore')

# Download NLTK stopwords and punkt resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

age_questions = ['كم عمرك', 'قديه عمرك', 'قديش عمرك', 'شو عمرك', 'احسب عمرك', 'عمرك كم', 'شو هو عمرك', 'عمرك', 'ئديه عمرك', 'عمرك؟']
name_questions = ['شو اسمك ', 'ما اسمك', 'اسمك', 'اسمك شو','اسمك ايه', 'احكيلي اسمك']

def prompt_user_input(prompt):
    return input(prompt).strip().lower()

def ask_for_help():
    help_response = prompt_user_input("سليم: هل تحتاج مساعدة؟ (نعم/لا) ")
    if "نعم" in help_response:
        print("سليم: هل تريد معرفة معلومات عن الأمراض أم شيء آخر؟")
        return True
    else:
        return False
def select_resources(gender, marital_status):
    try:
        resource = model_map[gender][marital_status]
        model = load(resource['model'])
        data = pd.read_csv(resource['data'])
        return model, data
    except KeyError:
        return None, None
def greeting_response(user_input):
    for greeting in user_greetings:
        if greeting in user_input:
            return random.choice(bot_greetings)

    for age_question in age_questions:
        if age_question in user_input:
            return "سليم: ليس لدي عمر فعلي، فأنا مجرد أداة ذكاء اصطناعي."

    for name_question in name_questions:
        if name_question in user_input:
            return "سليم: ليس لدي اسم فعلي، فأنا مجرد برنامج ذكاء صناعي."

    if any(keyword in user_input for keyword in ["مين انت", "انت مين", "من انت"]):
        return "سليم: أنا سليم، مساعد ذكاء صناعي. كيف يمكنني مساعدتك اليوم?"

    return None

def main_interaction():
    while True:
        user_input = input("أنت: ")

        if user_input.lower() == 'خروج':
            print('سليم: شكرا لتواصلك معي ')
            break

        response = greeting_response(user_input)

        if response:
            print(f"سليم: {response}")
            continue  # Skip the rest of the loop if there is a valid response

        print("سليم: أنا هنا للمساعدة الطبية، كيف يمكنني مساعدتك اليوم?")

        if ask_for_help():
            print("سليم: اختر ما تريد:\n1. استفسار عن الأمراض")
            choice = prompt_user_input("أنت: ")

            if choice == "1":
                predict_disease()
            else:
                print("سليم: خيار غير صالح.")

        else:
            print("سليم: خيار غير صالح.")
def ask_about_pain_location(data):
    print("\n: يرجى ادخال اسم الجزء الذي تشعر به الألم")
    body_parts = data['جزء الجسم'].unique()

    for index, body_part in enumerate(body_parts, 1):
        print(f"{index}. {body_part}")

    selected_body_part_name = prompt_user_input("\n: اختر رقم الجزء أو ادخل اسم الجزء: ")

    # Check if the input is a number
    if selected_body_part_name.isdigit():
        selected_body_part_index = int(selected_body_part_name)
        if 1 <= selected_body_part_index <= len(body_parts):
            selected_body_part_name = body_parts[selected_body_part_index - 1]
        else:
            print("إدخال خاطئ!")
            return None
    elif selected_body_part_name not in body_parts:
        print("إدخال خاطئ!")
        return None

    return selected_body_part_name


def ask_about_primary_symptom(data, selected_body_part_name):
    print("\n: يرجى ادخال العرض الأساسي")
    body_part_df = data[data['جزء الجسم'] == selected_body_part_name]
    symptoms_for_selected_body_part = body_part_df['1 العرض'].unique()

    for index, symptom in enumerate(symptoms_for_selected_body_part, 1):
        print(f"{index}. {symptom}")

    selected_symptom_index = int(prompt_user_input("\n: اختر رقم العرض الأساسي: "))
    selected_symptom_name = symptoms_for_selected_body_part[selected_symptom_index - 1]

    if selected_symptom_name not in symptoms_for_selected_body_part:
        print(" !ادخال خاطئ ")
        return None

    return selected_symptom_name

def ask_about_symptom_response(symptom):
    response = prompt_user_input(f": {symptom}? (نعم/لا) ")
    return response.lower()


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

# Replace the following block in your existing code with the new main_interaction() function
print("☺️ مرحبا، كيف يمكن لسليم مساعدتك؟ إذا كنت لا تحتاج المساعدة اكتب 'خروج'")
main_interaction()





