import pandas as pd
from joblib import load

# Map of gender and marital status to dataset and model names
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

def select_resources():

    gender = input("الرجاء ادخال الجنس (ذكر/انثى): ").strip().lower()
    marital_state = input("الرجاء ادخال الحالة الاجتماعية (أعزب/متزوج): ").strip().lower()

    try:
        resources = model_map[gender][marital_state]
    except KeyError:
        print("إدخال خاطئ! يرجى اختيار القيم الصحيحة")
        return None, None

    # Load the appropriate pre-trained model and dataset
    model = load(resources['model'])
    data = pd.read_csv(resources['data'])

    return model, data


def predict_disease():
    # Load the appropriate pre-trained model and dataset based on user's answers
    random, data = select_resources()

    if random is None or data is None:
        return

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
    print("\n: يرجى ادخال العرض 2")
    symptoms_for_selected_symptom1 = body_part_df[body_part_df['1 العرض'] == selected_symptom_name]['2 العرض'].unique()

    for index, symptom in enumerate(symptoms_for_selected_symptom1, 1):
        print(f"{index}. {symptom}")

    selected_symptom2_name = input("\n: العرض 2 ")

    if selected_symptom2_name not in symptoms_for_selected_symptom1:
        print(" !ادخال خاطئ ")
        return

    # Filter the dataset for the selected 'جزء الجسم', 'العرض 1', and 'العرض 2'
    filtered_df = body_part_df[(body_part_df['1 العرض'] == selected_symptom_name) & (body_part_df['2 العرض'] == selected_symptom2_name)]


    # Now, extract relevant columns (symptoms) for prediction based on 'العرض 1' and 'العرض 2'
    relevant_columns = [col for col in filtered_df.columns[3:] if filtered_df[col].values[0] == 1]

    # Create a dictionary to store the user input for symptoms
    user_input = {}

    # Ask the user for each symptom relevant to the selected symptoms
    for symptom in relevant_columns:
        response = input(f": هل تعاني من {symptom}؟ (نعم/لا/لا أعلم) ").lower()
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
    prediction_probs = random.predict_proba(user_data)

    # Create a DataFrame for clearer display
    prob_df = pd.DataFrame(prediction_probs, columns=random.classes_)

    # Get the top 3 disease probabilities
    top_3_probs = prob_df.iloc[0].nlargest(3)

    # Display the top 3 disease probabilities
    print("\nنسبة احتمالية وجود مرض معين:")
    for disease, probability in top_3_probs.items():
        percentage = probability * 100
        print(f"{disease}: %{percentage:.1f}")


# Call the function to make predictions
predict_disease()

# Call the prediction function
