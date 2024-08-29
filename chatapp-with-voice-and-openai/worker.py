from openai import OpenAI
import requests

openai_client = OpenAI()
def openai_process_message(user_message):
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences,summarize news, and give recommendations."

    openai_response = openai_client.chat.comoletions.create(
        model = "gpt-3.5-turbo",
        message=[
            {"role": "system", "Content" : prompt},
            {"role": "user", "content" : user_message}
        ],
        max_tokens=4000
    )
    print("openai response:", openai_response)
    response_text = openai_response.choices[0].message.content
    return response_text


def speech_to_text(audio_binary):
    base_url = "https://sn-watson-stt.labs.skills.network"
    api_url = base_url+'/speech-to-text/api/v1/recognize'

    params = { 
        'model': 'en-US_multimedia',
    }

    body = audio_binary
    response = requests.post(api_url, params=params, data=audio_binary).json()
    text='null'
    while bool(response.get('results')):
        print('speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text 
    return None


def text_to_speech(text, voice=""):
    base_url = "https://sn-watson-tts.labs.skills.network"
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice != "" and voice !="default":
        api_url+="&voice"+voice

    headers = {
        'Accept': 'audi0/wav',
        'Content-type' : 'application/json',
    }

    json_data = {
        'text': text,
    }

    response = requests.post(api, headers=headers, json=json_data)
    print('text to speech response:', response)
    return respons.content


    return None


def openai_process_message(user_message):
    return None
