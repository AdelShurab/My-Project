from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("emotion_detector")

@app.route("/emotionDetector")
def emotion_detector_function():
    """
    Analyzes the provided text and returns the detected emotions and the dominant emotion.
    """
    text_to_analyze = request.args.get('TextToAnalyze')
    
    if text_to_analyze is None:
        return "No text provided. Please provide text to analyze."

    response = emotion_detector(text_to_analyze)
    
    if response['dominant_emotion'] is None:
        response_text = "Invalid Input! Please try again."
    else:
        response_text = (
            f"For the given statement, the system response is 'anger': {response['anger']}, "
            f"'disgust': {response['disgust']}, 'fear': {response['fear']}, 'joy': {response['joy']}, "
            f"'sadness': {response['sadness']}. The dominant emotion is {response['dominant_emotion']}."
        )
    
    return response_text

@app.route("/")
def render_index_page():
    """
    Renders the index.html template.
    """
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
