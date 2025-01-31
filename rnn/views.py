from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = keras.models.load_model("rnn/chatbot_model.h5")

# Load the tokenizer
with open("rnn/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define the maximum sequence length (Make sure it's the same as in training)
max_sequence_length = 100
def generate_response(input_text, next_chars=50):
    for _ in range(next_chars):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)

        if predicted_index in tokenizer.index_word:
            predicted_char = tokenizer.index_word[predicted_index]
            input_text += predicted_char
        else:
            break  # Stop if prediction is invalid

    return input_text

# Django view to handle chatbot response
def chatbot_response(request):
    if request.method == "POST":
        user_input = request.POST.get("message", "")

        # Generate chatbot response
        response_text = generate_response(user_input)

        return JsonResponse({"response": response_text})

    return render(request, "index.html")
