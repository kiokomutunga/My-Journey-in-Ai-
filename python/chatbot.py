import re
import random

# Define patterns and responses
patterns = {
    r"(hi|hello|hey)": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    r"how are you": ["I’m good, thanks for asking!", "Doing well! How about you?"],
    r"(what is your name|who are you)": ["I’m your friendly chatbot 🤖", "You can call me ChatBot!"],
    r"(bye|goodbye)": ["Goodbye!", "See you later!", "Bye! Have a nice day!"]
}

def chatbot_response(user_input):
    user_input = user_input.lower()
    for pattern, responses in patterns.items():
        if re.search(pattern, user_input):
            return random.choice(responses)
    return "I’m not sure I understand. Can you rephrase?"

# Chat loop
print("Kiokoai: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "goodbye"]:
        print("Kiokoai:", random.choice(patterns[r"(bye|goodbye)"]))
        break
    response = chatbot_response(user_input)
    print("ChatBot:", response)
