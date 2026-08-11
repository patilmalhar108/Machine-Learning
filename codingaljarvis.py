from nltk.chat.util import Chat, reflections
reflections = {
    "I am":"You are",
    "I":"You",
    "My":"Yours",
    "You are":"I am",
    "You":"Me",
    "Me":"You"
}

pairs = [
    [r"my name is (.*)", ["Hello %1, how are you?"]],
    [r"hi|hey|hello", ["Hello!", "Hey there!"]],
    [r"what is your name?", ["I am jarvis a chat bot"]],
    [r"how are you?", ["I am doing good, how about you?"]],
    [r"sorry (.*)", ["It's alright","Nevermind"]],
    [r"I am fine", ["Great how can I help?"]],
    [r"(.*) age?", ["I am a computer program"]],
    [r"(.*) created you?", ["I was created using python NLTK library"]],
    [r"(.*) sports|game", ["I like football and cricket"]],
    [r"quit", ["Bye, take care"]]
]
def chat():
    print("Hi, I am Jarvis. How can I help you?")
    chatbot = Chat(pairs, reflections)
    chatbot.converse()

if __name__ == "__main__":
    chat()