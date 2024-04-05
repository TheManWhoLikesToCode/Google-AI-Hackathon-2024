from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot('MyBot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot.get_response(user_input)
    print("Bot: ", response)