#!/usr/bin/env python3
"""Script for Tkinter GUI chat client."""

from tkinter import *
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import tkinter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from sklearn.externals import joblib

from nltk.tokenize import RegexpTokenizer
import json
from random import randint

'''Declaring tokenizer'''
tokenizer = RegexpTokenizer(r'\w+')

'''Declaring stop words for keyword finding and topics list for recommendations'''
stopwords = []
stoplist = open("spacy_gensim.txt", "r")
topics = []
topicslist = open("topics.txt", "r")
keywords = []

'''Importing joke databases'''
with open("wocka.json", "r") as json_file:
    jokesDatabase1 = json.load(json_file)
    json_file.close()

with open("stupidstuff.json", "r") as json_file:
    jokesDatabase2 = json.load(json_file)
    json_file.close()

with open("reddit_jokes.json", "r") as json_file:
    jokesDatabase3 = json.load(json_file)
    json_file.close()

'''Importing stop words and topics from database'''
for word in stoplist:
    stopwords.append(word.rstrip("\n"))

for word in topicslist:
    topics.append(word.rstrip("\n"))

def receive():
    """Handles receiving of messages."""
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode("utf8")
            #msg = client.recv(BUFSIZ).decode("utf8")
            global keywords
            keywords = [word.lower() for word in tokenizer.tokenize(msg) if word.lower() not in stopwords]
            msg_list.insert(tkinter.END, rf_clf.predict([msg]))
            ss = sid.polarity_scores(msg)
            for k in sorted(ss):
                msg_list.insert(tkinter.END, k)
                msg_list.insert(tkinter.END, ss[k])
            #msg_list.insert(tkinter.END, rf_clf.predict_proba([msg]))
            #msg_list.insert(tkinter.END, rf_clf.classes_)
            msg_list.insert(tkinter.END, msg)
            print(getJoke())
        except OSError:  # Possibly client has left the chat.
            break


def send(event=None):  # event is passed by binders.
    """Handles sending of messages."""
    msg = my_msg.get()
    my_msg.set("")  # Clears input field.
    client_socket.send(bytes(msg, "utf8"))
    if msg == "{quit}":
        client_socket.close()
        top.quit()

'''Getting joke from database'''
def getJoke():
    for index in range(len(keywords)):
        for i in range(len(jokesDatabase1)):
            if keywords[index] in jokesDatabase1[i]["keywords"]:
                return jokesDatabase1[i]["body"]
        for i in range(len(jokesDatabase2)):
            if keywords[index] in jokesDatabase2[i]["keywords"]:
                return jokesDatabase2[i]["body"]
        for i in range(len(jokesDatabase3)):
            if keywords[index] in jokesDatabase3[i]["keywords"]:
                return jokesDatabase1[i]["body"]

        '''Recommends topic if no joke is found'''
        return "How about talking about " + topics[randint(0, 14)] + "?"

def on_closing(event=None):
    """This function is to be called when the window is closed."""
    my_msg.set("{quit}")
    send()

    
rf_clf = joblib.load('rf.pkl')
sid = SentimentIntensityAnalyzer()
print("Loaded ML models...")   
    
#----Now comes the sockets part----
HOST = input('Enter host: ')
PORT = input('Enter port: ')
if not PORT:
    PORT = 33000
else:
    PORT = int(PORT)

BUFSIZ = 1024
ADDR = (HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)

top = tkinter.Tk()
top.title("Social")

#######################################
#top.resizable(width=FALSE, height=FALSE)
#top.geometry('{}x{}'.format(460, 350))

#top_frame = Frame(top, bg='cyan', width = 450, height=50, pady=3).grid(row=0, columnspan=3)
#Label(top_frame, text = 'Model Dimensions').grid(row = 0, columnspan = 3)
#Label(top_frame, text = 'Width:').grid(row = 1, column = 0)
#Label(top_frame, text = 'Length:').grid(row = 1, column = 2)
#entry_W = Entry(top_frame).grid(row = 1, column = 1)
#entry_L = Entry(top_frame).grid(row = 1, column = 3)
########################################
top.configure(background = "RoyalBLue2")

messages_frame = tkinter.Frame(top)
my_msg = tkinter.StringVar()  # For the messages to be sent.
my_msg.set("Type your messages here.")
scrollbar = tkinter.Scrollbar(messages_frame)  # To navigate through past messages.
# Following will contain the messages.
msg_list = tkinter.Listbox(messages_frame, height=30, width=80,bg = "white", yscrollcommand=scrollbar.set)

scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack(pady = 10, padx = 10)
messages_frame.pack(pady = 10)

entry_field = tkinter.Entry(top, textvariable=my_msg, width = "30")
#entry_field.config(width = "10")
entry_field.bind("<Return>", send)
entry_field.pack()
command= lambda: send(client_socket)
send_button = tkinter.Button(top, text="Send", command=send, fg = "IndianRed1" )
send_button.config(width = "8" , height = "2", background = "pink")
send_button.pack()

top.protocol("WM_DELETE_WINDOW", on_closing)



receive_thread = Thread(target=receive)
receive_thread.start()
tkinter.mainloop()  # Starts GUI execution.