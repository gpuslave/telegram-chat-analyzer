""" Usage: .\\alg.py \\path\\to\\json

Generates overwiew of the most used words in the telegram chat.
"""

import json
import re
import csv
import sys
import matplotlib
# import pyarrow
# import numpy as np
import pandas as pd
import seaborn as sns


def linger():
    input("Press Enter to continue...")


def linger_with_exit(error_code):
    input("Press Enter to exit...")
    exit(int(error_code))


# def find_you(messages, person_id):
#     for message in messages:
#         if int(message["from_id"][4:]) != person_id:
#             return message["from_id"], message["from"]


# def find_person(messages, person_id):
#     for message in messages:
#         if int(message["from_id"][4:]) != person_id:
#             return message["from_id"], message["from"]


def find_ids(messages, person_id):
    """ Finds the ids and names of the two people in the chat

    output: you_name, you_id, person_name, person_id
    """
    person_name = ""
    you_name = ""
    you_id = -1
    found_person = False
    found_you = False

    # O(n)
    for message in messages:
        if int(message["from_id"][4:]) != person_id and not found_you:
            you_name = message["from"]
            you_id = int(message["from_id"][4:])
            found_you = True
        elif not found_person:
            person_name = message["from"]
            found_person = True

        if found_you and found_person:
            return you_name, you_id, person_name, person_id


if len(sys.argv) < 2:
    print("Usage: python alg.py <path to json file>")
    linger()
    sys.exit(1)

FILE_PATH = sys.argv[1]
try:
    with open(FILE_PATH, "r", encoding="utf8") as chatFile:
        if (chatFile.name.lower().endswith(".json")):
            content = chatFile.read()
        else:
            print("File is not a JSON file")
            sys.exit(1)

except FileNotFoundError:
    print("File not found")
    linger()
    exit(1)


parsed = json.loads(content)

you_name, you_id, person_name, person_id = find_ids(
    parsed["messages"], parsed["id"])

chat_name = "Chat with " + person_name
print(chat_name)

print("You: " + you_name + " (" + str(you_id) + ")")
print("Person: " + person_name + " (" + str(person_id) + ")")


linger()
exit(0)

distribution_sunset = {}
distribution_kessie = {}

messages = parsed["messages"]
for message in messages:
    message_string = ""

    if not message["text_entities"]:
        continue

    for i in range(len(message["text_entities"])):
        if message["text_entities"][i]["type"] == "plain":
            message_string = message["text_entities"][i]["text"].lower()
            break

    # print(message["id"])
    # pattern = r"(\W+)|(\d)"
    PATTERN = r'\W+'
    message_list = list(filter(None, re.split(PATTERN, message_string)))

    if not message_list:
        continue

    # rename str!
    if message["from"][0:3] == 'sun':
        for str in message_list:
            if str in distribution_sunset.keys():
                distribution_sunset[str] += 1
            else:
                distribution_sunset[str] = 1
    else:
        for str in message_list:
            if str in distribution_kessie.keys():
                distribution_kessie[str] += 1
            else:
                distribution_kessie[str] = 1

distribution_sunset = dict(
    sorted(distribution_sunset.items(), key=lambda x: x[1], reverse=True))
distribution_kessie = dict(
    sorted(distribution_kessie.items(), key=lambda x: x[1], reverse=True))

BOUNDARY = 0
with open('sun.csv', 'w', newline='', encoding='utf-8') as new_csv:
    z = csv.writer(new_csv)
    z.writerow(["word", "entries"])
    for new_k, new_v in distribution_sunset.items():
        if BOUNDARY == 35:
            break
        z.writerow([new_k, new_v])
        BOUNDARY += 1
BOUNDARY = 0
with open('kes.csv', 'w', newline='', encoding='utf-8') as new_csv:
    z = csv.writer(new_csv)
    z.writerow(["word", "entries"])
    for new_k, new_v in distribution_kessie.items():
        if BOUNDARY == 35:
            break
        z.writerow([new_k, new_v])
        BOUNDARY += 1

# sns.set()

sun_dataframe = pd.read_csv("sun.csv")
kes_dataframe = pd.read_csv("kes.csv")

fig, axs = matplotlib.pyplot.subplots(nrows=3)
sns.barplot(x="word", y="entries", data=sun_dataframe, ax=axs[0])
sns.barplot(x="word", y="entries", data=kes_dataframe, ax=axs[1])
sns.barplot(x="word", y="entries", data=sun_dataframe, ax=axs[2])
sns.barplot(x="word", y="entries", data=kes_dataframe, ax=axs[2])
matplotlib.pyplot.show()
# df = pd.DataFrame.from_dict(distribution_sunset)
# df.to_csv (r'test.csv', index=False, header=True)
# x = int(input())
# print(distribution_sunset)
# print()
# print(distribution_kessie)

# SAVING TO A FILE
# fig, axs = matplotlib.pyplot.subplots(nrows=3)
# sns.barplot(x="word", y="entries", data=sun_dataframe, ax=axs[0])
# sns.barplot(x="word", y="entries", data=kes_dataframe, ax=axs[1])
# sns.barplot(x="word", y="entries", data=sun_dataframe, ax=axs[2])
# sns.barplot(x="word", y="entries", data=kes_dataframe, ax=axs[2])

# # Save the figure to a file
# matplotlib.pyplot.savefig('output.png')

# If you want to save each plot to a separate file, you can do so like this:
# fig1, ax1 = matplotlib.pyplot.subplots()
# sns.barplot(x="word", y="entries", data=sun_dataframe, ax=ax1)
# fig1.savefig('output1.png')

# fig2, ax2 = matplotlib.pyplot.subplots()
# sns.barplot(x="word", y="entries", data=kes_dataframe, ax=ax2)
# fig2.savefig('output2.png')
