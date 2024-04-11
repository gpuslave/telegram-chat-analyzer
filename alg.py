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


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf8") as chatFile:
            if (chatFile.name.lower().endswith(".json")):
                try:
                    return json.loads(chatFile.read())
                except json.JSONDecodeError:
                    print("File is not a valid JSON file")
                    linger_with_exit(1)
            else:
                print("File is not a JSON file")
                linger_with_exit(1)

    except FileNotFoundError:
        print("File not found")
        linger_with_exit(1)


def create_csv(distribution_set, boundary=27):
    for key in distribution_set.keys():
        k = 0
        with open(str(key) + '.csv', 'w', newline='',
                  encoding='utf-8') as new_csv:
            z = csv.writer(new_csv)
            z.writerow(["word", "entries"])
            for new_k, new_v in distribution_set[key].items():
                if k >= boundary:
                    break
                z.writerow([new_k, new_v])
                k += 1


if len(sys.argv) < 2:
    print("Usage: python alg.py <path to json file>")
    linger()
    sys.exit(1)

FILE_PATH = sys.argv[1]
parsed = read_json_file(FILE_PATH)

you_name, you_id, person_name, person_id = find_ids(
    parsed["messages"], parsed["id"])


print("You: " + you_name + " (" + str(you_id) + ")")
print("Person: " + person_name + " (" + str(person_id) + ")")
chat_name = "Chat with " + person_name
print(chat_name)


distribution = {
    you_id: {},
    person_id: {},
}

for message in parsed["messages"]:
    message_string = ""

    # if a message does not contain any plain text entities then skip
    if not message["text_entities"]:
        continue

    # finding plain text entity
    for entity in message["text_entities"]:
        if entity["type"] == "plain":
            message_string = entity["text"].lower()
            break

    # pattern = r"(\W+)|(\d)"
    PATTERN = r'\W+'
    message_list = list(filter(None, re.split(PATTERN, message_string)))

    # if no plain text words in the message then skip
    if not message_list:
        continue

    # not to confuse with the id of the message itself
    message_from_id = int(message["from_id"][4:])

    for word in message_list:
        distribution[message_from_id][word] = \
            distribution[message_from_id].get(word, 0) + 1


for key in distribution.keys():
    distribution[key] = dict(
        sorted(distribution[key].items(), key=lambda x: x[1], reverse=True))

create_csv(distribution)


you_df = pd.read_csv(str(you_id) + ".csv")
person_df = pd.read_csv(str(person_id) + ".csv")

fig, axs = matplotlib.pyplot.subplots(nrows=3)
sns.barplot(x="word", y="entries", data=you_df, ax=axs[0])
sns.barplot(x="word", y="entries", data=person_df, ax=axs[1])
sns.barplot(x="word", y="entries", data=you_df, ax=axs[2])
sns.barplot(x="word", y="entries", data=person_df, ax=axs[2])
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
