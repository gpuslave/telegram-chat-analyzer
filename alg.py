import json
import re
import csv
import matplotlib
# import pyarrow
# import numpy as np
import pandas as pd
import seaborn as sns

'''
ersaddfas
'''
with open('C:\\Users\\lnemt\\Downloads\\Telegram Desktop\\ChatExport_2024-03-08\\result.json',
          encoding="utf8") as chatFile:
    content = chatFile.read()
# print(content[0:400])
parsed = json.loads(content)

chat_name = "Chat with " + parsed["name"]
print(chat_name)

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
