import csv
import sys
import pandas as pd
from sklearn.utils import shuffle
from itertools import permutations

s = sys.argv[1]

with open('ML\\rus_words.csv', "a", newline='\n',
          encoding='utf-8') as rus_words:
    csv_writer = csv.writer(rus_words)

    perms = [''.join(p) for p in permutations(s)]
    perms = list(set(perms))
    for p in perms:
        csv_writer.writerow([p, 1])

# with open('words.csv', "r", encoding='utf-8') as words:
#     reader = csv.reader(words)
#     words = list(reader)
#     words = [word[0] for word in words]
#     with open("ML\\rus_words.csv",
#               "w", newline='', encoding='utf-8') as rus_words:
#         csv_writer = csv.writer(rus_words)
#         csv_writer.writerow(["word", "is_laugh"])
#         for word in words:
#             csv_writer.writerow([word, "0"])


# df = pd.read_csv('ML\\rus_words.csv')
# df = shuffle(df)
# df.to_csv('ML\\shuffled_rus_words.csv', index=False)
