import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('ML\\rus_words.csv')
df = shuffle(df)
df.to_csv('ML\\shuffled_rus_words.csv', index=False)
