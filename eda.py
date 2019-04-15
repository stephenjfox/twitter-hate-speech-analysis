import pandas as pd

train = pd.read_csv('data/train_E6oV3lV.csv')

def spacer(char: str = '-', count=5):
    return char*count

def big_break():
    print('\n'*5)

print('Number of rows =', len(train))


print(spacer(), 'Counts', spacer())
positive_tweets = train[train['label'] == 0]
print('Positive labels:', len(positive_tweets)) # we found __no__ vvvv
negative_tweets = train[train['label'] == 1]
print('Negative labels:', len(negative_tweets)) # we found sexist/racial remarks
print('Ratio (p:n):', len(positive_tweets)/len(negative_tweets))

print(spacer(), 'Sample', spacer())
print(train.head())

## 1.1 Number of Words
train['word_count'] = train['tweet'].apply(lambda s: len(str(s).split(' ')))

print(train[['tweet', 'word_count']].head())


## 1.1a My experiment: average_word_length
big_break()
import functools

def average_word_length(tweet: str):
    split = str(tweet).split(' ')
    num_words = len(split) # roughly. Punctuation isn't a concern
    avg = functools.reduce(lambda a,b: a + b, (len(s) for s in split), 0) / num_words
    return avg

train['average_word_length'] = train['tweet'].apply(average_word_length)

print("Added average word length. Let's see our new features")
print(train.head())