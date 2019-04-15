import pandas as pd

train = pd.read_csv('data/train_E6oV3lV.csv')

def spacer(char: str = '-', count=5):
    return char*count

def big_break():
    print('\n'*5)

print('Number of rows =', len(train))


print(spacer(), 'Counts', spacer())
positive_tweets = train[train['label'] == 0]
print('Positive labels:', len(positive_tweets)) # we found __no__ sexist/racial remarks
negative_tweets = train[train['label'] == 1]
print('Negative labels:', len(negative_tweets)) # we found sexist/racial remarks
print('Ratio (p:n):', len(positive_tweets)/len(negative_tweets))

print(spacer(), 'Sample', spacer())
print(train.head())

## 1.1 Number of Words
train['word_count'] = train['tweet'].apply(lambda s: len(str(s).split(' ')))

print(train[['tweet', 'word_count']].head())


## 1.1a/1.3 My experiment: average_word_length
big_break()
import functools

def average_word_length(tweet: str):
    split = str(tweet).split(' ')
    num_words = len(split) # roughly. Punctuation isn't a concern
    avg = sum((len(s) for s in split)) / num_words
    return avg

train['average_word_length'] = train['tweet'].apply(average_word_length)

print("Added average word length. Let's see our new features")
print(train.head())

## 1.2 Character count

train['char_count'] = train['tweet'].str.len()

print('Added character count')
print(train.head())


## 1.4 Number of stopwords
big_break()

### start: mandatory install
### [TODO]: read the NLTK api and figure
import nltk
nltk.download('stopwords')
### end: mandatory install

from nltk.corpus import stopwords
stop = stopwords.words('english')

### stop words count
train['stopwords_count'] = train['tweet'].apply(lambda s: len([x for x in s.split() if x in stop]))
print('Added stop words count')
print(train.head())

### [TODO]: FIXME Why doesn't this work? I should be able to see the words as a list-columns...
# __test_stop_words_sample = train['tweet'].apply(lambda tweet: [x for x in tweet.split() if x in stop])
# print('Sampling some stop words')
# print(pd.concat([train, __test_stop_words_sample]).head())
### FIXME: see above

## 1.5 Number of hashtags

train['hashtag_count'] = train['tweet'].apply(lambda tweet: len([x for x in tweet.split() if x.startswith('#')]))

print('Added hashtag count')
print(train[['tweet', 'hashtag_count']].head())

## 1.5a Me: Number of user mentions
big_break()
train['mention_count'] = train['tweet'].apply(lambda tweet: len([x for x in tweet.split() if x.startswith('@')]))
print('Added user mention count')
print(train[['tweet', 'mention_count']].head())