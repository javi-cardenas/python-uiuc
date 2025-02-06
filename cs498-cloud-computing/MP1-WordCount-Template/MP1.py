import random 
import os
import string
import sys

stopWordsList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

delimiters = " \t,;.?!-:@[](){}_*/"

def getIndexes(seed):
    random.seed(seed)
    n = 10000
    number_of_lines = 50000
    ret = []
    for i in range(0,n):
        ret.append(random.randint(0, 50000-1))
    return ret

def process(userID):
    indexes = getIndexes(userID)
    ret = []
    
    # TODO
    # Split each sentence into a list of words
    sentences = sys.stdin.readlines()
    words_in_sentences = []
    for sentence in sentences:
        for delimiter in delimiters:
            sentence = sentence.lower().replace(delimiter, ' ').strip()
        tokens = sentence.split(" ")
        words = [token.lower().strip() for token in tokens if token.isalpha() and token not in stopWordsList] # ignore common words
        words_in_sentences.append(words)

    # Track word frenquencies for only certain titles
    word_count = {}
    for index in indexes:
        sentence = words_in_sentences[index]
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1

    # Sort words in desc order
    words_sorted = dict(sorted(word_count.items(), key=lambda item: (-item[1], item[0])))
    ret = [x for x, y in tuple(words_sorted.items())[:20]]

    for word in ret:
        print(word)

process(sys.argv[1])
