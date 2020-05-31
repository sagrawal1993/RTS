import rethinkdb as r
from rethinkdb.errors import RqlRuntimeError
import json
import bz2
import sys,os
import string
from nltk.corpus import stopwords, brown
from nltk import FreqDist
import nltk
import fuzzy
from dateutil.parser import parse
import numpy as np
import operator

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pickle

from py_ms_cognitive import PyMsCognitiveWebSearch
from summa import keywords

if len(sys.argv)!=2:
    print("Input Specification")
    print("Real_time_summarization.py interest_file.json")
    exit()

#Variables that contains the user credentials to access Twitter API 
access_token = "315806605-Nd55T3CGoqZCTDxVkYX1RUU2vkPurVdnhj6mplVg"
access_token_secret = "EURzArqkZwI9qb9Tir4WQyCU3vFguJcjI2XeKTQJQs"
consumer_key = "0DLj9lXFfRye5fzv3y9hA"
consumer_secret = "ydG8Xf7MgiC5YiGOVeJcGlD3jF08OB3kAojfaGhMEy0"

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    def __init__(self, rts, connection):
        self.rts_obj = rts
        self.selected_count = 0
        self.connection = connection

    def on_data(self, data):
        tweet = json.loads(data)
        #if (tweet.has_key('lang') and tweet['lang'] != 'en'):
        #    return True
        if (not tweet.has_key('text') or (tweet.has_key('lang') and tweet['lang'] != 'en')):
            return True 
        dt = parse(tweet['created_at'])
        dt = dt.date().isoformat().translate(None,'-')
        if (rts.update_tweet_map(tweet, dt, self.connection)):
            self.selected_count += 1
        return True

    def on_error(self, status):
        print "Error: ", status

    def get_count(self):
        return self.selected_count


def output_file(interest_file_json, conn):
    # interest_profile_file, score_threshold, 
    
    rts = RTS(interest_file_json, 0.2, 5.0)
    l = StdOutListener(rts,conn)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    #stream.filter(track=['python', 'javascript', 'ruby'])
    while True:
        try:
            stream.sample()
            if (l.get_count > 1000):
                return rts.get_tweet_map()
        except:
            pass 

def format_output(tweet_map):   
    print tweet_map
    result_file=""
    for day in tweet_map:
        for topic in tweet_map[day]:
            rank_list=[ (tweet_map[day][topic][i], i) for i in tweet_map[day][topic].keys() ]
            rank_list.sort(reverse=True)
            if len(rank_list)>100:
                rank_list=rank_list[:100]   
            count=1
            for entry in rank_list:
                result_file+=day+"\t"+str(topic)+"\tQ0\t"+str(entry[1])+"\t"+str(count)+"\t"+str(entry[0])+"\tISI\n"    
                count+=1
    fp=open("Judgement.txt","w")
    fp.write(result_file)
    fp.close()

def clean_line(line):
    line = line.lower()
    line = line.replace("'s", "")   # remove possesive apostrophe first (not to be confused with plurals)
    line = line.replace('\'', '')   # remove remaining apostrophes
    for char in string.punctuation: # remove punctuations with whitespace
        line = line.replace(char, ' ')
    line = line.split()
    #line = nltk.word_tokenize(line)
    return line

def get_123grams(str, ignore_set):
    word_vec=clean_line(str)
    word_set = set()
    prev_word = prev_two_words = None
    for word in word_vec:
        if word not in ignore_set:
            word_set.add(word)
            if prev_two_words != None:
                trigram = prev_two_words+"|"+word
                word_set.add(trigram)
            if prev_word != None:
                bigram = prev_word+"|"+word
                word_set.add(bigram)
                prev_two_words = bigram
            prev_word = word
    return word_set


def get_relevant_keywords(topic, search_string):

    if os.path.isfile(os.path.join('bing_search_results', topic+'.pkl')):
        print 'reading from file'
        with open(os.path.join('bing_search_results', topic+'.pkl')) as json_file:
            first_fifty_result = pickle.load(json_file)
    else:
        print 'writing to file'
        _API_KEY = '6d111315bc3d497392fe828bdd0208ec'
        search_service = PyMsCognitiveWebSearch(_API_KEY, search_string)
        first_fifty_result = search_service.search(limit=50, format='json')
        
        with open(os.path.join('bing_search_results', topic+'.pkl'), 'w') as json_file:
            pickle.dump(first_fifty_result, json_file)

    key_words = []
    final_keywords = []
    for result in first_fifty_result:
        try:
            snippet = str(result.snippet)
        except:
            continue
        print snippet
        try:
            new_keys = str(keywords.keywords(snippet))
        except:
            continue
        print new_keys
        key_words.extend(new_keys.split('\n'))
    for i, word in enumerate(key_words):
        new_word = '|'.join(word.split())
        final_keywords.append(new_word)
    return set(final_keywords)

class RTS(object):
        
    def __init__(self, file_name, score_threshold, redundancy_threshold):
        self.score_threshold = score_threshold
        self.redundancy_threshold = redundancy_threshold
        self.interest_profiles = InterestProfiles(file_name)
        self.interest_vectors = self.interest_profiles.get_weighted_terms()
        print self.interest_vectors
        self.removed_words = self.interest_profiles.get_removed_words()
        self.stop_words = set(stopwords.words("english"))
        self.soundex = fuzzy.Soundex(4) 
        self.tweet_map = {}
        self.seen_words = {}
        self.initialize_seen_words()
    
    def tweet_vector(self, tweet_json): 
        return get_123grams(tweet_json['text'], self.removed_words)

    def get_tweet_map(self):
        return self.tweet_map

    def initialize_tweet_map(self, date_string):
        self.tweet_map[date_string] = {}
        for topic in self.interest_vectors:
            self.tweet_map[date_string][topic] = {}

    def initialize_seen_words(self):
        for topic in self.interest_vectors:
            self.seen_words[topic] = {}

    def get_redundancy_score(self, tweet_vector, topic):
        return sum([self.seen_words[topic][x] for x in tweet_vector if x in self.seen_words[topic]])

    def update_seen_words(self, tweet_vector, topic):
        for word in tweet_vector:
            if word not in self.seen_words[topic] and word not in self.interest_vectors[topic]:
                # assign higher weight to bigrams and trigrams
                self.seen_words[topic][word] = len(word.split("|"))

    def update_tweet_map(self, tweet_json, date_string, conn):
        if date_string not in self.tweet_map:
            self.initialize_tweet_map(date_string)

        tweet_vector = self.tweet_vector(tweet_json)

        for topic in self.interest_vectors:
            score = self.interest_tweet_similarity(self.interest_vectors[topic], tweet_vector)

            if score > self.score_threshold:
                redundancy = self.get_redundancy_score(tweet_vector, topic)
                if redundancy < self.redundancy_threshold:
                    self.update_seen_words(tweet_vector, topic)
                    self.tweet_map[date_string][topic][tweet_json['id']] = score
                    notification = {}
                    notification['topic'] = topic
                    notification['tweet'] = tweet_json['text']
                    notification['score'] = score
                    notification['screen_name'] = tweet_json['user']['screen_name']
                    notification['user'] = tweet_json['user']['name']
                    notification['timestamp'] = tweet_json['created_at']
                    try:
                        print(r.table("mytable").insert(notification).run(conn))
                    except RqlRuntimeError as err:
                        print(err.message)
                    print tweet_json['text'], ": ", topic, ": ", score
                    return True
        return False

    def interest_tweet_similarity(self, interest_vector, tweet_vector):
        score = 0
        for word in tweet_vector:
            if word in interest_vector:
                score += interest_vector[word]
        return score

class InterestProfiles(object):
        
    def __init__(self, file_name):
        file_text = open(file_name).read()
        self.interest_json = json.loads(file_text)
        self.fluff_threshold = 2
        self.max_term_count = 20
        self.stop_words = set(stopwords.words("english"))
        self.topics = ["RT001", "RT002", "RT003", "RT004", "RT005"]
        self.narrative_word_count_dict = {}
        self.interest_vectors = {}
        self.get_interest_vectors()

    def get_removed_words(self):
        removed = set(self.stop_words)
        for word in self.narrative_word_count_dict:
            if self.narrative_word_count_dict[word] >= self.fluff_threshold:
                removed.add(word)
        return removed

    def get_interest_vectors(self):
        for interest in self.interest_json:
            for word in set(clean_line(interest['narrative'])):
                if word in self.narrative_word_count_dict:
                    self.narrative_word_count_dict[word] += 1
                else:
                    self.narrative_word_count_dict[word] = 1

        self.removed_words = self.get_removed_words()
        # for every topic store a dictionary mapping each term with a (weight, position) tuple
        for interest in self.interest_json:
            if interest["topid"] not in set(self.topics):
                continue
            self.interest_vectors[interest["topid"]] = {}

            title_set = get_123grams(interest['title'], self.removed_words)                        ############## TODO ##############
            description_set = get_123grams(interest['description'], self.removed_words)         ############## TODO ##############
            
            prev_word = None
            prev_two_words = None

            for word in clean_line(interest['narrative']):
                if word not in self.removed_words:
                    if word not in self.interest_vectors[interest["topid"]]:
                        self.interest_vectors[interest["topid"]][word] = get_word_weight(word)
                        
                        # Assign 2 times as much weight if word appears in title/description too
                        if word in title_set:
                            self.interest_vectors[interest["topid"]][word] *= 1.5
                        if word in description_set:
                            self.interest_vectors[interest["topid"]][word] *= 1.5
                    
                    bigram_weight = compute_bigram_weight(self.interest_vectors[interest["topid"]], prev_word, word)
                    if bigram_weight != None:
                        self.interest_vectors[interest["topid"]][bigram_weight[0]] = bigram_weight[1]
                        if bigram_weight[0] in title_set:
                            self.interest_vectors[interest["topid"]][bigram_weight[0]] *= 1.5
                        if bigram_weight[0] in description_set:
                            self.interest_vectors[interest["topid"]][bigram_weight[0]] *= 1.5

                    trigram_weight = compute_bigram_weight(self.interest_vectors[interest["topid"]], prev_two_words, word)
                    if trigram_weight != None:
                        self.interest_vectors[interest["topid"]][trigram_weight[0]] = trigram_weight[1]
                        if trigram_weight[0] in title_set:
                            self.interest_vectors[interest["topid"]][trigram_weight[0]] *= 1.5
                        if trigram_weight[0] in description_set:
                            self.interest_vectors[interest["topid"]][trigram_weight[0]] *= 1.5


                    if prev_word != None:
                        prev_two_words = prev_word+"|"+word
                    prev_word = word
            
            keywords = get_relevant_keywords(interest["topid"], interest['title']).difference(self.removed_words)
            for word in keywords:
                try:
                    self.interest_vectors[interest['topid']][word] *= 1.5
                except:
                    self.interest_vectors[interest['topid']][word] = 0.8
            
            # keep only the maximum weighted 20 terms
            self.interest_vectors[interest['topid']] = dict(sorted(self.interest_vectors[interest['topid']].iteritems(), key=operator.itemgetter(1), reverse=True)[:self.max_term_count])
            
            # Normalization of weight
            total = sum([x for x in self.interest_vectors[interest["topid"]].values()])
            for word in self.interest_vectors[interest["topid"]]:
                self.interest_vectors[interest["topid"]][word] = self.interest_vectors[interest["topid"]][word]*10.0/total
        
    def get_weighted_terms(self):
        return self.interest_vectors

def get_word_weight(word):
    # Assign weight based on word frequency
    if word in frequency_list :
        freq = frequency_list[word]
        return np.exp(freq*-1.0/1000)

    if all(c.islower() for c in word):
        # Word not present in corpus is likely a named entity (assign high preference)
        return 0.8
    return 0.3

def get_word_freq():
    return FreqDist(i.lower() for i in brown.words())

def compute_bigram_weight(dictionary, prev_word, word):
    if prev_word != None:
        bigram = prev_word+"|"+word
        if bigram not in dictionary:
            return (bigram, 1.5*(dictionary[word] + dictionary[prev_word]))
    return None

if __name__ == "__main__":
    conn = r.connect(host='localhost', port=28015, db='test')
    frequency_list = get_word_freq()
    glb_map=output_file(sys.argv[1], conn)
    format_output(glb_map)
    conn.close()