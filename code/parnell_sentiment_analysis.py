import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path
from natsort import os_sorted

#create a filepath to the directory
dir_path = Path("../sources/")

#get filepaths to all the xml files in the directory
xml_files = (file for file in dir_path.iterdir() if file.is_file() and file.name.lower().endswith('.xml'))

#sorts files numerically as in a system file directory
xml_files = os_sorted(xml_files)

#basename returns filename removing directory path.
#split to remove ".xml" extension so that we can use later in dataframe
filenames = []
for path in xml_files:
    filename = os.path.basename(path)
    filename = filename.split(".")[0]
    filenames.append(filename)

#open xml files and convert into beautiful soup object for text extraction from xml elements
all_docs = []
for file in xml_files:
    with file.open('r', encoding='utf-8') as xml:
        doc = BeautifulSoup(xml, "lxml-xml")
        all_docs.append(doc)

#load nltk sentence detector for dividing text into sentences
#load full stop abbreviations to sentence detector to prevent them from ending sentences 
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(["hon", "mr", "rev", "dr", "m.p", "c.s", "c.v", "c.e", "t.l", "j.r", "j.j", "a.j",
                                "r.b", "j.g", "j.l", "j.r" "patk", "j.f", "n.b", "p.j", "c.j", "t.d", "r", "p.p",
                                "c.c", "wm", "capt"                                ])
sentence_tokenizer = PunktSentenceTokenizer(punkt_param)   

#new words added to sentiment analyser with polarity score, load dictionary from external json file
with open("sentiment_analyser_edit_lexicon.json") as data:
    sent_analyser_words = json.load(data)

#initialze sentiment analyser       
sid = SentimentIntensityAnalyzer()

#update sentiment analyser lexicon with new words from json file opened above
sid.lexicon.update(sent_analyser_words)

#lists for dataframes for each sentence in corpus
files = []
speech_ids = []
years = []
sentence_list = []
scores = []

#extract filename for each file
#extract text from each file with corresponding filename
#extract speech id that file refers to
#extract year of source publication
file_doc = zip(filenames, all_docs)
for file, doc in file_doc:      
    speech = doc.find("term", {"key": True})
    speech = speech["key"]
    date = doc.find("date")
    year_text = date["when"]
    year = re.match(r"\d{4}", year_text)
    if year != None:
        year = year.group()
    else:
        pass
    
    #extract body text from body element, remove bracketed text
    #clean text to ensure words are separate and divide text into sentences
    #perform sentiment analysis on sentence using model initialised above
    #append filename, speech_id, publication year, sentence and score to above lists for each sentence  
    body = doc.find_all("body")
    for text in body:
        text = text.get_text()
        text_non_bracket = re.sub(r"\(.*?\)|\[.*?\]", "", text)
        text_clean = text_non_bracket.strip().replace("\n", " ").replace("-", " ").replace("—", " ")
        sentences = sentence_tokenizer.tokenize(text_clean)
        for sentence in sentences:
            files.append(file)
            speech_ids.append(speech)
            years.append(year)
            sentence_list.append(sentence)
            score = sid.polarity_scores(sentence)
            scores.append(score)

#creation of word lists so we can see how model judges polarity in relation to our corpus  
#split sentence words for the corpus into separate tokens
#use sid.polarity scores to find the individual sentiment score of each word as given by the sentiment model
#append to positive, negative or neutral word list accordingly
#word scores run from -4 to 4
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for sentence in sentence_list:
    tokenized_sentence = word_tokenize(sentence)
    for word in tokenized_sentence:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

#remove duplicate words from each list            
pos_words_unique = list(set(pos_word_list))
neu_words_unique = list(set(neu_word_list))
neg_words_unique = list(set(neg_word_list))

#make all words lower case
pos_words_unique = [item.lower() for item in pos_words_unique]
neu_words_unique = [item.lower() for item in neu_words_unique]
neg_word_unique = [item.lower() for item in neg_words_unique]

#sort all words in lists alphabetically
pos_words_unique.sort()
neu_words_unique.sort()
neg_words_unique.sort()


#create text file with sentiment classification of all corpus words
#this enables us to see how the classifier in its current form classifies words
#we can then amend the json file used above to improve performance of sentiment analysis by including new words/scores
with open("outputs/word_sentiments_parnell.txt", "w") as f:
    f.write("positive polarity words")
    f.write(str(pos_words_unique))
    f.write("\n")
    f.write("neutral polarity words")
    f.write(str(neu_words_unique))
    f.write("\n")
    f.write("negative polarity words")
    f.write(str(neg_words_unique))
    f.write("\n")
    
#sentiment score for each sentence returns a dictionary of different scores
#divide dictionary values into lists for each score
#we can then use these lists to create any number of dataframes for different perspectives on the data
neg = []
neu = []
pos = []
compound = []
    
for score in scores:
    for key, value in score.items():
        if key == "neg":
            neg.append(value)
        if key == "neu":
            neu.append(value)
        if key == "pos":
            pos.append(value)
        if key == "compound":
            compound.append(value)
            
#create dictionary then dataframe without neutral and add positive and negative together for strong emotions using lists above
#this enables us to see the strong emotion scores for each sentence without compound or neutral scores
strong_emotion_data = {}

strong_emotion_data["file"] = files
strong_emotion_data["year"] = years
strong_emotion_data["speech_id"] = speech_ids
strong_emotion_data["sentence"] = sentence_list        
strong_emotion_data["negative"] = neg
strong_emotion_data["positive"] = pos                       
        
#create pandas dataframe to make data exports/manipulation easier
strong_df = pd.DataFrame(strong_emotion_data)

#remove sentences that have been turned into full stops by data cleaning
strong_df = strong_df[strong_df.sentence != "."]

#set index as speech_id
strong_df.set_index("speech_id", inplace=True)

#convert negative into positive and add columns together, create new sum column for results
strong_df["negative"] = strong_df["negative"].abs()
sum_column = strong_df["positive"] + strong_df["negative"]
strong_df["sum positive/negative"] = sum_column

#save dataframe to csv with all results
strong_df.to_csv("outputs/parnell_vader_strong_non_compound_scores_sentiment_all.csv")

#divide dataframe into low scoring and high scoring emotion, the parameters can be changed
strong_high_score_df = strong_df.loc[(strong_df["sum positive/negative"] >= 0.3)]
low_high_score_df = strong_df.loc[(strong_df["sum positive/negative"] <= 0.1)]

#save the above dataframes to csv files
strong_high_score_df.to_csv("outputs/parnell_vader_strong_non_compound_scores_sentiment_high.csv")
low_high_score_df.to_csv("outputs/parnell_vader_strong_non_compound_scores_sentiment_low.csv")
            
#create main dataframe, which includes/uses neutral and compound scores.
#compound score takes into account sentence syntax for creating a positive/negative score
data = {}

data["file"] = files
data["year"] = years
data["speech_id"] = speech_ids
data["sentence"] = sentence_list        
data["negative"] = neg
data["neutral"] = neu
data["positive"] = pos
data["compound score"] = compound  

#create pandas dataframe to make data exports/manipulation easier
df = pd.DataFrame(data)

#remove sentences that have been turned into full stops by data cleaning
df = df[df.sentence != "."]   

#create new dataframes with high and neutral sentiment windows, divide data accordingly
#windows can be changed
df_pos_neg = df.loc[(df['compound score'] <= -0.8) | (df['compound score'] >= 0.8)]
df_neutral = df.loc[(df['compound score'] >= -0.2) & (df['compound score'] <= 0.2)]

#save dataframes to csv, index to speech_id
df_pos_neg.set_index("speech_id", inplace=True)
df_pos_neg.to_csv("outputs/parnell_vader_pos_neg_speech_sentiment_analysis_scores.csv")

df_neutral.set_index("speech_id", inplace=True)
df_neutral.to_csv("outputs/parnell_vader_neutral_speech_sentiment_analysis_scores.csv")
