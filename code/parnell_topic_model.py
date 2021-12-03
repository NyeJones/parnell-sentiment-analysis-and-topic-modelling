import string
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

pd.options.mode.chained_assignment = None

#topic modelling performed on both normal vader sentiment scores and scores without neutral
filenames = [
            "outputs/parnell_vader_neutral_speech_sentiment_analysis_scores.csv", 
            "outputs/parnell_vader_pos_neg_speech_sentiment_analysis_scores.csv",
            "outputs/parnell_vader_strong_non_compound_scores_sentiment_low.csv",
            "outputs/parnell_vader_strong_non_compound_scores_sentiment_high.csv"
            ]            

#create dataframe from sentiment analysis csv data
sentiment_dfs = []
for file in filenames:
    df = pd.read_csv(file)
    sentiment_dfs.append(df)

#clean text to make consistent and remove punctuation to help with removing duplicates
#duplicates can have slightly different punctuation
#remove duplicate sentences from different sources to improve topic modelling performance
for df in sentiment_dfs:
    df["no_punct_sentence"] = df["sentence"].str.replace("[^\w\s]", "", regex=True).replace('\n', ' ').replace(u'\xa0', u' ').replace('& ', 'and ').replace('—', ' ')
    df.drop_duplicates("no_punct_sentence", inplace = True)    

# basename returns filename removing directory path.
# split to remove ".xml" extension so that we can use it for charts etc further down
file_bases = []
for file in filenames:
    filename = os.path.basename(file)
    filename = filename.split(".")[0]
    file_bases.append(filename)
    
#loop through dataframes for each sentiment analysis category
#file bases looped through for use in visualisations
for sent_df, file_base in zip(sentiment_dfs, file_bases):
    
    #create lists for all sentences in file and their years
    sentences = list(sent_df["no_punct_sentence"])
    years = list(sent_df["year"])

    #create stopwords and extend to remove common words specific to corpus that will not aid analysis
    #this list can be extended following initial results and then run code again for better results
    nltk_stopwords = stopwords.words("english")
    nltk_stopwords.extend([
                            "hon", "would", "hear", "whether", "member", "cheers", "moved", "applause",
                            "mr", "said", "gentleman", "debate", "adjournment", "house", "member", "mp",
                            "objection", "members", "could", "never", "us", "may", "upon", "better", "shall",
                            "confident", "brought", "say", "way", "important", "well", "great", "see", "think",
                            "show", "however", "also", "little", "received", "ever", "many", "might", "means"
                            "called", "held", "view", "considered", "course", "called", "part", "one", "must"
                            "day", "part", "present", "put", "done", "sir", "believe", "name", "three", "without"
                            "true", "give", "came", "come", "thought", "certainly", "last", "made", "best", "wish"
                            "like", "done", "kind", "know", "regard", "much", "know", "ask", "last", "consider",
                            "made", "must", "told", "since", "let", "led"
                            ])

    #tokenize text into individual words, omitting any stopwords as well
    non_stop_sents = []
    for sentence in sentences:
        non_stop_tokens = [i for i in sentence.lower().split() if i not in nltk_stopwords]
        non_stop_sent = " ".join(non_stop_tokens)
        non_stop_sents.append(non_stop_sent)
    
    
    #create vectorizer to produce tfidf scores for contents of each file
    #max_df - maximum percentage of sentences a word can appear in - avoid too frequent words
    #min_df - minimum number of sentences with word - avoid too infrequent words
    #max_features - limit analysis to top words by frequency across all sentences
    #lowercase establishes word consistency across sentences      
    vectorizer = TfidfVectorizer(
                                    lowercase=True,
                                    max_df = 0.9,
                                    min_df = 3,
                                    ngram_range = (1,2),
                                    max_features = 200
                                )

    #apply vectorizer to list of file contents        
    transformed_docs = vectorizer.fit_transform(non_stop_sents)

    #tfidf results as numeric arrays
    transformed_docs_as_array = transformed_docs.toarray()

    #for each array get feature names and tfidf scores for words for each sentence
    #convert word and score into a list of tuples for each sentence
    #convert list of tuples into a dataframe for each sentence
    #remove words with score of zero
    dfs = []
    for doc in transformed_docs_as_array:
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))       
        doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=["term", "score"]).sort_values(by="score", ascending=False).reset_index(drop=True)
        doc_as_df.drop(doc_as_df.index[20:], inplace=True)
        doc_non_zero = doc_as_df.loc[(doc_as_df['score'] != 0)]
        dfs.append(doc_non_zero)
    
    #add publication year of source for each sentence
    #sentences will ultimately be divided by year to find patterns by timeframe
    dfs_2 = []
    for year, df in zip(years, dfs):
        df["year"] = year
        dfs_2.append(df)
               
    #combine results for each sentence into a single dataframe, maintain row as sentence results
    #group sentences by year
    dfs_3 = [df.set_index("year") for df in dfs_2]
    df_all = pd.concat(dfs_3)
    df_grouped = df_all.groupby("year")
    
    #loop for csv and visualisation outputs for each year group of sentences
    for name, group in df_grouped:

        #save initial results to csv
        group.to_csv(f"outputs/top_mod/{name}_{file_base}_topic_model.csv")

        #create new csv which combines the scores for terms with high tfidf scores across multiple sentences
        combined_df = group.groupby("term").sum().sort_values(by="score", ascending=False)
        combined_df.to_csv(f"outputs/top_mod/{name}_{file_base}_combined_score_topic_model.csv")
                
        #set font for chart
        font = {"fontname":"Arial"}
        
        #limit final combined term results across all sentences for purpose of good visualisation, number can be amended
        comb_df_head = combined_df.head(n=50)
        
        #create axis for bar chart
        ax = comb_df_head.plot.bar(legend=None)
        
        #remove outline on the right and top of chart
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        #set titles and font for all text in chart
        plt.title(f"{name} {file_base} \n Highest Scoring Sentence Topics", **font, fontsize=10)
        plt.xlabel("Terms", **font)
        plt.ylabel("Combined TFIDF Score", **font)
        plt.xticks(**font)
        plt.yticks(**font)
        
        #set tight layout to ensure all text visible and save/show plot
        plt.tight_layout()
        plt.subplots_adjust(0.2)
        plt.savefig(f"outputs/top_mod/images/{name}_{file_base}_highest_scoring_topics.png")
        plt.close()
        
        #output top 50 to csv
        comb_df_head.insert(1, "date", name)
        comb_df_head.to_csv(f"outputs/top_mod/df_heads/{name}_{file_base}_combined_score_df_head.csv")