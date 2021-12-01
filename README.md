# Parnell-Speeches
A repository for a Cambridge Humanities Research Grant funded sentiment analysis and topic modelling project relating to the speeches of Charles Stewart Parnell, using the transcribed speeches of Parnell, from various sources, as a dataset. The focus of the project is on finding high sentiment  areas of text in general, rather than positive or negative sentiment.

## Data
The dataset currently comprises 349 TEI P5 XML records relating to reports of Parnell’s speeches transcribed from various reports of the speeches. Metadata is recorded for each report on its publisher, date when published and place of publication. A separate authority file is kept for the records of the speeches themselves.

## Running the files
The project uses the Python programming language (version 3.9.4) and incorporates two separate files with the .py extension, one for the sentiment analysis stage of the project and another for topic modelling. The sentiment analysis file should be run before the topic modelling file.

## Sentiment Analysis Code and Outputs
The sentiment analysis file primarily uses the Vader tool from the Natural Language Toolkit library. Vader uses a pretrained lexicon to define words as having a degree of positive or negative sentiment, and also has the capacity to take into account changes in the meaning of words due to sentence structure. In our use of the tool we focus on the level of sentiment in general, whether positive or negative.

Outputs are in CSV format with rows containing the data for each sentence from each source. These outputs are split into those using the general “compound” score and those just using positive and negative scores. For non-compound data, positive and negative scores are combined, with negative changed to positive, to create a purely positive “high sentiment” score. In all cases outputs are split into high and low/neutral sentiment CSV files.

## Topic Modelling Code and Outputs
The topic modelling file takes the CSV outputs created above and uses a Term Frequency – Inverse Document Frequency (TFIDF) tool taken from the scikit-learn (sklearn) library to extract the most prominent words in each source. 

Each sentence is cleaned of anomalies that might disrupt the topic modelling process and duplicate sentences from different sources are removed from the corpus. These sentences are then tokenized before being fed into the sklearn vectorizer to get a TFIDF score for each word. The words under consideration are limited to the top 200 words across the corpus by the vectorizer.

The results for each term are grouped by source year and then the scores for each term within that year are added together to give a combined term score. There are four CSV outputs for each year.

* (year) parnell_vader_neutral_speech_sentiment_analysis_scores
* (year) parnell_vader_pos_neg_speech_sentiment_analysis_scores
* (year) parnell_vader_strong_non_compound_scores_sentiment_high
* (year) parnell_vader_strong_non_compound_scores_sentiment_low

These grouped results are used to create four visualisations of the top high sentiment terms for each year as additional outputs, with a limitation of the top 50 terms for each visualisation.
