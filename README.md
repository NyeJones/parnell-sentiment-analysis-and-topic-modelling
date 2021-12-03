## Data
The dataset currently comprises 349 TEI P5 XML records relating to reports of Parnell's speeches transcribed from various sources. For each report, metadata is recorded on its publisher and date and place of publication. A separate authority file is kept for the records of the speeches themselves.

## Running the scripts
The project uses the Python programming language for three scripts:

* parnell_sentiment_analysis.py
* parnell_topic_model.py
* parnell_topic_model_unique_high_low_sentiment.py

The scripts should be run in the order given above.

## Libraries

The scripts use a number of Python libraries which are not part of the standard and will need to be installed using [pip](https://pip.pypa.io/en/stable/) on the command line in the following ways:

```bash
pip install beautifulsoup4

pip install matplotlib

pip install natsort

pip install nltk

pip install pandas

pip install scikit-learn
```

## Sentiment Analysis Code and Outputs
The sentiment analysis script primarily uses the Vader tool from the Natural Language Toolkit library. Vader uses a pretrained lexicon to define words as having a degree of positive or negative sentiment, and also has the capacity to take into account changes in the meaning of words due to sentence structure. In our use of the tool we focus on the level of sentiment in general, whether positive or negative.

Outputs are in CSV format with rows containing the data for each sentence from each source. They can be found [here](https://github.com/NyeJones/parnell-sentiment-analysis-and-topic-modelling/tree/main/code/outputs). These outputs are split into those using the general 'compound' score and those just using positive and negative scores. For non-compound data, positive and negative scores are combined, with negative changed to positive, to create a purely positive 'high sentiment' score. In all cases outputs are split into high and low/neutral sentiment CSV files.

Adjustments to the lexicon used by the script can be made in sentiment_analyser_edit_lexicon.json

## Topic Modelling Code and Outputs
The topic modelling script takes the CSV outputs created above and uses a Term Frequency Inverse Document Frequency (TFIDF) tool taken from the scikit-learn (sklearn) library to extract the most prominent words in each source. 

Each sentence is cleaned of anomalies that might disrupt the topic modelling process and duplicate sentences from different sources are removed from the corpus. These sentences are then tokenized before being fed into the sklearn vectorizer to get a TFIDF score for each word. The words under consideration are limited to the top 200 words across the corpus by the vectorizer.

The results for each term are grouped by source year and then the scores for each term within that year are added together to give a combined term score. There are eight CSV outputs for each year. They can be found [here](https://github.com/NyeJones/parnell-sentiment-analysis-and-topic-modelling/tree/main/code/outputs/top_mod). These grouped results are used to create four visualisations of the top combined sentiment terms for each year as additional outputs, with a limitation of the top 50 terms for each visualisation. The combined results give us a clearer idea of which terms are most prominent across each year. The visualisations can be found [here](https://github.com/NyeJones/parnell-sentiment-analysis-and-topic-modelling/tree/main/code/outputs/top_mod/images).

The topic model script also outputs four CSV files for each year containing the max top 50 scores for each combined sentiment CSV. They can be found [here](https://github.com/NyeJones/parnell-sentiment-analysis-and-topic-modelling/tree/main/code/outputs/top_mod/df_heads). These files are used by the parnell_topic_model_unique_high_low_sentiment.py script to find the unique top high and neutral sentiment terms for each year from the combined results for each year. These files can be found [here](https://github.com/NyeJones/parnell-sentiment-analysis-and-topic-modelling/tree/main/code/outputs/top_mod/df_heads/unique_terms).
