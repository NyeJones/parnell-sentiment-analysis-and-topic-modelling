import glob
import pandas as pd

#loads df_heads file containing top 50 neutral/high sentiment scores for compound/non compound topic models of sentiment
path = "outputs/top_mod/df_heads/*.csv"

#create list of filenames for files loaded above
filenames = []
for filename in glob.glob(path):
    filenames.append(filename)
    
step = 4

filenames_years = [filenames[i:i+step] for i in range(0, len(filenames), step)]

#creates results for unique scores of compound results and non-compound results
#finds unique high sentiment/neutral sentiment scores for top topics
#removes terms present in both high sentiment/neutral
for files_year in filenames_years:

    files = []
    for file in files_year:
        files.append(file)
    
    #compound scores
    df_1 = pd.read_csv(files[0])
    df_1["score category"] = "unique neutral score"
    df_2 = pd.read_csv(files[1])
    df_2["score category"] = "unique high sentiment score"
    
    #unique compound score dataframe
    df_compound_both = df_1.append(df_2)
    df_compound_unique = df_compound_both.drop_duplicates(subset=["term"], keep=False)
    df_compound_unique = df_compound_unique.rename(columns={"score": "combined score"})
    
    #saves results for each year to csv
    compound_year = df_compound_unique["date"].iloc[0]
    df_compound_unique.to_csv(f"outputs/top_mod/df_heads/unique_terms/{compound_year}_compound_top_mod_unique.csv", index=False)
    
    #non-compound scores
    df_3 = pd.read_csv(files[2])
    df_3["score category"] = "unique neutral score"
    df_4 = pd.read_csv(files[3])
    df_4["score category"] = "unique high sentiment score"
    
    #unique non-compound score dataframe
    df_non_compound_both = df_3.append(df_4)
    df_non_compound_unique = df_non_compound_both.drop_duplicates(subset=["term"], keep=False)
    df_non_compound_unique = df_non_compound_unique.rename(columns={"score": "combined score"})
    
    #saves results for each year to csv
    non_compound_year = df_non_compound_unique["date"].iloc[0]
    df_non_compound_unique.to_csv(f"outputs/top_mod/df_heads/unique_terms/{non_compound_year}_non_compound_top_mod_unique.csv", index=False)
    