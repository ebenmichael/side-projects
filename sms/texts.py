# -*- coding: utf-8 -*-
"""
Looking at my text data
"""
import sqlite3 as lite
import pandas as pd
import os
import wordcloud
import matplotlib.pyplot as plt
from sklearn import feature_extraction
import numpy as np


def load_number(number,db):
    """Gets the texts messages from a given telephone number (or email address)
    Input:
        number: String. telephone number of email address
	db = name of sqlite database
    Output:
        dat: dataframe
    """
        
    #load data
    con = lite.connect('sms.db')
    #write the sql command
    command = "SELECT * FROM clean_message WHERE id = '%s';" %number
    with con:
        dat = pd.read_sql_query(command,con,parse_dates=True)
        dat.columns = ['date','id','text','from_me']
        #strip out any columns with null values
        dat = dat.dropna()
        return(dat)


def by_date(df):
    """Groups by date, gets counts, sums, and string concatenations
    Input:
        df: DataFrame from load_number
    Output:
        df_new: new DataFrame with new columns and grouped by date
    """
    #groupby date, get counts of messages sent, and counts of how much i sent
    df_new = df.groupby("date")["from_me"].agg(["count","sum"])
    #rename columns
    df_new.columns = ["total_texts","num_sent"]
    #get number recieved
    df_new["num_received"] = df_new["total_texts"] - df_new["num_sent"]
    #get texts aggregated
    df_new["texts"] = df.groupby("date")["text"].apply(lambda x: ". ".join(x))
    return(df_new)
    
def get_grouped_texts(number,db):
    """Gets text data grouped by date
        number: String. telephone number oe email address
	db = name of sqlite database
    Output:
        dat: dataframe
    """
    
    dat = by_date(load_number(number,db))
    return(dat)
    
def write_texts(number,outdir = None):
    """Writes all the texts from a day into a text file for all dates
    Input:
         number: String. telephone number oe email address
	 db = name of sqlite database 
         outdir: directory to write to. Defaults to pwd/number
    """
    #set outdir to current working directory/number if it is None
    if outdir is None:
        outdir = os.path.join(os.getcwd(),number)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #load data
    dat = get_grouped_texts(number,db)
    #iterate over dates
    for date,row in dat.iterrows():
        #file name is date.txt
        fname = os.path.join(outdir,date) + ".txt"
        text = row["texts"]
        with open(fname,"w") as f:
            f.write(text)
            
def wordclouds(number,db):
    """Makes a worcloud of word from me and other person
    Input:
        name: String. Name of person
    """
    #isolate text
    texts = load_number(number,db)
    me = " ".join(texts[texts["from_me"] == 1]["text"]).lower()
    other = " ".join(texts[texts["from_me"] == 0]["text"]).lower()
    
    #make figure
    plt.axis("off")
    fig = plt.figure(figsize = (16,9))
    ax1 = fig.add_subplot(211)
    ax1.imshow(wordcloud.WordCloud(width = 1600, height = 900).generate(me))
    ax1.set_title("Me")
    ax1.set_axis_off()
    ax2 = fig.add_subplot(212)
    ax2.imshow(wordcloud.WordCloud(width = 1600, height = 900).generate(other))
    ax2.set_title(name)
    ax2.set_axis_off()
    plt.show()
    
def vectorize_tfdif(texts):
    """Uses CountVectorizer and TFIDFTRansformation
    Input:
        texts: iterable of texts
    Output:
        X: tfidf transformed feature matrix
        words: dictionary with words as keys, column indices as values
    """
    #vectorize
    vec = feature_extraction.text.CountVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    words = vec.vocabulary_
    #tfidf
    tfidf = feature_extraction.text.TfidfTransformer()
    X = tfidf.fit_transform(X)
    return(X,words)
    
