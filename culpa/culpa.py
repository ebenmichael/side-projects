# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:45:35 2015

@author: eli
"""

#First, interacting with RESTful API

import json
import requests
from lxml import html
from bs4 import BeautifulSoup
import pandas as pd
from sklearn import feature_extraction
from sklearn import preprocessing

def get_dept_queries():
    """Gets the list of departments to query the CULPA API with"""
    
    #get all departments
    soup = BeautifulSoup(requests.get("http://culpa.info").text)
    dept_tree = soup.find('div',attrs = {'class':'box department-list'})
    #go through this and extract the queries to use below
    #full for loop
    queries = []
    for link in dept_tree.find_all("a"): #finds all links
        #isolate link
        dept = link.get("href")
        #turn into a query for API
        #if its a department
        if "departments" in dept:
            dept_id = dept[len("/departments/"):] #get dept id
            #ignore CORE
            if not dept_id.isdigit():
                continue
            query =  "/department_id/" + dept_id
            queries.append(query)
        #if its a class (for Core classes)
        if "courses" in dept:
            course_id = dept[len("/courses/"):]
            query = "/course_id/" + course_id
            queries.append(query)
    return(queries)


def get_profs(queries):
    """Perform a series of queries to CULPA API to get professor data"""
    #connect with CULPA API
    prof_url = "http://api.culpa.info/professors" 
    #get a list of dictionaries with professor information
    prof_list= []
    #keep track of ids already seen with a set
    id_set = set()
    for query in queries:
            
        js = requests.get(prof_url + query)
        #ensure that we're getting a json file (i.e. no html)
        if not js.headers["content-type"] == "application/json":
            continue   
        #get json object
        js = js.json()
        #make sure it was a success
        if js["status"] == "success":
            #get all the professor objects in the department/that teach the class
            profs = js["professors"]
            
            for prof in profs:
                #make sure that we haven't included this professor already
                if not prof["id"] in id_set:
                    prof_list.append(prof)
    return(prof_list)

    
    
def extract_review(prof_id):
    """Extracts review and workload text for prof with prof_id"""
    #connect with api and get reviews
    rev_url = "http://api.culpa.info/reviews/professor_id/"
    revs = requests.get(rev_url + str(prof_id))
    #ensure that we got a json object in string form
    if not revs.headers["content-type"] == "application/json":
        return
    #get json object
    revs = revs.json()
    #make sure that we got a success
    if not revs["status"] == "success":
        return
    #get the reviews
    reviews = revs["reviews"]
    #use exception handling to handle empty reviews
    try:
        #make dictionary with id -> revieq text and workload text
        ids = {review["id"]:[review["review_text"].lower(),review["workload_text"].lower()] 
                for review in reviews}
        return(ids)
    except KeyError:
        return

def combine_review(ids):
    """Combines reviews from ids
    Input:
        ids: a dict with review ids as keys and [review_text,workload_text] 
                as values 
    """
    #use a string join over iterable
    comb_review = "\n".join(review[0] for review in ids.values())
    comb_workload = "\n".join(review[1] for review in ids.values())
    
    return(comb_review,comb_workload)
    
def extract_combine(prof_id):
    """Extracts and combines the review and workload text for prof with prof_id"""
    #get reviews with ids
    ids = extract_review(prof_id)
    if ids is not None:
        #if something was returned, combine the reviews
        return(combine_review(ids))
    else:
        return
        
def get_prof_reviews(profs):
    """Go through each professor, extract their reivews, and combine them
    Input:
        profs: list of dictionaries
    """
    new_profs = []
    for i,prof in enumerate(profs):
        #get id
        prof_id = prof["id"]
        print(i,prof_id)
        #get reviews
        out = extract_combine(prof_id)
        #if no output, then we didn't get review text for some reason
        #don't include that professor
        if out is None:
            continue
        review,work = out
        #assign the review and workload text to the dictionary
        prof["review_text"] = review
        prof["work_text"] = work
        new_profs.append(prof)
    return(new_profs)


def num_agree(review_id):
    """Finds the number of Agree,Disagree,Funny for review_id"""
    
    url = "http://culpa.info/reviews/"
    
    soup = BeautifulSoup(requests.get(url + str(review_id)).text,"lxml")

    #find the inputs with the info we want    
    
    agree_input = soup.find("input",attrs = {"class":"agree"})
    #parse num_agree out of Agree (num_agree)
    agree_str = agree_input['value']
    num_agree = int(agree_str[agree_str.find("(")+1:agree_str.find(")")])

    disagree_input = soup.find("input",attrs = {"class":"disagree"})

    disagree_str = disagree_input['value']
    num_disagree = int(disagree_str[disagree_str.find("(")+1:disagree_str.find(")")])    

    funny_input = soup.find("input",attrs = {"class":"funny"})

    funny_str = funny_input['value']
    num_funny = int(funny_str[funny_str.find("(")+1:funny_str.find(")")]) 
    
    return(num_agree,num_disagree,num_funny)


def to_data_frame(profs):
    """Converts to dataframe"""
    
    return(pd.DataFrame(profs))
    
def to_term_doc_mat(dat,col, tfidf = True, max_df = 1.0, min_df = 5):
    """Gets a (tfidf) bag of words feature representation for dataframe
    Input:
        dat:
            DataFrame which has text and professor name
            col: Column of DataFrame with text
            tfidf: Boolean. Whether to do tfidf transformation
    """
    #instantiate a CountVectorizer object
    count_vec = feature_extraction.text.CountVectorizer(stop_words="english",
                                                        min_df=min_df,
                                                        max_df = max_df)
    #fit on review text nd transform to bag of words
    X = count_vec.fit_transform(dat[col])
    if tfidf:
        #do tfidf
        tfidf = feature_extraction.text.TfidfTransformer()
        X = tfidf.fit_transform(X)
    return(X,count_vec.vocabulary_)

def encode_classes(target):
    """Encodes classes, goes from strings or whatver to numbers
    Input:
        target: array, pandas Series, etc. of classes
    Output:
        target: array of ecnoded classes
    """
    #instantiate LAbelEncoder and fit and transform
    le = preprocessing.label.LabelEncoder()
    le.fit(target)
    target = le.transform(target)
    return(target)
