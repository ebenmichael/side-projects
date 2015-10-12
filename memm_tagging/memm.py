# -*- coding: utf-8 -*-
"""
Module for Maximum Entropy Markov Model Classification


"""

from sklearn import linear_model,preprocessing
import numpy as np
from collections import defaultdict
import time

class MEMM:
    """MEMM class, fits a maximum entropy (log-linear) model on features"""
    def __init__(self,penalty='l2', dual=False, tol=0.0001, C=1.0, 
                 fit_intercept=True, intercept_scaling=1, class_weight=None, 
                 random_state=None, solver='lbfgs', max_iter=100, 
                 multi_class='multinomial', verbose=0):
        """Initializes parameters for sklearn.linear_model.LogisticRegression"""              
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C 
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling 
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter 
        self.multi_class = multi_class
        self.verbose = verbose
        #initiliaze logistic regression
        self.logit = linear_model.LogisticRegression(penalty,dual,tol,C,
                                                fit_intercept,intercept_scaling,
                                                class_weight, random_state,
                                                solver, max_iter, multi_class,
                                                verbose)
    def getPrevs(self,y, firstTags = ["*","*"]):
        """Returns an array corresponding to the previous two tags for every
        element of a sequence X with tags y
        Parameters
        ----------     
        y:  array-like, shape(seq_len,), where seq_len is the length of the
            sequence.
            Tag sequence.
        
        firstTags:  array-like, shape (2,).
                    Tags for the two elements preceeding the start of the 
                    sequence. Defaults to padding start symbols "*", but can 
                    be anything if the sequence is a subsequence of a larger
                    sequence.
        Returns
        -------
        yPrev:      array-like, shape (n)
        """
        #length of sequence
        seq_len = len(y)
        #initialize yPrev with dtype of y
        yPrev = np.zeros([seq_len,2],dtype = y.dtype)
        #put first tags as first values of yPrev
        yPrev[0,] = firstTags
        yPrev[1,] = [firstTags[1],y[0]]
        #index over tags and collect previous two tags for each element
        for i in range(seq_len):
            yPrev[i,] = [y[i-1], y[i-2]]
            
        #return the previous tags
        return(yPrev)
        
    def fit(self,X,y, firstTags = None):
        """Fits a log-linear model with all the feature vectors for each
        sequence as an array, X, of (possible sparse) matrices, and an array
        of tag sequences as vectors y
        
        Parameters
        ----------
        X:  array-like, shape (n_sequences), where n_sequences is the number
            of sequences. 
            Contains the sequences as {array-like, sparse martix}
            shape (seq_len,n_features), where seq_len is the length of the 
            sequence and n_features is the number of features.

            
        y:  array-like, shape(n_sequences,).
            Contains the tag sequences for each sequence
        
        firstTags:  array-like, shape (n_sequences,2).
                    Contains the tags for the two elements preceeding the 
                    start of each sequence. Defaults to None for the case where
                    there are no preceeding elements and the start padding
                    symbol "*" is used. For sequences which are subsequences 
                    of larger sequences, input the preceeding tags for 
                    firstTags
        """
        ###Get previous tags for each sequence
        yPrev = []
        first = firstTags is not None
        for i in range(len(X)):
            if first:
                yPrev.append(self.getPrevs(y[i],firstTags = firstTags[i]))
            else:
                yPrev.append(self.getPrevs(y[i]))
        ###collapse all sequences and tag sequences
        X = np.concatenate(X)
        y = np.concatenate(y)
        yPrev = np.concatenate(yPrev)
        
        
        ###encode labels into one-hot-encoding
        #get labels plus start symbol "*"
        prevLabs = np.unique(np.append(yPrev,["*"]))
        tags = np.unique(y)
        #encode labels into integer labels for both previous tags and tags
        #the difference between lePrev and leTag is that lePrev includes
        #the start tag
        lePrev = preprocessing.LabelEncoder()
        lePrev.fit(prevLabs)
        yPrev = lePrev.transform(yPrev)
        
        leTag = preprocessing.LabelEncoder()
        leTag.fit(tags)
        y = leTag.transform(y)
        #encode labels with one hot encoding
        enc = preprocessing.OneHotEncoder(n_values = len(prevLabs))
        yPrev = enc.fit_transform(yPrev).toarray()
        #keep LabelEncoder and OneHotEncoder to encode future labels
        self.lePrev = lePrev
        self.leTag = leTag
        self.enc = enc
        
        #add one hot encoding of previous tags as final columns of X
        X = np.append(X,yPrev,1)
        ###fit log linear model
        self.logit.fit(X,y)
    

    def viterbi(self,X,tags,yPrev = ["*","*"]):
        
        """Uses the Viterbi algorithm to tag a sequence X using the trained
        log-linear model on history features
        
        Parameters
        ----------
        X:  {array-like,sparse matrix} shape (seq_len,n_features)
            where seq_len is the length of the sequence and n_features is the 
            number of features. The sequence to tag
            
        yPrev: array-like shape(2,). The previous two tags before the sequence
               starts. Default is padding ["*","*"]
               
        tags: tagset of possible tags
        
        Returns
        -------
        y:  array shape (seq_len,).
            Most likely tag sequence
        """
        #convert X to a numpy array if it isn't already
        if not type(X).__module__ == "numpy":
            X = np.array(X)
            
        n = X.shape[0] #length of sequence
        #initialize dynamic programming table pi[k][(u,v)] which represents
        #the maximal probability for a sequence ending at k with tags u and v
        pi = [defaultdict(float)]
        #initialize back pointer array bp[k][(u,v)]
        bp = [defaultdict(float)]
        
        #initialize first value of pi
        pi[0][tuple(yPrev)] = 1
        
        #iterate over elements of the sequence
        for k in range(1,n+1):
            #add a dictionary to pi and bp
            pi.append(defaultdict(float))
            bp.append(defaultdict(float))
            #check if k is 1 or 2 to change possible tags to the starting tags
            if k == 1:
                tagu = [yPrev[1]]
                tagw = [yPrev[0]]
            elif k == 2:
                tagu = tags
                tagw = [yPrev[1]]
            else:
                tagu = tags
                tagw = tags
            #get kth element of X as features
            elt = X[k-1,]
            #index over possible tags at k- and k
            
            for u in tagu:
                
                for v in tags:
                    #get integer label of v
                    
                    vInt = self.leTag.transform([v])[0]
                    #maximize over possible tags at k-2
                    currMax = 0
                    argMax = ''
                    
                    for w in tagw:
                        
                        #get one hot encoding of [w,u] and add to end of elt
                        prevTags = self.enc.transform(self.lePrev.transform([w,u]))
                        prevTags = prevTags.toarray()
                        eltWithPrev = np.append(elt,prevTags)
                        #get probability estimate from log linear model
                        
                        p = self.logit.predict_proba(eltWithPrev)[0][vInt]
                        
                                               
                        #multiply by pi value 
                        prob = pi[k-1][(w,u)] * p
                        #print(prob)
                        
                        if prob > currMax:
                            currMax = prob
                            
                            argMax = w
                     
                     
                    #assign max and argmax to pi and bp
                    pi[k][(u,v)] = currMax
                    bp[k][(u,v)] = argMax   
        #set final tags u,v as those which maximize pi[n][(u,v)]
        currMax = 0
        uMax = ''
        vMax = ''
        #if the sequence is of length 1, restrict the set of tags
        if n == 1:
            tagu = ['*']
        else:
            tagu = tags
        #iterate over possible tags
        for u in tagu:
            for v in tags:
                prob = pi[n][(u,v)]
                #print(u,v,prob)
                if prob > currMax:
                    currMax = prob
                    uMax= u
                    vMax = v
        #place last two tags into most likely tag sequence y
        y = ['']*n
        #if only a one word sentence, only change y[0]
        if n == 1:
            y[0] = vMax
        else:
            
            y[n-2] = uMax
            y[n-1] = vMax
        #work through backpointers to get the tag sequence
        for k in reversed(range(1,n-1)):
            y[k-1] = bp[k+2][(y[k],y[k+1])]
        
        #return the max probability tag sequence
        return(y,pi)
     

                   
    def predict(self,X,tags,firstTags = None):
        """Given a list of sequences X, predict the tag sequences
        Parameters
        ----------
        X:  array-like, shape (n_sequences), where n_sequences is the number
            of sequences. 
            Contains the sequences as {array-like, sparse martix}
            shape (seq_len,n_features), where seq_len is the length of the 
            sequence and n_features is the number of features.
             
             tags: tagset of possible tags
    
        firstTags:  array-like, shape (n_sequences,2).
                    Contains the tags for the two elements preceeding the 
                    start of each sequence. Defaults to None for the case where
                    there are no preceeding elements and the start padding
                   symbol "*" is used. For sequences which are subsequences 
                   of larger sequences, input the preceeding tags for 
                   firstTags
            
            Returns
            -------
            pred:   array, shape (n_sequences).
                Contains the predicted tag sequences for the sequences in X
        """
        
        pred = [''] * len(X)
        first = firstTags is not None
        #index over sequences and tag them
        for i in range(len(X)):
            if first:
                pred[i] = self.vitberi(X[i],tags,firstTags[i])
            else:
                pred[i] = self.viterbi(X[i],tags)
        return(pred)
    