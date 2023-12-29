import numpy as np 
import nltk as nlk 
from nltk.stem import PorterStemmer
import json
pt=PorterStemmer()

nlk.download('stopwords')

st_words=nlk.corpus.stopwords.words('english')

train_data=[]
test_data=[]

with open('Music_Review_train.json','r') as json_file:
    for line in json_file:
        train_data.extend([json.loads(line)])
with open('Music_Review_train.json','r') as json_file:
    for line in json_file:
        test_data.extend([json.loads(line)])


# classifying the positive and negative review for the test case file
test_pos_review=[]
test_neg_review=[]

for review in test_data:
    review_text = review['reviewText']
    overall_rating = review['overall']

    if overall_rating >= 2.0:
        test_pos_review.append(review_text)
    else:
        test_neg_review.append(review_text)


# classifying the positive and negative review for train case file
train_pos_review=[]
train_neg_review=[]

for review in train_data:
    review_text = review['reviewText']
    overall_rating = review['overall']

    if overall_rating >= 2.0:
        train_pos_review.append(review_text)
    else:
        train_neg_review.append(review_text)


# making the n grams

#unigrams

wordfreq={}
poswordfreq={}
negwordfreq={}



#bigrams

biwordfreq = {}
biposwordfreq = {}
binegwordfreq = {}



#trigrams 
triwordfreq = {}
triposwordfreq = {}
trinegwordfreq = {}

 
for i in range(len(train_pos_review)):
    z=[pt.stem(y) for y in train_pos_review[i].split() if y not in st_words]
    for i in range (len(z)-2):
        ## for unigrams
        if z[i] in poswordfreq:
            poswordfreq[z[i]]+=1
        else:
            poswordfreq[z[i]]=1
        if z[i+2] in poswordfreq:
            poswordfreq[z[i+2]]+=1
        else:
            poswordfreq[z[i+2]]=1
        ## for keeping a track of total unique words and their 
        if z[i] not in wordfreq:
            wordfreq[z[i]]=1
        if z[i+2] not in wordfreq:
            wordfreq[z[i+2]]=1
        ## for bigrams
        bi=z[i]+' '+z[i+1]
        if bi in biposwordfreq:
            biposwordfreq[bi]+=1
        else:
            biposwordfreq[bi]=1
        if bi not in biwordfreq:
            biwordfreq[bi]=1
        
        ## for trigrams
        tri=z[i]+' '+z[i+1]+' '+z[i+2]
        if tri in triposwordfreq:
            triposwordfreq[tri]+=1
        else:
            triposwordfreq[tri]=1
        if tri not in triwordfreq:
            triwordfreq[tri]=1
    ## last bigram word
    lsword=z[len(z)-2]+' '+z[len(z)-1]
    if lsword in biposwordfreq:
        biposwordfreq[lsword]+=1
    else:
        biposwordfreq[lsword]=1
    if lsword not in biwordfreq:
            biwordfreq[lsword]=1
 
for i in range(len(train_neg_review)):
    z=[pt.stem(y) for y in train_neg_review[i].split() if y not in st_words]
    for i in range (len(z)-2):
        ## for unigrams
        if z[i] in negwordfreq:
            negwordfreq[z[i]]+=1
        else:
            negwordfreq[z[i]]=1
        if z[i+2] in negwordfreq:
            negwordfreq[z[i+2]]+=1
        else:
            negwordfreq[z[i+2]]=1
        ## for keeping a track of total unique words and their 
        if z[i] not in wordfreq:
            wordfreq[z[i]]=1
        if z[i+2] not in wordfreq:
            wordfreq[z[i+2]]=1
        ## for bigrams
        bi=z[i]+' '+z[i+1]
        if bi in binegwordfreq:
            binegwordfreq[bi]+=1
        else:
            binegwordfreq[bi]=1
        if bi not in biwordfreq:
            biwordfreq[bi]=1
        
        ## for trigrams
        tri=z[i]+' '+z[i+1]+' '+z[i+2]
        if tri in trinegwordfreq:
            trinegwordfreq[tri]+=1
        else:
            trinegwordfreq[tri]=1
        if tri not in triwordfreq:
            triwordfreq[tri]=1
    ## last bigram word
    lsword=z[len(z)-2]+' '+z[len(z)-1]
    if lsword in biposwordfreq:
        biposwordfreq[lsword]+=1
    else:
        biposwordfreq[lsword]=1
    if lsword not in biwordfreq:
            biwordfreq[lsword]=1   
## calculation of probabilities
        
u_prob_word={}
u_neg_word={}
bi_prob_word={}
bi_neg_word={}
tri_prob_word={}
tri_neg_word={}

## using laplace smoothing
## the formula becomes
## 
# Pˆ(wi | c) =count(wi, c)+1/ sum((count(w, c)+1)+1)
toal_pos_u = sum(list(poswordfreq.values()))
toal_neg_u = sum(list(negwordfreq.values()))

toal_pos_bi = sum(list(biposwordfreq.values()))
toal_neg_bi = sum(list(binegwordfreq.values()))

toal_pos_tri = sum(list(triposwordfreq.values()))
toal_neg_tri = sum(list(trinegwordfreq.values()))



for word in wordfreq.keys():
    if word in poswordfreq:
        u_prob_word[word]=poswordfreq[word]/(toal_pos_u+len(wordfreq))
    else:
        u_prob_word[word]=1/(toal_pos_u+len(wordfreq))
    
    if word in negwordfreq:
        u_neg_word[word]=negwordfreq[word]/(toal_neg_u+len(wordfreq))
    else:
        u_neg_word[word]=1/(toal_neg_u+len(wordfreq))
    
for word in biwordfreq.keys():
    if word in biposwordfreq:
        bi_prob_word[word]=biposwordfreq[word]/(toal_pos_bi+len(biwordfreq))
    else:
        bi_prob_word[word]=1/(toal_pos_bi+len(wordfreq))
    
    if word in binegwordfreq:
        bi_neg_word[word]=binegwordfreq[word]/(toal_neg_bi+len(biwordfreq))
    else:
        bi_neg_word[word]=1/(toal_neg_bi+len(wordfreq))
    
for word in triwordfreq.keys():
    if word in triposwordfreq:
        tri_prob_word[word]=triposwordfreq[word]/(toal_pos_tri+len(triwordfreq))
    else:
        tri_prob_word[word]=1/(toal_pos_tri+len(wordfreq))
    
    if word in trinegwordfreq:
        tri_neg_word[word]=trinegwordfreq[word]/(toal_neg_tri+len(triwordfreq))
    else:
        tri_neg_word[word]=1/(toal_neg_tri+len(wordfreq))
    
phi=len(train_pos_review)/(len(train_pos_review)+len(train_neg_review))


### finally testing out!!!!!

## scores.....


bi_corr_cnt=0
bi_incorr_cnt=0
tri_corr_cnt=0
tri_incorr_cnt=0


 
for i in range(len(train_pos_review)):
    z=[pt.stem(y) for y in train_pos_review[i].split() if y not in st_words]
    
    ## the probabilities for a document to fall in positive or negative review
    
    bi_pos_prob=0
    bi_neg_prob=0
    
    tri_pos_prob=0
    tri_neg_prob=0
    
    for i in range (len(z)-2):
        ## for bigrams and trigrams the individual probablity of word is required according to naive bais
        if z[i] in u_prob_word:
            bi_pos_prob+=np.log(u_prob_word[z[i]])
            tri_pos_prob+=np.log(u_prob_word[z[i]])
        
        if z[i] in u_neg_word:
            bi_neg_prob+=np.log(u_neg_word[z[i]])
            tri_neg_prob+=np.log(u_neg_word[z[i]])
        
        if z[i+2] in u_prob_word:
            bi_pos_prob+=np.log(u_prob_word[z[i+2]])
            tri_pos_prob+=np.log(u_prob_word[z[i+2]])
        
        if z[i+2] in u_neg_word:
            bi_neg_prob+=np.log(u_neg_word[z[i+2]])
            tri_neg_prob+=np.log(u_neg_word[z[i+2]])
        
        ## for bigrams
        bi=z[i]+' '+z[i+1]
        if bi in bi_prob_word:
            bi_pos_prob+=np.log(bi_prob_word[bi])
        if bi in bi_neg_word:
            bi_neg_prob+=np.log(bi_neg_word[bi])
        
        ## for trigrams
        tri=z[i]+' '+z[i+1]+' '+z[i+2]
        if tri in tri_prob_word:
            tri_pos_prob+=np.log(tri_prob_word[tri])
        if tri in tri_neg_word:
            tri_neg_prob+=np.log(tri_neg_word[tri])
        
    ## last bigram word
    lsword=z[len(z)-2]+' '+z[len(z)-1]
    if lsword in bi_prob_word:
        bi_pos_prob+=np.log(bi_prob_word[bi])
    if lsword in bi_neg_word:
        bi_neg_prob+=np.log(bi_neg_word[bi])
    
    bi_pos_prob+=np.log(phi)
    bi_neg_prob+=np.log(1-phi)
 
    tri_pos_prob+=np.log(phi)
    tri_neg_prob+=np.log(1-phi)

    if(bi_pos_prob>bi_neg_prob):
        bi_corr_cnt+=1
    else:
        bi_incorr_cnt+=1
    
    if(tri_pos_prob>tri_neg_prob):
        tri_corr_cnt+=1
    else:
        tri_incorr_cnt+=1
 
for i in range(len(train_neg_review)):
    z=[pt.stem(y) for y in train_neg_review[i].split() if y not in st_words]
    
    ## the probabilities for a document to fall in positive or negative review
    
    bi_pos_prob=0
    bi_neg_prob=0
    
    tri_pos_prob=0
    tri_neg_prob=0
    
    for i in range (len(z)-2):
        ## for bigrams and trigrams the individual probablity of word is required according to naive bais
        if z[i] in u_prob_word:
            bi_pos_prob+=np.log(u_prob_word[z[i]])
            tri_pos_prob+=np.log(u_prob_word[z[i]])
        
        if z[i] in u_neg_word:
            bi_neg_prob+=np.log(u_neg_word[z[i]])
            tri_neg_prob+=np.log(u_neg_word[z[i]])
        
        if z[i+2] in u_prob_word:
            bi_pos_prob+=np.log(u_prob_word[z[i+2]])
            tri_pos_prob+=np.log(u_prob_word[z[i+2]])
        
        if z[i+2] in u_neg_word:
            bi_neg_prob+=np.log(u_neg_word[z[i+2]])
            tri_neg_prob+=np.log(u_neg_word[z[i+2]])
        
        ## for bigrams
        bi=z[i]+' '+z[i+1]
        if bi in bi_prob_word:
            bi_pos_prob+=np.log(bi_prob_word[bi])
        if bi in bi_neg_word:
            bi_neg_prob+=np.log(bi_neg_word[bi])
        
        ## for trigrams
        tri=z[i]+' '+z[i+1]+' '+z[i+2]
        if tri in tri_prob_word:
            tri_pos_prob+=np.log(tri_prob_word[tri])
        if tri in tri_neg_word:
            tri_neg_prob+=np.log(tri_neg_word[tri])
        
    ## last bigram word
    lsword=z[len(z)-2]+' '+z[len(z)-1]
    if lsword in bi_prob_word:
        bi_pos_prob+=np.log(bi_prob_word[bi])
    if lsword in bi_neg_word:
        bi_neg_prob+=np.log(bi_neg_word[bi])
    
    bi_pos_prob+=np.log(phi)
    bi_neg_prob+=np.log(1-phi)
 
    tri_pos_prob+=np.log(phi)
    tri_neg_prob+=np.log(1-phi)

    if(bi_pos_prob>bi_neg_prob):
        bi_corr_cnt+=1
    else:
        bi_incorr_cnt+=1
    
    if(tri_pos_prob>tri_neg_prob):
        tri_corr_cnt+=1
    else:
        tri_incorr_cnt+=1


biaccuracy = (100*bi_corr_cnt)/(bi_corr_cnt+bi_incorr_cnt)
triaccuracy = (100*tri_corr_cnt)/(tri_corr_cnt+tri_incorr_cnt)

print("Testing Accuracy along with unigrams and bigrams is",biaccuracy,"%")
print("Testing Accuracy along with unigrams and trigrams(additonal feature) is",triaccuracy,"%")
