import pandas as pd
import numpy as np
from PIL import Image

import nltk
import re
from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from matplotlib import pylab as plt

lemma = nltk.wordnet.WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def split_stem_stop(df1,stopword, tag=False):
    df = df1.copy(deep=True)
    if tag:
        df['all_sentences'] = df[0] +' '+df[1] +' '+df[2] +' '+df[3] +' '+df[4] +' '+df[5] +' '+df[6] +' '+df[7] +' '+df[8] +' '+df[9] +' '+df[10] +' '+df[11] +' '+df[12]
    else:
        df['all_sentences'] = df['0'] +' '+df['1'] +' '+df['2'] +' '+df['3'] +' '+df['4']
    df['list_sentence'] = df['all_sentences'].str.split(' ')
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [y.lower() for y in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [re.sub(r'[^\w\s]','',i) for i in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [lemma.lemmatize(y) for y in x if y not in stopword])
    #print('test')
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [stemmer.stem(y) for y in x])
    return df

def display_images(description_pd,top_20_index,index,test=False):
    fig, ax = plt.subplots(4,5,figsize=(15,10))
    num = -1
    fig.suptitle(description_pd['0'][index] +'\n' + description_pd['1'][index] + 
        '\n' + description_pd['2'][index] +'\n' + description_pd['3'][index]
        +'\n' + description_pd['4'][index])
    for k in range(4):
        for m in range(5):
            if num == -1:
                image = index
            else:
                image = top_20_index[index][num]
            im0 = Image.open('../Final/data_final/images_train/' + str(image) + '.jpg').resize((224,224))
            pix = im0.load()
            arr_RGB = np.zeros((im0.size[1],im0.size[0],3))
            for i in range(im0.size[0]):
                for j in range(im0.size[1]):
                    arr_RGB[j,i,0] = pix[i,j][0]
                    arr_RGB[j,i,1] = pix[i,j][1]
                    arr_RGB[j,i,2] = pix[i,j][2]
            RGB_norm = arr_RGB/255
            ax[k,m].imshow(RGB_norm[:,:,:])
            num += 1

def MAP20_score(top_20_indices,ground_truth):
    list_pos = []
    score = 0
    for count,i in enumerate(ground_truth):
        try:
            pos = list(np.where(top_20_indices[count] == i)[0])[0]
            score += 1/(1+pos)
            list_pos.append(pos)
        except:
            score = score
            list_pos.append(-1)
    return score,list_pos

def evaluate_cv(data_X,data_Y,cv_fold = 5,alpha=0.001,random_seed = 0):
    
    np.random.seed(random_seed)
    group = np.random.randint(cv_fold, size=data_X.shape[0],)
    
    list_score = []
    
    for i in range(cv_fold):
        X_train = data_X[np.where(group != i)]
        X_hold = data_X[np.where(group == i)]
        
        Y_train = data_Y[np.where(group != i)]
        Y_hold = data_Y[np.where(group == i)]
        
        regrtest = Ridge(alpha=alpha)
        regrtest.fit(X_train,Y_train)
        
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='brute',metric='euclidean').fit(Y_hold)
        distances, indices = nbrs.kneighbors(regrtest.predict(X_hold))
        score, list_pos = MAP20_score(indices,range(int(data_X.shape[0]/cv_fold)))
        list_score.append(score/int(data_X.shape[0]/cv_fold))
    return list_score

def convert_TFIDF_to_w2v(mapping_array,TFIDF_object,):
    fasttext2TFIDF = np.zeros((10000,300))
    for i in range(10000):
        first_time = True
        for index in np.argsort(TFIDF_object[i].toarray()[0])[::-1][:15]:
            weight = TFIDF_object[i].toarray()[0][index]
        
            if first_time:
                fasttext2TFIDF[i] = mapping_array[index]*weight
                first_time = False
            else:
                fasttext2TFIDF[i] = fasttext2TFIDF[i] + mapping_array[index]*weight
    return fasttext2TFIDF