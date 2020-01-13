import pandas as pd
import numpy as np
from PIL import Image


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

import nltk
import re
lemma = nltk.wordnet.WordNetLemmatizer()
from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
print(len(stop_words))
def split_stem_stop(df1,stopword, tag=False):
    df = df1.copy(deep=True)
    if not tag:
        df['all_sentences'] = df['0'] +' '+df['1'] +' '+df['2'] +' '+df['3'] +' '+df['4']
    df['list_sentence'] = df['all_sentences'].str.split(' ')
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [y.lower() for y in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [re.sub(r'[^\w\s]','',i) for i in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [lemma.lemmatize(y) for y in x if y not in stopword])
    #df['stemmed'] = df['list_sentence'].apply(' '.join)
    for i in range(df.shape[0]):
        df['list_sentence'][i] = df['list_sentence'][i]
    return df