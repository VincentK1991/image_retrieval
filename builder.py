import torch
import numpy as np
import glob
from PIL import Image
import timeit
from sklearn.feature_extraction import text
import nltk
import re
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stemmer = nltk.stem.snowball.SnowballStemmer('english')
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json

def stem_stop(df1):
    df = df1.copy(deep=True)
    df['list_sentence'] = df['text'].str.split(' ')
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [y.lower() for y in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [re.sub(r'[^\w\s]','',i) for i in x])
    df['list_sentence'] = df['list_sentence'].apply(lambda x: [stemmer.stem(y) for y in x if y not in stop_words])
    df['stemmed_text'] = df['list_sentence'].str.join(' ')
    return df

def tokenize_frame(frame1):
    frame = frame1.copy(deep=True)
    frame['tokens'] = frame['stemmed_text'].apply(lambda x: word_tokenize(x))
    return frame

def create_vocab_dict(vocab):
    dict_str2int = {}
    dict_int2str = {}
    for i,j in enumerate(vocab,1):
        dict_str2int[j] = i
        dict_int2str[i] = j
    return dict_str2int, dict_int2str

def apply_dict(list1,dict1):
    result = []
    for i in list1:
        result.append(dict1[i])
    return result

def word2token(frame1,dict1):
    frame = frame1.copy(deep=True)
    frame['numeric_tokens'] = frame['tokens'].apply(lambda x: apply_dict(x,dict1))
    return frame

def pad_truncate_array(frame,max_len=400,truncate='Post'):
    result = []
    for item in frame['numeric_tokens']:
        if len(item) > max_len:
            if truncate == 'Post':
                result.append(item[:400])
            elif truncate == 'Pre':
                result.append(item[len(item) - max_len:])
            else:
                print('something wrong')
        else:
            result.append(item + [0]*(max_len - len(item)))
    return np.array(result)

def main():
	"""  """
	all_image = glob.glob('Data/images_train/*.jpg')
	all_text = glob.glob('Data/descriptions_train/*.txt')
	sorted(all_image)
	sorted(all_text)
	text_train, text_test, image_train, image_test = train_test_split(all_text, all_image, test_size=0.1, random_state=42)

  list_text_train = []
  for item in text_train:
    list_text_train.append(open(item,"r").read().split('\n'))
  
  list_text_val = []
  for item in text_test:
    list_text_val.append(open(item,"r").read().split('\n'))
  
  temp_train =[]
    temp_train.append([''.join(i) for i in list_text_train])
  
  temp_val = []
    temp_val.append([''.join(i) for i in list_text_val])

  train1 = pd.DataFrame({'text':temp_train[0]})
  val1 = pd.DataFrame({'text':temp_val[0]})
  train1 = stem_stop(train1)
  val1 = stem_stop(val1)

  train1 = tokenize_frame(train1)
  val1 = tokenize_frame(val1)

  train_val_tokens =  pd.concat([train1,val1],ignore_index=True)

  vocab_list = []
  for i in train_val_tokens['tokens']:
    for j in list(np.unique(i)):
        temp_list.append(j)
  vocab_list = list(np.unique(temp_list)) # get unique vocab

  dict_str2int, dict_int2str = create_vocab_dict(vocab_list)

  dict_str2int['PAD'] = 0
  dict_int2str[0] = 'PAD'

  with open('dict_str2idx.json', 'w') as fp:
    json.dump(dict_str2int, fp)

  with open('dict_idx2str.json', 'w') as fp:
    json.dump(dict_int2str, fp)


  train_tokenized = word2token(train1 ,dict_str2int)
  val_tokenized = word2token(val1, dict_str2int)

  array_train_tokens = pad_truncate_array(train_tokenized,max_len=50,truncate='Pre')
  array_val_tokens = pad_truncate_array(val_tokenized,max_len=50,truncate='Pre')

  train_text_tensor = torch.tensor(array_train_tokens,dtype=torch.long)
  val_text_tensor = torch.tensor(array_val_tokens,dtype=torch.long)

  torch.save(train_text_tensor,'train_text_tensor.pt')
  torch.save(val_text_tensor,'val_text_tensor.pt')

	preprocess = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),])

	train_image_tensor = torch.zeros(len(image_train),3,224,224)
	test_image_tensor = torch.zeros(len(image_test),3,224,224)


	start = timeit.default_timer()
	for count,item in enumerate(image_train):
		temp = Image.open(item).convert('RGB').resize((224,224))
		train_image_tensor[count] = preprocess(temp)
  		
  		if count %1000 == 0:
    		stop = timeit.default_timer()
    		print('image pre-processing iterations {} took {} sec'.format(count,stop- start))
    		start = timeit.default_timer()

    torch.save(train_image_tensor,'train_image_tensor.pt')
    print('finish pre-processing training images')

	start = timeit.default_timer()
	for count,item in enumerate(image_test):
		temp = Image.open(item).convert('RGB').resize((224,224))
		test_image_tensor[count] = preprocess(temp)
  		
  		if count %1000 == 0:
    		stop = timeit.default_timer()
    		print('image pre-processing iterations {} took {} sec'.format(count,stop- start))
    		start = timeit.default_timer()

    torch.save(test_image_tensor,'validation_image_tensor.pt')
    print('finish pre-processing validation images')

if __name__=="__main__":
	main()




















