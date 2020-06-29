import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def triplet_loss(ground,predict,decoy,margin=0.2):
  measure = torch.nn.PairwiseDistance(p=2)
  distance = measure(ground,predict) - measure(ground,decoy) + margin
  loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
  return loss

def MAP_k(predict,ground,topk=20):
  nbrs = NearestNeighbors(n_neighbors=topk, algorithm='brute',metric='euclidean').fit(ground)
  distances, indices = nbrs.kneighbors(predict)
  list_rank = []
  score = 0
  ground_index = [i for i in range(len(ground))]
  for count, item in enumerate(ground_index):
    try:
      rank = list(np.where(indices[count] == item)[0])[0]
      score += 1/(1+rank)
      list_rank.append(rank)
    except:
      score = score
      list_rank.append(-1)
  return score, list_rank

def embed_text_image(model,loader_object,output_dim=100):
  array_image = np.zeros((loader_object.length,output_dim))
  array_text = np.zeros((loader_object.length,output_dim))
  model.eval()
  index = 0
  batch_size = loader_object.batch_size
  with torch.no_grad():
    for batch in loader_object:
      batch = (item.to(device) for item in batch)
      input_text, input_img, input_decoy = batch
      image_output = model.get_embedding(input_img)
      text_output = model.get_embedding(input_text,is_image=False)

      array_image[index:index + batch_size] = image_output.detach().cpu().numpy()
      array_text[index:index + batch_size] = text_output.detach().cpu().numpy()
      index += batch_size
  
  return array_image, array_text