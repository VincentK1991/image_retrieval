

import torch
import numpy as np
import torch.nn as nn
import torchvision
from sklearn.neighbors import NearestNeighbors
import timeit
import json, argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from helper import triplet_loss, MAP_k, embed_text_image
import loader
import model

def main(args):
	""" 
	1. load model
	2. load data
	3. optimizer
	4. train batch
	5. eval batch 
	"""
	image_encoder_model = model.ImageEncoder(output_image_size=args.embedding_dim)
	text_encoder_model = model.TextEncoder(input_dim=50,hidden_dim=args.hidden_dim,vocab_size = args.vocab_size,
											embedding_dim=args.word_embedding_dim,output_dim=args.embedding_dim,dropout=args.dropout)
	triple_encoder_model = model.TripleEncoder(image_encoder_model,text_encoder_model)

	state_dict_triple_encoder = torch.load(args.model_weight)
	triple_encoder.load_state_dict(state_dict_triple_encoder)

	triple_encoder_model = triple_encoder_model.to(device)
	print('finished loading the model')
	MSE_criteriion = nn.MSELoss()

	val_text_tensor = torch.load('Tensor/val_tokenized_text_tensor.pt')
	val_image_tensor = torch.load('Tensor/val_image_tensor.pt')
	val_dataloader = loader.TripleDataLoader(val_text_tensor,val_image_tensor,batch_size=args.batch_size)

	train_text_tensor = torch.load('Tensor/train_tokenized_text_tensor.pt')
	train_image_tensor = torch.load('Tensor/train_image_tensor.pt')
	train_dataloader = loader.TripleDataLoader(train_text_tensor,train_image_tensor,batch_size=args.batch_size)
	print('finished building the dataloader')
	optimizer = torch.optim.Adam(triple_encoder_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

	train_dict = {'total_loss':[],'tri_loss':[],'RMSE_loss':[]}
	val_dict = {'total_loss':[],'tri_loss':[],'RMSE_loss':[]}

	start = timeit.default_timer()
	for epoch in range(args.epochs):
		triple_encoder_model.train()
		tri_loss = []
		RMSE_loss = []
		total_loss = []
		
		for batch in train_dataloader:
			loss_batch = train_process(args,batch,triple_encoder_model,optimizer)
			RMSE_loss.append(loss_batch[0])
			tri_loss.append(loss_batch[1])
			total_loss.append(loss_batch[2])

		train_dict['total_loss'].append(np.mean(total_loss))
		train_dict['RMSE_loss'].append(np.mean(RMSE_loss))
		train_dict['tri_loss'].append(np.mean(tri_loss))

		if epoch % args.eval_every == 0:
			triple_encoder_model.eval()
			tri_loss = []
			RMSE_loss = []
			total_loss = []
			with torch.no_grad:
				for batch in val_dataloader:
					loss_batch = eval_process(args,batch,triple_encoder_model)
					RMSE_loss.append(loss_batch[0])
					tri_loss.append(loss_batch[1])
					total_loss.append(loss_batch[2])
			val_dict['total_loss'].append(np.mean(total_loss))
			val_dict['RMSE_loss'].append(np.mean(RMSE_loss))
			val_dict['tri_loss'].append(np.mean(tri_loss))
			
			array_image, array_text = embed_text_image(triple_encoder_model,train_dataloader,args.embedding_dim)
			train_score,train_rank = MAP_k(array_text,array_image,args.MAP_k)

			array_image, array_text = embed_text_image(triple_encoder_model,val_dataloader,args.embedding_dim)
			val_score,val_rank = MAP_k(array_text,array_image,args.MAP_k)
			print(' ')
			stop = timeit.default_timer()
			print('train epoch: {}, train loss: {:.3f}, val loss: {:.3f}, train score: {:.3f} , val score: {:.3f}, took {:.3f} sec'.format(
				epoch,train_dict['total_loss'][-1],val_dict['total_loss'][-1],train_score/len(train_rank),val_score/len(val_rank),stop - start))
      		start = timeit.default_timer()
      		print('----------------------------------')
    if args.save_model:
    	torch.save(triple_encoder_model.state_dict(), 'pytorch_model.bin')
    with open('validation_loss.json', 'w') as fq:
    	json.dump(val_dict, fq)
    with open('train_loss.json', 'w') as fq:
    	json.dump(train_dict, fq)



def train_process(args,batch,iterations,model,optimizer):
	"""do 1 batch training"""
	optimizer.zero_grad()
	batch = (item.to(device) for item in batch)
	input_text, input_img, input_decoy = batch
	text_output, img_output, decoy_outputt = model(input_text,input_img,input_decoy)
	ground = img_output.detach()
  	tri_loss = triplet_loss(ground,text_output,decoy_output,args.margin)
  	RMSE_loss = torch.sqrt(MSE_criterion(text_output,ground))
  	total_loss = args.weight_RMSE*RMSE_loss + (1-args.weight_RMSE)*tri_loss
  	total_loss.backward()
  	optimizer.step()
  	return RMSE_loss.item(),tri_loss.item(),total_loss.item()

def eval_process(args,batch,model):
	""" do 1 batch evaluation """
  	batch = (item.to(device) for item in batch)
  	input_text, input_img, input_decoy = batch
  	text_output, img_output, decoy_outputt = model(input_text,input_img,input_decoy)
  	ground = img_output.detach()
  	tri_loss = triplet_loss(ground,text_output,decoy_output,args.margin)
  	RMSE_loss = torch.sqrt(MSE_criterion(text_output,ground))
  	total_loss = args.weight_RMSE*RMSE_loss + (1-args.weight_RMSE)*tri_loss
  	return RMSE_loss.item(),tri_loss.item(),total_loss.item()


	return

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='image retrieval')

    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train for (default: 5)')
    parser.add_argument('--embedding_dim', type=int, default=768, metavar='E',
                        help='number of final embedding dimensions (default: 768)')
    parser.add_argument('--vocab_size', type=int, default=15224, metavar='E',
                        help='number of vocab including <PAD> (default: 15224)')
    parser.add_argument('--vocab_size', type=int, default=15224, metavar='E',
                        help='number of vocab including <PAD> (default: 15224)')
    parser.add_argument('--hidden_dim', type=int, default=150, metavar='E',
                        help='number of hidden unit in the text embedding model (default: 150)')
    parser.add_argument('--word_embedding_dim', type=int, default=200, metavar='E',
                        help='number of unit in the word embedding layer (default: 200)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='E',
                        help='drop out probability in the text embedding model (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=32, metavar='E',
                        help='batch size for training and evaluation (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='E',
                        help='learning rate for Adam optimizer (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='E',
                        help='weight decay for Adam optimizer (default: 0)')

    parser.add_argument('--weight_RMSE', type=float, default=1, metavar='E',
                        help='weight of the RMSE loss (weight of the triplet loss is 1-weight_RMSE) (default: 1)')
    parser.add_argument('--margin', type=float, default=0.1, metavar='E',
                        help='margin of the triplet loss (default: 0.1)')

    parser.add_argument('--train_text', type=str, default='Tensor/train_tokenized_text_tensor.pt', metavar='E',
                        help='pt file of the tokenized text')
    parser.add_argument('--train_att', type=str, default='Tensor/train_tokenized_att_tensor.pt', metavar='E',
                        help='pt file of the tokenized attention mask')
    parser.add_argument('--train_image', type=str, default='Tensor/train_tokenized_image_tensor.pt', metavar='E',
                        help='pt file of the pre-processed image')

    parser.add_argument('--val_text', type=str, default='Tensor/val_tokenized_text_tensor.pt', metavar='E',
                        help='pt file of the tokenized text')
    parser.add_argument('--val_att', type=str, default='Tensor/val_tokenized_att_tensor.pt', metavar='E',
                        help='pt file of the tokenized attention mask')
    parser.add_argument('--val_image', type=str, default='Tensor/val_tokenized_image_tensor.pt', metavar='E',
                        help='pt file of the pre-processed image')
    parser.add_argument('--model_weight', type=str, default='pytorch_model.bin', metavar='E',
                        help='the weight file of pytorch model')


    parser.add_argument('--MAP_k', type=int, default=20, metavar='E',
                        help='number of nearest neighbors to query from the shared embeded dimensions (default: 20)')

    parser.add_argument('--save_model', type=bool, default=True, metavar='E',
                        help='whether to save model in the current directory (default: True)')

    main(parser.parse_args())





