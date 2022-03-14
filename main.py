from config import cfg
from config import log_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import dgl
from dgl import DGLGraph
import networkx as nx
import random
import pickle
import scipy.sparse as sp
import itertools as it
from sklearn.metrics import accuracy_score
from early_stop import EarlyStopping
from logger import setup_logger
import logging
import random
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.linalg import inv
from subtask import pairwise_prediction
import os
from gcn import GNN

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



	

logger_name = cfg.dyn_w +'_' +cfg.gnn +'_' +cfg.task+'_MTGNN_' + cfg.dataset
time_now = setup_logger('MTGNN', logger_name, level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('MTGNN')

log_config(cfg,logger)

class Task(nn.Module):
	def __init__(self,model):
		super(Task, self).__init__()
		self.model  = model
		self.linear = nn.Linear(cfg.h_size,cfg.num_class).to(device)

	def forward(self,g,features,calc_mad):
		x,output = self.model(g,features,calc_mad)
		logits = self.linear(x)
		return x,logits,output

def dynamic_weight_average(loss_tracker,epoch):
	num_tasks = len(loss_tracker)
	weight_assign = []
	if epoch<5:
		for k in range(num_tasks):
			weight_assign.append(1.0/num_tasks)
		weight_assign = np.array(weight_assign)
	else:
		loss_sum = sum(loss_tracker)
		for k in range(num_tasks):
			weight_assign.append(loss_tracker[k]/loss_sum)
		weight_assign = np.array(weight_assign)/cfg.T
		max_w = np.max(weight_assign)
		weight_assign = weight_assign - max_w
		w_exp = np.exp(weight_assign)
		weight_assign = num_tasks*w_exp/w_exp.sum()
	return weight_assign

def main():
	dataset = cfg.dataset
	sparse_node_label = pickle.load(open(dataset + '/' + dataset +'_node_label.pkl', "rb"))
	dense_node_label  = np.argmax(sparse_node_label,1)
	sparse_adj = pickle.load(open(dataset + '/' + dataset +'_adj.pkl', "rb"))
	dense_adj = sparse_adj.A
	G_nx = nx.from_numpy_matrix(dense_adj,create_using=nx.Graph())
	G = DGLGraph(G_nx)
	G.add_edges(G.nodes(),G.nodes()) # self_loop
	sparse_node_features = pickle.load(open(dataset + '/' + dataset + '_features.pkl','rb'))
	dense_node_features = sparse_node_features.toarray()

	train_nodes = []
	val_nodes   = []
	test_nodes  = []
	for i in range(cfg.num_class):
		x = np.argwhere(dense_node_label==i).reshape(-1)
		np.random.shuffle(x)
		train_nodes.append(x[0:cfg.split[0]])
		val_nodes.append(x[cfg.split[0]:cfg.split[0] + cfg.split[1]])
		test_nodes.append(x[cfg.split[0] + cfg.split[1]:])

	pos_pairs = []
	for x in train_nodes:
		pos_pairs += list(it.combinations(x,2))
	neg_pairs = []

	cross_labels = it.combinations(list(range(cfg.num_class)),2)
	for i,j in cross_labels:
		x = train_nodes[i]
		y = train_nodes[j]
		neg_pairs += list(it.product(x,y,repeat=1))
	neg_pairs = random.sample(neg_pairs,cfg.neg_size*len(pos_pairs))

	# edges = np.array(pos_edges + neg_edges)
	# edge_labels = np.concatenate([np.ones(len(pos_edges)),np.zeros(len(neg_edges))],0)

	train_pairs = np.array(pos_pairs + neg_pairs)
	train_pair_labels = np.concatenate([np.ones(len(pos_pairs)),np.zeros(len(neg_pairs))],0)

	pos_pairs = []
	for x in val_nodes:
		pos_pairs += list(it.combinations(x,2))
	neg_pairs = []

	cross_labels = it.combinations(list(range(cfg.num_class)),2)
	for i,j in cross_labels:
		x = val_nodes[i]
		y = val_nodes[j]
		neg_pairs += list(it.product(x,y,repeat=1))
	neg_pairs = random.sample(neg_pairs,cfg.neg_size*len(pos_pairs))

	val_pairs = np.array(pos_pairs + neg_pairs)
	val_pair_labels = np.concatenate([np.ones(len(pos_pairs)),np.zeros(len(neg_pairs))],0)

	train_nodes = np.concatenate(train_nodes,0)
	val_nodes   = np.concatenate(val_nodes,0)
	test_nodes  = np.concatenate(test_nodes,0)

	y_train_true     = dense_node_label[train_nodes]
	y_val_true       = dense_node_label[val_nodes]
	y_test_true      = dense_node_label[test_nodes]

	### random walk based link sampling ###
	A = dense_adj
	A_tilde = A + np.identity(A.shape[0])
	D = A_tilde.sum(axis=1)

	Lambda = np.identity(A.shape[0])

	L = np.diag(D) - A
	P = inv(L + cfg.alpha * Lambda)

	train_dict = dict(zip(train_nodes,map(int,y_train_true)))
	probability = []
	for k in range(cfg.num_class):
		nodes = train_nodes[y_train_true==k]
		prob = P[:, nodes].sum(axis = 1).flatten()
		probability.append(prob)
	probability = np.stack(probability,axis=1)
	probability = probability/np.sum(probability,axis=1,keepdims=True)
	probability = np.nan_to_num(probability)
	rw_predicted_labels = np.argmax(probability,axis=1)
	#print(len(rw_predicted_labels))

	for node in train_nodes:
		rw_predicted_labels[node] = dense_node_label[node]
		probability[node][dense_node_label[node]] = 1.0
	
	pos_edges = []
	neg_edges = []
	for src,dst in list(G_nx.edges()):
		src_label = rw_predicted_labels[src]
		dst_label = rw_predicted_labels[dst]
		# if src_label!=dst_label:
		#     if probability[src][src_label]>cfg.threshold and probability[dst][dst_label]>cfg.threshold:
		#         continue
		if src_label==dst_label:
			if probability[src][src_label]>cfg.threshold and probability[dst][dst_label]>cfg.threshold:
				pos_edges.append([src,dst])
	logger.info('original number of egdes : {:d} now number of egdes : {:d}'.format(len(G_nx.edges()),len(pos_edges)))

	if cfg.dataset!='pubmed':
		candidate = list(nx.non_edges(G_nx))
		num_neg_edges = cfg.neg_size*len(pos_edges)
		neg_edges = []
		k = 0

		random.shuffle(candidate)
		for src,dst in candidate:
			src_label = rw_predicted_labels[src]
			dst_label = rw_predicted_labels[dst]
			if src_label!=dst_label:
				if probability[src][src_label]>cfg.threshold and probability[dst][dst_label]>cfg.threshold:
					neg_edges.append([src,dst])
					k += 1
			if k==num_neg_edges:
				break
	else:
		neg_edges = []
		for src,dst in pos_edges:
			i = 0
			while i<cfg.neg_size:
				if random.random()>0.5:
					dst = random.randint(0,A.shape[0]-1)
				else:
					src = random.randint(0,A.shape[0]-1)
				src_label = rw_predicted_labels[src]
				dst_label = rw_predicted_labels[dst]
				if src_label!=dst_label:
					if probability[src][src_label]>cfg.threshold and probability[dst][dst_label]>cfg.threshold:
						neg_edges.append([src,dst])
						i += 1

	edges = np.array(pos_edges + neg_edges)
	edge_labels = np.concatenate([np.ones(len(pos_edges)),np.zeros(len(neg_edges))],0)
	count = 0
	for i,(src,dst) in enumerate(pos_edges):
		if dense_node_label[src]==dense_node_label[dst]:
			x = 1
		else:
			x = 0
		if x==1:
			count += 1
	print('true links : {:d} all links : {:d}'.format(count,len(pos_edges)))

	net = Task(GNN()).to(device)
	optimizer = torch.optim.Adam(net.parameters(),lr=cfg.lr)

	early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

	# to track the training loss as the model trains
	train_loss = 0
	# to track the validation loss as the model trains
	valid_loss = 0
	# initialize for loss tracker
	if cfg.task =="nlp":
		loss_tracker = [1.0,1.0,1.0]
	else:
		loss_tracker =[1.0]

	for i in range(cfg.epoch):
		net.train()
		x,y,_ = net(G.to(device),torch.FloatTensor(dense_node_features).to(device),False)

		train_logp = F.log_softmax(y[train_nodes], 1)
		n_loss = F.nll_loss(train_logp, torch.LongTensor(y_train_true).to(device))

		if "l" in cfg.task:
			edge_src = x[edges[:,0]]
			edge_dst = x[edges[:,1]]
			logits   = torch.sum(torch.mul(edge_src,edge_dst),1)
			l_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([cfg.neg_size]).to(device))(logits,torch.FloatTensor(edge_labels).to(device))


		if "p" in cfg.task:
			pair_src = y[train_pairs[:,0]]
			pair_dst = y[train_pairs[:,1]]
			logits   = torch.sum(torch.mul(pair_src,pair_dst),1)
			p_loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([cfg.neg_size]).to(device))(logits,torch.FloatTensor(train_pair_labels).to(device))


		if cfg.dyn_w == "1":
			weight_assign = dynamic_weight_average(loss_tracker,i)
		else:
			weight_assign = loss_tracker


		if cfg.task=="nlp":
			loss = weight_assign[0]*n_loss + weight_assign[1]*l_loss + weight_assign[2]*p_loss
		elif cfg.task == "n":
			loss = n_loss

		train_n_loss = n_loss.item()
		if "l" in cfg.task:
			train_l_loss = l_loss.item()
		if "p" in cfg.task:
			train_p_loss = p_loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_train_pred = np.argmax(train_logp.detach().cpu().numpy(),1)
		train_acc = accuracy_score(y_train_true,y_train_pred)

		train_loss = n_loss.item()

		net.eval()
		with torch.no_grad():
			x,y,_ = net(G.to(device),torch.FloatTensor(dense_node_features).to(device),False)
			val_logp = F.log_softmax(y[val_nodes], 1)
			n_loss = F.nll_loss(val_logp, torch.LongTensor(y_val_true).to(device))

			if "l" in cfg.task:
				edge_src = x[edges[:,0]]
				edge_dst = x[edges[:,1]]
				logits   = torch.sum(torch.mul(edge_src,edge_dst),1)
				l_loss = nn.BCEWithLogitsLoss()(logits,torch.FloatTensor(edge_labels).to(device))

			if "p" in cfg.task:
				pair_src = y[val_pairs[:,0]]
				pair_dst = y[val_pairs[:,1]]
				logits   = torch.sum(torch.mul(pair_src,pair_dst),1)
				p_loss = nn.BCEWithLogitsLoss()(logits,torch.FloatTensor(val_pair_labels).to(device))

			if cfg.task != 'n':
				if cfg.dyn_w =="1":
					for t in list(cfg.task):
						if t =="n":
							train_t_loss = train_n_loss
							t_loss = n_loss.item()
						elif t =="l":
							train_t_loss = train_l_loss
							t_loss = l_loss.item()
						elif t == "p":
							train_t_loss = train_p_loss
							t_loss = p_loss.item()
						loss_tracker.append((train_t_loss-t_loss)/train_t_loss)

			y_val_pred = np.argmax(val_logp.detach().cpu().numpy(),1)
			val_acc = accuracy_score(y_val_true,y_val_pred)



			valid_loss = n_loss.item()

			if (i+1)%10==0:
				logger.info('Epoch {:05d} | Train Loss {:.4f} Valid loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f}'.format(
					i+1,train_loss,valid_loss,train_acc,val_acc))
			if (i+1)>cfg.start_es:
				early_stopping(valid_loss, net)

			if early_stopping.early_stop:
				logger.info("Early stopping")
				break

	net.load_state_dict(torch.load('./saver/' + cfg.dataset + '_' + cfg.task + '_' + cfg.current_time + '_checkpoint.pt'))

	net.eval()
	x,logits,output = net(G.to(device),torch.FloatTensor(dense_node_features).to(device),cfg.mad_calc)


	test_logp = F.log_softmax(logits[test_nodes], 1)
	y_test_pred = np.argmax(test_logp.detach().cpu().numpy(),1)
	test_acc = accuracy_score(y_test_true,y_test_pred)
	logger.info('Test accuracy_score = {:.4f}'.format(test_acc))

if __name__ == '__main__':
	main()