'''
Node2vec implementation by Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

import numpy as np
import networkx as nx
import random

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		walk = [start_node]
		
		while len(walk) < walk_length:
			cur = walk[-1]
			#遍历并排序当前结点邻近的结点
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					#根据采样得出的索引值来取出临近结点中的结点
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print("nodes:",nodes[:10])
		print("node length:",len(nodes))
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q
		# print("pre p:",p)
		# print("pre q:",q)
		unnormalized_probs = []
		#遍历dst(尾)结点的相邻的点
		for dst_nbr in sorted(G.neighbors(dst)):
			# 如果当前结点等于头结点，则添加 当前权重除以p
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			# 如果当前结点与头结点有公共边，则添加 当前权重
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
			# 其他情况则添加 当前权重除以q的值
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		#得到当前的边的概率分布
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		#返回处理后的概率分布
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed
		print("preprocess transition probs...")
		alias_nodes = {}
		# i=0
		for node in G.nodes():
			# if i>5:
			# 	break
			# print("node:",node)
			# for item in G.neighbors(node):
			# 	print(item)
			# 	break
			# 通过调用neighbors(node)来找寻当前node的邻近的点，然后排序，再遍历，存放到unormalized_probs中
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			# print("unnnormalized probs:",unnormalized_probs)
			#对unnormalized_probs进行求和
			norm_const = sum(unnormalized_probs)
			# print("norm const:",norm_const)
			# 对邻近的点除以总和来得到相邻的点所对应的概率
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			# 对生成的概率分布进一步处理，这块真挺复杂的...
			alias_nodes[node] = alias_setup(normalized_probs)
			# print("alias nodes:",alias_nodes[node])
			# break
			# i+=1

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				# print("edge:",edge)
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
				# break

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		print("alias edges:")
		for item in alias_edges:
			print(item)
			print(alias_edges[item])
			break
		print("alias node:")
		for item in alias_nodes:
			print(item)
			print(alias_nodes[item])
			break

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	'''
	# print("pre probs:",probs)
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)
	# print("init J:",J)
	# print("init q:",q)

	#通过遍历probs，索引乘以当前probs长度的值小于1时，将当前索引添加到smaller中，其他情况则将当前索引的值添加到larger中。
	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)
	
	# print("smaller:",smaller," smaller length:",len(smaller))
	# print("larger:",larger," larger length:",len(larger))

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()
	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)
	
	# print("J:",J)
	# print("q:",q)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]