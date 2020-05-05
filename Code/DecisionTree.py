import numpy as np
import math
import operator

iris_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
vote_labels = {'republican': 0, 'democrat': 1, 'n': 0, '?': 1, 'y': 2}


class DecisionTree:

	def __init__(self, **kwargs):
		self.dataType = kwargs['dataType']
		if (self.dataType == "house-votes"):
			self.possible_labels = ["republican", "democrat"]
			self.possible_values = [0, 1, 2]
		elif (self.dataType == "MONK"):
			self.possible_labels = ["0", "1"]
			self.possible_values = [0,1,2,3,4]
		elif (self.dataType == "iris"):
			self.possible_labels = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
		

	def train(self, train, train_labels):
		train_list = []
		train_labels_list = list(train_labels)
		for row in train:
			train_list.append(list(row))
		if (self.dataType == "house-votes"):
			self.arbre = self.buildNoeudDiscret(train_list, train_labels_list)
		elif (self.dataType == "MONK"):
			self.arbre = self.buildNoeudDiscret(train_list, train_labels_list)
		elif (self.dataType == "iris"):
			self.arbre = self.buildNoeudContinu(train_list, train_labels_list)
		#print(self.arbre.show())

	def predict(self, exemple, label):
		result = self.arbre.solve(exemple)
		return result == label, label

	def test(self, test, test_labels, test_name =""):
		metriques = {}
		for i in set(test_labels):
			metriques[i] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

		for i in zip(test, test_labels):
			resultat = self.predict(i[0], i[1])
			if resultat[0]:
				for label, m in metriques.items():
					if label == i[1]:
						metriques[label]['TP'] += 1
					else:
						metriques[label]['TN'] += 1
			else:
				for label, m in metriques.items():
					if label == i[1]:
						metriques[label]['FN'] += 1
					elif label == resultat[1]:
						metriques[label]['FP'] += 1
					else:
						metriques[label]['TN'] += 1

		for label, m in metriques.items():
			print("\nClasse: {}, {}".format(test_name, label))
			print("Matrice de confusion: " + str(m))
			print("Accuracy: " + str((m['TP'] + m['TN']) / (m['TP'] + m['TN'] + m['FP'] + m['FN'])))
			print(" -----------------------------------")

	def buildNoeudDiscret(self, data, data_labels):
		if (data == []):
			return EndNode('unknown')
		if (data[0] == []):
			return EndNode('unknown')
		if (all(elem == data_labels[0] for elem in data_labels)):
			return EndNode(data_labels[0])
		t = len(data_labels)
		entropieS = 0
		for label in self.possible_labels:
			c = data_labels.count(label)
			p = c/t
			e = (-p)*math.log2(p) if p != 0 else 0
			entropieS += e
		sds = self.makeSplitDataSets(data, data_labels)
		gains = []
		for split in sds:
			somme = 0
			dic = self.cool(split)
			for key, value in dic.items():
				tt = sum(value)
				temp = 0
				for bb in value:
					p = bb/tt
					e = (-p)*math.log2(p) if p != 0 else 0
					temp += e
				somme += (tt/t)*temp
			gain = entropieS-somme
			gains.append(gain)
		best = max(gains)
		bestIndex = gains.index(best)
		next_datas, next_data_labels = self.splitTableByAttribute(data, data_labels, bestIndex)
		noeuds = {}
		for value in self.possible_values:
			noeud =EndNode('unknown') if value not in next_datas.keys() else  self.buildNoeudDiscret(next_datas[value], next_data_labels[value])
			noeuds[value] = noeud
		def choosingFunction(dataSet):
			return self.possible_values[dataSet[bestIndex]]
		return DecisionNode(noeuds, choosingFunction)

	def buildNoeudContinu(self, data, data_labels):
		if (data == []):
			return EndNode('unknown')
		if (data[0] == []):
			return EndNode('unknown')
		if (all(elem == data_labels[0] for elem in data_labels)):
			return EndNode(data_labels[0])
		t = len(data_labels)
		entropieS = 0
		for label in self.possible_labels:
			c = data_labels.count(label)
			p = c/t
			e = (-p)*math.log2(p) if p != 0 else 0
			entropieS += e
		sds = self.makeSplitDataSets(data, data_labels)
		gains = []
		value = []
		for split in sds:
			split.sort(key=lambda pair: pair[0])
			dic = self.cool2(split)
			tGain = {}
			for key, value in dic.items():
				buff = 0
				tt = sum(value[0])
				if (tt != 0):
					temp = 0
					for bb in value[0]:
						p = bb/tt
						e = (-p)*math.log2(p) if p != 0 else 0
						temp += e
					buff += (tt/t)*temp
				tt = sum(value[1])
				if (tt != 0):
					temp = 0
					for bb in value[1]:
						p = bb/tt
						e = (-p)*math.log2(p) if p != 0 else 0
						temp += e
					buff += (tt/t)*temp
				tGain[key] = entropieS - buff
			gain = max(tGain, key=tGain.get)
			gains.append((tGain[gain], gain))
		best = max(gains)[0]
		temp = [item for item in gains if item[0] == best][0]
		bestIndex = gains.index(temp)
		value = temp[1]
		next_datas, next_data_labels = self.orderTablesByAttribute(data, data_labels, bestIndex, value)
		noeud1 = self.buildNoeudContinu(next_datas[0], next_data_labels[0])
		noeud2 = self.buildNoeudContinu(next_datas[1], next_data_labels[1])
		noeuds = [noeud1, noeud2]
		def choosingFunction(dataSet):
			return 0 if dataSet[bestIndex] < value else 0
		return DecisionNode(noeuds, choosingFunction)

	def makeSplitDataSets(self, data, data_labels):
		splitDataSets = []
		for i in range(len(data[0])):
			row = []
			for j in range(len(data)):
				row.append((data[j][i], data_labels[j]))
			splitDataSets.append(row)
		return splitDataSets

	def cool(self, split):
		possible = {}
		for value, label in split:
			if (value not in possible):
				possible[value] = [0 for i in self.possible_labels]
			i = self.possible_labels.index(label)
			possible[value][i] += 1
		return possible

	def cool2(self, split):
		possible = {}
		for i in range(1, len(split)):
			cur, pre = split[i], split[i-1]
			if cur[1] != pre[1]:
				avg = (cur[0] + pre[0])/2
				possible[avg] = [[0 for i in self.possible_labels], [0 for i in self.possible_labels]]
		for value, label in split:
			for key in possible:
				i = self.possible_labels.index(label)
				j = 0 if value < key else 1
				possible[key][j][i] +=1
		return possible

	def orderTablesByAttribute(self, data, data_labels, splitIndex, value):
		mix = zip(data, data_labels)
		temp = sorted(mix, key=lambda pair: pair[0][splitIndex])
		res1, res2 = zip(*temp)
		t1, t2 = [[],[]], [[],[]]
		for i in range(len(res1)):
			row = res1[i]
			row = row
			v = row.pop(splitIndex)
			j = 0 if v < value else 1
			t1[j].append(row)
			t2[j].append(res2[i])
		return t1, t2

	def splitTableByAttribute(self, data, data_labels, splitIndex):
		t1, t2 = {}, {}
		for i in range(len(data)):
			row = data[i]
			row = row
			v = row.pop(splitIndex)

			if (v not in t1):
				t1[v] = []
				t2[v] = []
			t1[v].append(row)
			t2[v].append(data_labels[i])
		return t1, t2


class DecisionNode:
	def __init__(self, nodes, decisionFunc):
		self.nodes = nodes
		self.decisionFunc = decisionFunc

	def solve(self, dataSet):
		nodeI = self.decisionFunc(dataSet)
		return self.nodes[nodeI].solve(dataSet)

	def show(self, level = 0):
		ret = "-"*level + "{}\n".format(level)
		for key, node in self.nodes.items():
			temp = node.show(level+1)
			ret += temp
		return node


class EndNode:
	def __init__(self, value):
		self.value = value
		
	def solve(self, set):
		return self.value

	def show(self, level = 0):
		return "-"*level + "*{}*\n".format(self.value)
	