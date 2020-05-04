"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
    * train 	: pour entrainer le modèle sur l'ensemble d'entrainement
    * predict 	: pour prédire la classe d'un exemple donné
    * test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes train, predict et test de votre code.
"""


# le nom de votre classe
# NeuralNet pour le modèle Réseaux de Neurones
# DecisionTree le modèle des arbres de decision
import math
import random


class Classifier:  # nom de la class à changer

    def __init__(self, dimension, n_attr, labels_possibles, profondeur=1, poids_par_defaut=0.05, apprentissage=1):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.profondeur = profondeur
        self.dimension = dimension
        self.poids_par_defaut = poids_par_defaut
        self.labels_possibles = labels_possibles
        self.apprentissage = apprentissage
        inputrow = []
        all_hidden_rows = []
        outputrow = []
        for i in range(dimension):
            inputs = []
            for j in range(n_attr):
                inputs += [poids_par_defaut]
            inputrow += [inputs]
        for i in range(profondeur - 1):
            hiddenrow = []
            for j in range(dimension):
                hidden = []
                for k in range(dimension):
                    hidden += [self.poids_par_defaut]
                hiddenrow += [hidden]
            all_hidden_rows += [hiddenrow]
        for i in range(len(labels_possibles)):
            outputs = []
            for j in range(dimension):
                outputs += [self.poids_par_defaut]
            outputrow += [outputs]
        self.poids = [[inputrow]] + all_hidden_rows + [[outputrow]]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def compute(self, inputs, poids):
        somme = 0
        for i, p in zip(inputs, poids):
            somme += i*poids
        return self.sigmoid(poids)

    def train(self, train, train_labels, epochs=5):  # vous pouvez rajouter d'autres attribus au besoin
        """
        c'est la méthode qui va entrainer votre modèle,
        train est une matrice de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        train_labels : est une matrice de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire



        ------------

        """
        train_data = list(zip(train, train_labels))

        for e in range(epochs):
            random.shuffle(train_data)
            attributes_lists, labels = zip(*train_data)

            for attributes, label in zip(attributes_lists, labels):
                outputs = [[attributes]]
                current_outputs = attributes
                for i in self.poids:
                    new_outputs = []
                    for j in i:
                        new_outputs += [self.compute(current_outputs, j)]
                    current_outputs = new_outputs
                    outputs += [new_outputs]

                # Retro-propagation

                deltas = []
                for i in range(len(self.labels_possibles)):
                    score = 0
                    if self.labels_possibles.index(label) == i:
                        score = 1
                    o = outputs[-1][i]
                    deltas += [o * (1 - o) * (score - o)]

                for i in range(len(outputs) - 2, -1, -1):
                    new_deltas = []
                    if i != 0:
                        for j in range(outputs[i]):
                            somme = 0
                            for p, d in zip(self.poids[i], deltas):
                                somme += p[j] * d

                            o = outputs[i][j]
                            new_deltas += [o * (1 - o) * somme]
                    for j in range(outputs[i]):
                        for k in range(len(deltas)):
                            self.poids[i][j][k] = self.apprentissage * deltas[k] * outputs[i][j]
                    deltas = new_deltas






    def predict(self, exemple, label):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification

        """
        output = exemple
        for i in self.poids:
            new_outputs = []
            for j in i:
                new_outputs += [self.compute(output, j)]
            output = new_outputs

        resultat = max(range(len(output)), key=lambda x: output[x])

        return self.labels_possibles[resultat] == label, self.labels_possibles[resultat]


    def test(self, test, test_labels):
        """
        c'est la méthode qui va tester votre modèle sur les données de test
        l'argument test est une matrice de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        test_labels : est une matrice taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les données de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy (ou le taux d'erreur)

        Bien entendu ces tests doivent etre faits sur les données de test seulement

        """
        for exemple, label in zip(test, test_labels):
            print(self.predict(exemple, label))
# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.
