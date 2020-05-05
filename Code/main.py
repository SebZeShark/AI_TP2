import numpy as np
import matplotlib.pyplot as plt
import sys
import load_datasets as ld
from NeuralNet import NeuralNet # importer la classe du Réseau de Neurones
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning
from Code import DecisionTree, load_datasets


def main():
    i = ld.load_iris_dataset(0.7)
    c = ld.load_congressional_dataset(0.7)
    m1 = ld.load_monks_dataset(1)
    m2 = ld.load_monks_dataset(2)
    m3 = ld.load_monks_dataset(3)

    """for nom, dataset, n_attr, classes in [("Iris", i, 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]),
                                 ("Congressional", c, 16, ["democrat", "republican"]),
                                 ("Monks 1", m1, 6, ["0", "1"]),
                                 ("Monks 2", m2, 6, ["0", "1"]),
                                 ("Monks 3", m3, 6, ["0", "1"])]:

    print("\n=============\n" + nom + " dimension tests\n=============")
    for d in [5,10,15,20,25,40,50]:
        data = list(zip(dataset[0], dataset[1]))
        erreur = 0
        correct = 0
        for k in range(8):
            borne_i = k*len(data)//8
            borne_s = (k+1)*len(data)//8
            train, train_labels = zip(*data[borne_i:borne_s])
            test, test_labels = zip(*(data[0:borne_i] + data[borne_s:]))
            NN_i = NeuralNet(d, n_attr, classes, profondeur=1)
            NN_i.train(train, train_labels, epochs=25)
            res = NN_i.test(test, test_labels)
            erreur += res[0]

        print("Erreur totale moyenne avec dimension " + str(d) + " : " + str(erreur/8))"""

    """ for nom, dataset, n_attr, classes, d in \
                                 [("Iris", i, 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], 50),
                                  ("Congressional", c, 16, ["democrat", "republican"], 10),
                                  ("Monks 1", m1, 6, ["0", "1"], 25),
                                  ("Monks 2", m2, 6, ["0", "1"], 50),
                                  ("Monks 3", m3, 6, ["0", "1"], 25)]:

     print("\n=============\n" + nom + " profondeur tests\n=============")
     for p in range(1, 6):
         data = list(zip(dataset[0], dataset[1]))
         erreur = 0
         correct = 0
         for k in range(8):
             borne_i = k*len(data)//8
             borne_s = (k+1)*len(data)//8
             train, train_labels = zip(*data[borne_i:borne_s])
             test, test_labels = zip(*(data[0:borne_i] + data[borne_s:]))
             NN_i = NeuralNet(d, n_attr, classes, profondeur=p)
             NN_i.train(train, train_labels, epochs=25)
             res = NN_i.test(test, test_labels)
             correct += res[1]

         print("Pourcentage correct moyen avec profondeur " + str(p + 2) + " : " + str(correct/8))
         NN_i = NeuralNet(d, n_attr, classes, profondeur=p)
         NN_i.courbe_apprentissage(dataset[0], dataset[1])"""

    """for nom, dataset, n_attr, classes, d, p in \
                                    [("Iris", i, 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], 50, 4),
                                     ("Congressional", c, 16, ["democrat", "republican"], 10, 3),
                                     ("Monks 1", m1, 6, ["0", "1"], 25, 3),
                                     ("Monks 2", m2, 6, ["0", "1"], 50, 4),
                                     ("Monks 3", m3, 6, ["0", "1"], 25, 3)]:

        print("\n=============\n" + nom + " ZERO tests\n=============")
        NN_i = NeuralNet(d, n_attr, classes, profondeur=p, zero=True)
        NN_i.courbe_apprentissage(dataset[0], dataset[1])

        print("\n=============\n" + nom + " NON-ZERO tests\n=============")
        NN_i = NeuralNet(d, n_attr, classes, profondeur=p)
        NN_i.courbe_apprentissage(dataset[0], dataset[1])"""

    print("---------Réseau de neurones-------------")
    for nom, dataset, n_attr, classes, d, p in \
                                        [("Iris", i, 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], 50, 4),
                                         ("Congressional", c, 16, ["democrat", "republican"], 10, 3),
                                         ("Monks 1", m1, 6, ["0", "1"], 25, 3),
                                         ("Monks 2", m2, 6, ["0", "1"], 50, 4),
                                         ("Monks 3", m3, 6, ["0", "1"], 25, 3)]:

            print("\n=============\n" + nom + "  tests\n=============")
            NN_i = NeuralNet(d, n_attr, classes, profondeur=p)
            NN_i.train(dataset[0], dataset[1])
            NN_i.test(dataset[2], dataset[3])

    train_congress, train_labels_congress, test_congress, test_labels_congress = load_datasets.load_congressional_dataset(
        0.7)
    treeClassifierCongress = DecisionTree.DecisionTree(dataType="house-votes")
    treeClassifierCongress.train(train_congress, train_labels_congress)
    treeClassifierCongress.predict(test_congress[0], test_labels_congress[0])

    print("---------Arbre de décision-------------")
    print("Iris")
    train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.7)
    tci = DecisionTree.DecisionTree(dataType="iris")
    tci.train(train_iris, train_labels_iris)
    tci.test(test_iris, test_labels_iris)

    print("Monk1")
    train_monk1, train_labels_monk1, test_monk1, test_labels_monk1 = load_datasets.load_monks_dataset(1)
    tcm1 = DecisionTree.DecisionTree(dataType="MONK")
    tcm1.train(train_monk1, train_labels_monk1)
    tcm1.test(test_monk1, test_labels_monk1)
    print("Monk2")
    train_monk2, train_labels_monk2, test_monk2, test_labels_monk2 = load_datasets.load_monks_dataset(2)
    tcm2 = DecisionTree.DecisionTree(dataType="MONK")
    tcm2.train(train_monk2, train_labels_monk2)
    tcm1.test(test_monk2, test_labels_monk2)
    print("Monk3")
    train_monk3, train_labels_monk3, test_monk3, test_labels_monk3 = load_datasets.load_monks_dataset(3)
    tcm3 = DecisionTree.DecisionTree(dataType="MONK")
    tcm3.train(train_monk3, train_labels_monk3)
    tcm3.test(test_monk3, test_labels_monk3)









if __name__ == "__main__":
    main()