import numpy as np
import matplotlib.pyplot as plt
import sys
import load_datasets as ld
from NeuralNet import NeuralNet # importer la classe du Réseau de Neurones
# importer d'autres fichiers et classes si vous en avez développés
# importer d'autres bibliothèques au besoin, sauf celles qui font du machine learning


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









if __name__ == "__main__":
    main()