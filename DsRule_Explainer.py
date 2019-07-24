import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Explainer(object):
    def __init__(self, dataset, black_box):
        self.dataset = dataset
        self.black_box = black_box
        self.dataset['pred'] = self.black_box.predict(self.dataset)

    def coverage(self, predicate, condition, dataset, predicted_class):
        total = len(self.dataset[self.dataset['pred'] == predicted_class])
        if type(condition) == type([]):
             part = len(dataset.ix[(dataset[predicate] > condition[0]) & (dataset[predicate] < condition[1])])
        else:
             part = len(dataset[dataset[predicate] == condition])
        return part/total



    def accuracy(self, predicate, condition, dataset, predicted_class):
        if type(condition) == type([]):
            dataset = dataset.ix[(dataset[predicate]>condition[0]) & (dataset[predicate]<condition[1])]
            acc = len(dataset[dataset['pred'] == predicted_class])/len(dataset)
            return acc
        else:
            dataset = dataset[dataset[predicate] == condition]
            acc = len(dataset[dataset['pred'] == predicted_class]) / len(dataset)
            return acc

    def class_explainer(self, category, predicates):
        pass

    def instance_explainer(self, instance, predicates):
        self.predicate = []
        pred_class = self.black_box.predict(instance)[0]
        iter_dataset = self.dataset.copy(deep=True)
        for p in range(len(predicates)):
            maxAccuracy = None
            selPredicate = None
            selCoverage = None
            for predicate in predicates:
                cur_accuracy = self.accuracy(predicate, predicates[predicate], iter_dataset, pred_class)
                if maxAccuracy is None or cur_accuracy > maxAccuracy:
                    maxAccuracy = cur_accuracy
                    selPredicate = predicate
                    selCoverage = self.coverage(selPredicate,predicates[selPredicate],iter_dataset,pred_class)
            predicate = selPredicate
            condition = predicates[selPredicate]
            self.predicate.append((predicate, condition, maxAccuracy,selCoverage))
            predicates.pop(selPredicate)
            if type(condition) == type([]):
                iter_dataset = iter_dataset.ix[(iter_dataset[predicate] > condition[0]) & (iter_dataset[predicate] < condition[1])]
            else:
                iter_dataset = iter_dataset[iter_dataset[predicate] == condition]
        return self.predicate



