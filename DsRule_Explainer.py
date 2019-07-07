import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class InstanceExplainer(object):
    def __init__(self, dataset, instance, boundary_instance, predicted_class):
        self.dataset = dataset
        self.instance = instance
        self.boundary_instance = boundary_instance
        self.predicted_class=predicted_class
        self.predicates={}
        self.texts_to_show={}
        self.accuracy_of_rules=[]
        #each item in rules is the best rule picked based on the previous rules
        self.rules=[]


    def greedy_construct_rules(self):
        sample=self.dataset
        for i in len(self.predicates):
            round_of_accuracy={}
            for column_name,predicate in self.predicates:
                if len(predicate)==1:
                    sample=sample[sample[column_name]==predicate]
                    accuracy=sum(sample['predicted_class'] == self.predicted_class)/len(sample)
                    round_of_accuracy[accuracy]=(column_name,predicate)
                else:
                    sample=sample[sample[column_name]<=predicate[1] and sample[column_name]>=predicate[0]]
                    accuracy = sum(sample['predicted_class'] == self.predicted_class) / len(sample)
                    round_of_accuracy[accuracy] = (column_name, predicate)
            best_accuracy=max(round_of_accuracy.keys())
            picked_rule = round_of_accuracy[best_accuracy]

            self.accuracy_of_rules.append(best_accuracy)
            self.rules.append(picked_rule)




    def explain_instance(self,num_rules=1):
        self.greedy_construct_rules()
        for i in range(num_rules):
            print(self.texts_to_show[i])

    def extract_rules(self):
        columns = np.array(self.dataset.columns)
        for k in range(len(columns)):
            if self.boundary_instance[k] == self.instance[k]:
                self.predicates[columns[k]] = [self.instance[k]]
            else:
                self.predicates[columns[k]] = [min(self.instance[k], self.boundary_instance[k]),
                                               max(self.instance[k], self.boundary_instance[k])]

