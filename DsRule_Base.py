import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from DsRule_GreedySearch import GradientDescentGreedySearch
from DsRule_GreedySearch import SimulatedAnnealingGreedySearch
from DsRule_Explainer import InstanceExplainer


class DsRule:
    def __init__(self, dataset, black_box_model):
        assert isinstance(dataset, pd.DataFrame), "The parameter of dataset only accepts pandas dataframe as input"
        self.dataset = dataset
        self.black_box_model = black_box_model
        self.column_names = list(self.dataset.columns)
        self.column_max_values = dict(self.dataset.max(axis=0))
        self.column_min_values = dict(self.dataset.min(axis=0))
        assert self.column_names == list(self.column_min_values.keys()), "non-equal column names"
        assert self.column_names == list(self.column_max_values.keys()), "non-equal column names"
        self.column_categories = {name: self.category_check(name) for name in self.column_names}
        self.column_ranges = {name: self.range_calculate(name) for name in self.column_names}

    def range_calculate(self, name):
        if self.column_categories[name]:
            return self.column_min_values[name], self.column_max_values[name]
        else:
            return None

    def category_check(self, name):
        if is_numeric_dtype(self.dataset[name]):
            if self.dataset[name].dtype == np.float64:
                return True
            elif self.dataset[name].dtype == np.int64:
                if len(self.dataset[name].unique()) >= 0.1*len(self.dataset):
                    return True
                else:
                    return False
            else:
                raise TypeError('Unsupported data type of column: '+name)
        else:
            return False

    def fast_instance_explainer(self, instance):
        numeric_column_names=[]
        for column_name in self.column_categories:
            if self.column_categories[column_name]:
                numeric_column_names.append(column_name)
        greedy_search = SimulatedAnnealingGreedySearch(self.black_box_model, numeric_column_names, self.column_ranges)
        boundary_instance = greedy_search.simulated_annealing_with_all(instance)
        if boundary_instance is None:
            print("Couldn't find the decision set of the instance, please try other methods or change the " +
                  "parameter of the function")
        else:
            print(boundary_instance)
        #instance_explainer=InstanceExplainer(self.dataset,instance,boundary_instance,predicted_class)

   #def global_instance_explainer(self,instance):
