import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
from DsRule_GreedySearch import GradientDescentGreedySearch
from DsRule_GreedySearch import SimulatedAnnealingGreedySearch
from DsRule_Explainer import Explainer

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
    #### True:Numeric
    #### False:Un-Numeric
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
        return boundary_instance

    def instance_explainer(self, instance,length=3):
        category=self.black_box_model.predict(instance)
        rule_range=self.class_level_explainer(category[0])
        instance_range={}
        for column_name in rule_range:
            if len(np.array(rule_range[column_name]).shape)==1:
                instance_range[column_name]=list(instance[column_name])[0]
            else:
                for interval in rule_range[column_name]:
                    if list(instance[column_name])[0]<=interval[1] and list(instance[column_name])[0]>=interval[0]:
                        if interval[1]==interval[0]:
                            instance_range[column_name]=interval[0]
                        else:
                            instance_range[column_name]=interval
                        break
        explainer = Explainer(self.dataset,self.black_box_model)
        predicates = explainer.instance_explainer(instance, instance_range)
        print("If ",end='')
        for i in range(length):
            if i > 0:
                print(" and ",end='')
            print("feature ",predicates[i][0]," is ",predicates[i][1],end='')
        print(" Then class ", category[0])
        print("Accuracy is ", predicates[length-1][2])
        return predicates

    def class_level_explainer(self, category, cluster_model=KMeans, n_clusters=5, max_iter=2000):
        predicted_dataset = pd.DataFrame()
        predicted_dataset['target'] = self.black_box_model.predict(self.dataset)
        assert category in predicted_dataset['target'].unique()
        boundary_points = []
        class_data = self.dataset[predicted_dataset['target'] == category]

        cluster_model = cluster_model(n_clusters=n_clusters, max_iter=max_iter).fit(class_data)
        center_points = pd.DataFrame(data=cluster_model.cluster_centers_, columns=self.column_names,index=list(range(n_clusters)))
        for i in range(len(center_points)):
            boundary_point = self.fast_instance_explainer(center_points[i:i+1])
            boundary_points.append(boundary_point)
        rule_range = self.rule_translation(center_points, boundary_points,class_data)
        return rule_range

    #### unfinished
    #### 为整合区间
    #### 整数值没有
    def rule_translation(self,center_points,boundary_points,data):
        rule_range={}
        for i in range(len(center_points)):
            center=center_points[i:i+1]
            boundary=boundary_points[i]
            if boundary is None:
                continue
            for column_name in self.column_names:
                if not self.column_categories[column_name] and rule_range.get(column_name,None) is not None:
                    rule_range[column_name]=sorted(list(data[column_name].unique()))
                else:
                    interval = abs(list(center[column_name])[0]-list(boundary[column_name])[0])

                    left = min(list(center[column_name])[0] - interval, list(center[column_name])[0] + interval)
                    left = max(left, self.column_min_values[column_name])
                    right = max(list(center[column_name])[0] - interval, list(center[column_name])[0] + interval)
                    right = min(right, self.column_max_values[column_name])
                    if self.dataset[column_name].dtype == np.int64:
                        left = int(round(left, 0))
                        right = int(round(right, 0))
                    else:
                        left = round(left, 2)
                        right = round(right, 2)

                    if rule_range.get(column_name, None) is None:
                        rule_range[column_name] = [[left, right]]
                    else:
                        rule_range[column_name].append([left, right])
        rule_range = self.integrate(rule_range)
        return rule_range

    def integrate(self, intervals):
        new_intervals = {}
        for column_name in intervals:
            if len(np.array(intervals[column_name]).shape) == 1:
                new_intervals[column_name] = intervals[column_name]
            else:
                interval = intervals[column_name]
                interval = sorted(interval, key=lambda x: x[0])
                new_interval = []
                for inter in interval:
                    if len(new_interval) == 0:
                        new_interval.append(inter)
                    elif inter[0] > new_interval[-1][1]:
                        new_interval.append(inter)
                    elif inter[1] >= new_interval[-1][1]:
                        new_interval[-1] = [new_interval[-1][0], inter[1]]
                    else:
                        pass
                new_intervals[column_name] = new_interval
        return new_intervals


