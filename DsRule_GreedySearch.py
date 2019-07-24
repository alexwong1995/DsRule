import time
import random
import math
import threading


class GradientDescentGreedySearch(object):
    def __init__(self, black_box_model, column_names, column_ranges, step=0.01, iteration=5000, learning_rate=1):
        self.black_box_model = black_box_model
        self.column_names = column_names
        self.column_ranges = column_ranges
        self.step = step
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.time = None
        self.finish = False
        self.slope = {}

    def computing_one_direction(self, direction, test_instance, predict_class):
        test_instance_mod = test_instance.copy()
        for sign, column in zip(direction, self.column_names):
            value = test_instance[column]
            interval_size = self.column_ranges[column][1] - self.column_ranges[column][0]
            if sign == '0':
                value_mod = value - self.step * self.learning_rate * interval_size
            else:
                value_mod = value + self.step * self.learning_rate * interval_size
            flag = (value_mod > self.column_ranges[column][1])
            flag2 = (value_mod < self.column_ranges[column][0])
            if flag.bool() or flag2.bool():
                continue
            test_instance_mod[column] = value_mod
        prob_mod = self.black_box_model.predict_proba(test_instance_mod)
        prob_prev = self.black_box_model.predict_proba(test_instance)
        slope = prob_mod[0][predict_class][0] - prob_prev[0][predict_class][0]
        if slope < 0:
            self.slope[abs(slope)] = test_instance_mod

    def gradient_descent_of_one_step(self, directions, test_instance, predict_class):
        threads = []
        self.slope = {}
        for direction in directions:
            test_instance_mod = test_instance.copy()
            t = threading.Thread(target=self.computing_one_direction,
                                 args=(direction, test_instance_mod, predict_class))
            threads.append(t)
        for t in threads:
            t.start()
        if len(self.slope) != 0:
            max_slope = max(self.slope)
            return self.slope[max_slope]
        else:
            return None

    def gradient_descent_with_all(self, test_instance):
        prev = time.time()
        original_class = self.black_box_model.predict(test_instance)
        directions = generate_directions(self.column_names)
        num = 0
        while num <= self.iteration:
            test_instance_next = self.gradient_descent_of_one_step(directions, test_instance, original_class)
            if test_instance_next is None:
                break
            test_instance = test_instance_next
            updated_class = self.black_box_model.predict(test_instance)
            if updated_class != original_class:
                #print("yes")
                self.finish = True
                break
            num += 1
        curr = time.time()
        self.time = curr-prev
        if not self.finish:
            test_instance = None
        return test_instance


class SimulatedAnnealingGreedySearch(object):
    def __init__(self, black_box_model, column_names, column_ranges, iteration=3000, step=0.01, tmp=1e5, tmp_min=1e-3,
                 alpha=0.98, jump_rate=40):
        self.black_box_model = black_box_model
        self.column_names = column_names
        self.column_ranges = column_ranges
        self.step = step
        self.iteration = iteration
        self.tmp = tmp
        self.tmp_min = tmp_min
        self.alpha = alpha
        self.jump_rate = jump_rate
        self.time = None
        self.finish = False
        self.slope = {}

    def computing_one_direction(self, direction, test_instance, predict_class):
            test_instance_mod = test_instance.copy()
            for sign, column in zip(direction, self.column_names):
                value = test_instance_mod[column]
                interval_size = self.column_ranges[column][1] - self.column_ranges[column][0]
                delta = (random.random()) * self.step * 0.5 * interval_size
                if sign == '0':
                    delta = -delta
                value = value + delta
                flag = (value > self.column_ranges[column][1])
                flag2 = (value < self.column_ranges[column][0])
                if flag.bool() or flag2.bool():
                    value -= 2 * delta
                test_instance_mod[column] = value
            prob_mod = self.black_box_model.predict_proba(test_instance_mod)
            prob_prev = self.black_box_model.predict_proba(test_instance)
            slope = prob_mod[0][predict_class][0] - prob_prev[0][predict_class][0]
            if slope < 0:
                self.slope[abs(slope)] = test_instance_mod

    def simulated_annealing_of_one_step(self, directions, test_instance, predict_class):
        threads = []
        self.slope = {}
        prob_original = self.black_box_model.predict_proba(test_instance)
        for direction in directions:
            test_instance_mod = test_instance.copy()
            t = threading.Thread(target=self.computing_one_direction,
                                 args=(direction, test_instance_mod, predict_class))
            threads.append(t)
        for t in threads:
            t.start()
        if len(self.slope) != 0:
            max_slope = max(self.slope)
            return self.slope[max_slope]
        else:
            test_instance_rand = test_instance.copy()
            for column in self.column_names:
                value = test_instance_rand[column]
                delta = (random.random() - 0.5) * self.step * self.jump_rate
                value = value + delta
                flag = (value > self.column_ranges[column][1])
                flag2 = (value < self.column_ranges[column][0])
                if flag.bool() or flag2.bool():
                    value -= 2 * delta
                test_instance_rand[column] = value
            prob_modified = self.black_box_model.predict_proba(test_instance_rand)
            dE = prob_modified[0][predict_class] - prob_original[0][predict_class]
            if judge(dE[0], self.tmp):
                test_instance = test_instance_rand
            return test_instance

    def simulated_annealing_with_all(self, test_instance):
        prev = time.time()
        original_class = self.black_box_model.predict(test_instance)
        directions = generate_directions(self.column_names)
        while self.tmp > self.tmp_min:
            test_instance = self.simulated_annealing_of_one_step(directions, test_instance, original_class)
            self.tmp = self.tmp * self.alpha
            updated_class = self.black_box_model.predict(test_instance)
            if updated_class != original_class:
                #print("yes")
                self.finish = True
                break
        curr = time.time()
        self.time = curr - prev
        if not self.finish:
            test_instance = None
        return test_instance


def generate_directions(column_names):
    directions = []
    for i in range(2 ** len(column_names)):
        direction = []
        for k in bin(i)[2:].zfill(len(column_names)):
            direction.append(k)
        directions.append(direction)
    return directions


def judge(dE, t):
    if dE < 0:
        return True
    else:
        p = math.exp(-(dE / t))
        if p > random.random():
            return True
        else:
            return False
