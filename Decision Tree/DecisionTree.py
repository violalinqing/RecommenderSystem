from __future__ import division
import math
import sys
import ast
import csv
from collections import Counter

class data():
    def __init__(self, classifier):
        self.examples = []
        self.attributes = []
        self.attr_types = []
        self.classifier = classifier
        self.class_index = None



def read_data(dataset, datafile):

    f = open(datafile)
    original_file = f.read()

    line_count = 0
    for rows in original_file.splitlines():
        if line_count == 0:
            dataset.attributes = rows.split(',')
        else:
            dataset.examples.append(rows.split(','))
        line_count += 1

    dataset.attr_types = ['true','true','false','true','true','true','false','false','true','true','false','true','true','false']

def  preprocess(dataset):
    class_values = []
    for example in dataset.examples:
        class_values.append(example[dataset.class_index])

    class_mode = Counter(class_values)
    class_mode = class_mode.most_common(1)[0][0]

    for index in range(len(dataset.attributes)):
        values_0class = []
        for example in dataset.examples:
            if example[dataset.class_index] == '0':
                values_0class.append(example[index])
        values_1class = []
        for example in dataset.examples:
            if example[dataset.class_index] == '1':
                values_1class.append(example[index])
        values = Counter(values_0class)

        mode0 = values.most_common(1)[0][0]

        if mode0 == '?':
            mode0 = values.most_common(2)[1][0]

        values = Counter(values_1class)
        mode1 = values.most_common(1)[0][0]

        if mode1 == '?':
            mode1 = values.most_common(2)[1][0]

        mode_01 = [mode0, mode1]

        attr_modes = [0] * len(dataset.attributes)

        attr_modes[index] = mode_01

        for example in dataset.examples:
            if (example[index] == '?'):
                if (example[dataset.class_index] == '0'):
                    example[index] = attr_modes[index][0]
                elif (example[dataset.class_index] == '1'):
                    example[index] = attr_modes[index][1]
                else:
                    example[index] = class_mode

        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                if dataset.attributes[x] == 'True':
                    example[x] = float(example[x])

class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None

def build_tree(dataset, parent_node, classifier):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if node.parent == None:
        node.height = 0
    else:
        node.height = node.parent.height + 1

    num_one = count_one(dataset, classifier)
    if(len(dataset.examples)== num_one):
        node.is_leaf = True
        node.classification = 1
        return node
    elif(num_one == 0):
        node.is_leaf = True
        node.classification = 0
        return node
    else:
        node.is_leaf = False

    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = None
    min_gain = 0.01
    entropy = calc_entropy(dataset, classifier)
    for index in range(len(dataset.attributes)):
        if dataset.attributes[index] != classifier:
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[index] for example in dataset.examples]
            attr_value_list = list(set(attr_value_list))
            if (len(attr_value_list) > 100):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total / 10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x * ten_percentile])
                attr_value_list = new_list
            for val in attr_value_list:
                local_gain = calc_gain(dataset, entropy, val, index)
                if local_gain > local_max_gain:
                    local_max_gain = local_gain
                    local_split_val = val
            if local_max_gain > max_gain:
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = index
    if (max_gain <= min_gain or node.height > 20):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)
        return node
    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val

    upper_dataset = data(classifier)
    lower_dataset = data(classifier)
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    upper_dataset.attr_types = dataset.attr_types
    lower_dataset.attr_types = dataset.attr_types
    for example in dataset.examples:
        if example[attr_to_split] >= split_val:
            upper_dataset.examples.append(example)
        else:
            lower_dataset.examples.append(example)
    node.upper_child = build_tree(upper_dataset, node, classifier)
    node.lower_child = build_tree(lower_dataset, node, classifier)

    return node


def count_one(dataset, classifier):
    count = 0
    for a in range(len(dataset.attributes)):
        if dataset.attributes[a] == classifier:
            class_index = a
        else:
            class_index = len(dataset.attributes) - 1
    for example in dataset.examples:
        if example[class_index] == '1':
            count += 1

    return  count

def calc_entropy(dataset,classifier):
    num_one = count_one(dataset, classifier)
    total_examples = len(dataset.examples)
    p = num_one/total_examples
    entropy = 0
    if(p != 0):
        entropy += p * math.log(p,2)
    p = (total_examples-num_one)/total_examples
    if(p != 0):
        entropy += p * math.log(p,2)
    entropy = -entropy
    return entropy

def calc_gain(dataset, entropy, val, attr_index):
    classifier = dataset.attributes[attr_index]
    total_examples = len(dataset.examples)
    gain_upper_dataset = data(classifier)
    gain_lower_dataset = data(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attr_types = dataset.attr_types
    gain_lower_dataset.attr_types = dataset.attr_types

    for example in dataset.examples:
        if example[attr_index] >= val:
            gain_upper_dataset.examples.append(example)
        else:
            gain_lower_dataset.examples.append(example)

    if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0):
        return -1

    local_entropy = 0
    local_entropy += calc_entropy(gain_upper_dataset, classifier) * len(gain_upper_dataset.examples)/total_examples
    local_entropy += calc_entropy(gain_lower_dataset, classifier) * len(gain_lower_dataset.examples)/total_examples
    return entropy - local_entropy

def classify_leaf(dataset, classifier):
    num_one = count_one(dataset, classifier)
    total_examples = len(dataset.examples)
    num_zeros = total_examples - num_one
    if(num_one >= num_zeros):
        return 1
    else:
        return 0

def validTree(node,validset):
    count = 0
    for example in validset.examples:
        count += valid_example(example,node)
    total = len(validset.examples)
    return count/total

def valid_example(example, node):
    if node.is_leaf == True:
        if node.classification == int(example[-1]):
            return 1
        else:
            return 0
    val = example[node.attr_split_index]
    if val >= node.attr_split_value:
        return valid_example(example, node.upper_child)
    else:
        return valid_example(example, node.lower_child)

if __name__ == "__main__":
    dataset = data("")
    read_data(dataset,'btrain.csv')
    classifier = dataset.attributes[-1]
    dataset.classifier = classifier
    dataset.class_index = range(len(dataset.attributes))[-1]

    preprocess(dataset)

    node = build_tree(dataset, None, classifier)

    validset = data(classifier)
    read_data(validset, 'bvalidate.csv')
    validset.class_index = range(len(validset.attributes))[-1]
    preprocess(validset)
    best_score = validTree(node,validset)
    print "pre-pruning accuracy :" + str(100*best_score)
