# Final version: dynamic threshold (theta) and interval-based group smoothing
import sys
import copy
import sys
import os
import pickle
sys.path.insert(0,os.path.dirname(os.getcwd()))

import numpy as np
import math
from treelib import Tree, Node
import os
from func_module import freqoracle

# calculate theta
def calculate_oue_variance(epsilon, users_per_layer):
    """
    Computes the variance of OUE mechanism for binary input.
    """
    return 4 * math.exp(epsilon) / (users_per_layer * (math.exp(epsilon) - 1)**2)

def theta_calculation(h, n, epsilon, total_user_count):
    """
    Calculate the optimal threshold theta based on theoretical analysis.

    Parameters:
    - h: int, total height of the tree
    - n: int, depth of the node where no further splitting is desired
    - epsilon: float, privacy budget for the mechanism
    - total_user_count: int, total number of users contributing to the tree

    Returns:
    - theta: float, splitting threshold
    """
    users_per_layer = total_user_count / h
    var = calculate_oue_variance(epsilon, users_per_layer)

    numerator = (h - n - 1/3) * var
    denominator = (h - n + 1/3)

    theta = math.sqrt(numerator / denominator)
    return theta

# construct sub-domain partition vectors
def construct_translation_vector(domain_size, branch):
    translation_vector = []
    for i in range(branch):
        translation_vector.append(np.array(
            [i * domain_size // branch, i * domain_size // branch]))  # 向下取整
    return translation_vector

# remove duplicated sub-domain partition vectors
def duplicate_remove(list1):
    list2 = []
    for li1 in list1:
        Flag1 = True

        for li2 in list2:
            if (li1 == li2).all():
                Flag1 = False
                break

        if Flag1 == True:
            list2.append(li1)
    return list2

def user_record_partition(time_index, intermediate_tree_height, domain_size, data_path,start_time, end_time):
    data_name = f"{data_path}/t{start_time+time_index}.txt"
    dataset = np.loadtxt(data_name)
    dataset = np.round(dataset).astype(np.int32)

    user_sample_id = np.random.randint(0, intermediate_tree_height, len(dataset)).reshape(len(dataset), 1)
    user_histogram = np.zeros((intermediate_tree_height, domain_size), dtype=np.int32)
    for k, item in enumerate(dataset):
        user_histogram[user_sample_id[k], item] += 1
    return user_histogram

def complete_tree_update(intermediate_tree, tree_height, branch, translation_vector, layer_index):
    for node in intermediate_tree.leaves():
        TempItem0 = np.zeros(node.data.interval.shape)
        for j in range(0, len(node.data.interval), 2):

            if node.data.interval[j + 1] - node.data.interval[j] > 1:
                TempItem0[j] = node.data.interval[j]
                TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch + \
                                   node.data.interval[j]
            else:
                TempItem0[j] = node.data.interval[j]
                TempItem0[j + 1] = node.data.interval[j + 1]

        for item1 in translation_vector:
            node_name = f"{tree_height}-{layer_index}"
            node_frequency = 0
            node_divide_flag = True
            node_count = 0
            node_interval = TempItem0 + item1
            intermediate_tree.create_node(node_name, node_name, parent=node.identifier,
                                   data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))
            layer_index += 1

def dynamic_tree_construction(intermediate_tree, intermediate_tree_height, branch, translation_vector, user_dataset_partition,epsilon, user_num):
    tree_height = -1
    while tree_height < intermediate_tree_height-1:
        tree_height += 1
        theta = theta_calculation(intermediate_tree_height, tree_height, epsilon, user_num)
        dynamic_tree_update(intermediate_tree, tree_height, theta, branch, translation_vector)
        translation_vector[:] = translation_vector[:] // np.array([branch, branch])
        translation_vector = duplicate_remove(translation_vector)
        node_frequency_aggregation(intermediate_tree, user_dataset_partition[tree_height], epsilon)

def dynamic_tree_update(intermediate_tree, tree_height, theta, branch, translation_vector):
    for node in intermediate_tree.leaves():
        if not node.data.divide_flag:
            continue

        elif (tree_height > 0 and node.data.divide_flag) and (node.data.frequency < theta):
            node.data.divide_flag = False
            continue

        else:
            TempItem0 = np.zeros(node.data.interval.shape)
            for j in range(0, len(node.data.interval), 2):
                if node.data.interval[j + 1] - node.data.interval[j] > 1:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = (node.data.interval[j + 1] - node.data.interval[j]) // branch + \
                                       node.data.interval[j]
                else:
                    TempItem0[j] = node.data.interval[j]
                    TempItem0[j + 1] = node.data.interval[j + 1]

            for idx, item1 in enumerate(translation_vector):
                if node.tag == 'Root':
                    parent_index = 0
                    current_level = -1
                else:
                    tag_parts = node.tag.split('-')
                    current_level = int(tag_parts[0])
                    parent_index = int(tag_parts[1])
                node_index = parent_index * branch + idx
                node_name = f"{tree_height}-{node_index}"
                node_frequency = 0
                node_divide_flag = True
                node_count = 0
                node_interval = TempItem0 + item1
                intermediate_tree.create_node(node_name, node_name, parent=node.identifier,
                                       data=Nodex(node_frequency, node_divide_flag, node_count, node_interval))


def complete_tree_construction(intermediate_tree, intermediate_tree_height, branch, translation_vector, user_dataset_partition,epsilon):
    tree_height = 0
    while tree_height < intermediate_tree_height:
        layer_index = 0
        complete_tree_update(intermediate_tree, tree_height, branch, translation_vector, layer_index)
        translation_vector[:] = translation_vector[:] // np.array([branch, branch])
        translation_vector = duplicate_remove(translation_vector)
        node_frequency_aggregation(intermediate_tree, user_dataset_partition[tree_height], epsilon)
        tree_height += 1

def node_frequency_aggregation(intermediate_tree, user_dataset, epsilon):
    # estimate the frequency values, and update the frequency values of the nodes
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))

    user_record_list = []
    for node in intermediate_tree.leaves():
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        user_record_list.append(user_dataset[d1_left:d1_right].sum())

    noise_vector = freqoracle.OUE_Noise(epsilon, np.array(user_record_list, np.int32), sum(user_record_list))
    noisy_frequency = freqoracle.Norm_Sub(noise_vector, len(noise_vector), sum(user_record_list), p, q)

    for i, node in enumerate(intermediate_tree.leaves()):
        if node.data.count == 0:
            node.data.frequency = noisy_frequency[i]
            node.data.count += 1
        else:
            node.data.frequency = ((node.data.count * node.data.frequency) + noisy_frequency[i]) / (node.data.count + 1) #需要和之前的frequency做平均，即 （count*frequency+now_frequency）/（count+1）
            node.data.count += 1


class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval, sum=0):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval
        self.sum = sum

# Calculate err_a,t
def cal_err_a(last_tree, now_tree, var):
    err_a_t = 0.0
    nodes = [node for node in now_tree.all_nodes()]
    num_nodes = len(nodes)

    for node in nodes:
        last_node = last_tree.get_node(node.identifier)
        if last_node is not None:
            diff = node.data.frequency - last_node.data.frequency
            err_a_t += diff ** 2

    err_a_t -= num_nodes * var

    if num_nodes > 0:
        err_a_t = err_a_t / num_nodes

    return err_a_t


def get_var(epsilon, user_num ):
    return 4 * math.exp(epsilon) / (user_num * (math.exp(epsilon) - 1) ** 2)

# Optimal Privacy Budget Allocation based on Section 4.3
def get_optimal_privacy_budget(err_a_list, epsilon, w,user_num,height):
    min_total_err = float('inf')
    optimal_k = 0

    sorted_err_indices = sorted(range(len(err_a_list)), key=lambda x: err_a_list[x], reverse=True)

    for k in range(w + 1):
        total_err = 0
        if k > 0:
            epsilon_k = epsilon / k
            var_k = get_var(epsilon_k,user_num/height)
            for i in range(k):
                total_err += var_k

        for i in range(k, w):
            total_err += err_a_list[sorted_err_indices[i]]

        if total_err < min_total_err:
            min_total_err = total_err
            optimal_k = k

    optimal_budget_allocations = [0] * w
    if optimal_k > 0:
        epsilon_k = epsilon / optimal_k
        for i in range(optimal_k):
            optimal_budget_allocations[sorted_err_indices[i]] = epsilon_k
    return optimal_budget_allocations[-1]


def convert_to_complete_tree(dynamic_tree, branch, intermediate_tree_height):
    complete_tree = Tree()
    root_node = dynamic_tree.get_node("root")
    complete_tree.create_node(tag="root", identifier="root", data=root_node.data)

    queue = [("root", 0, 0)]

    while queue:
        node_id, current_height, current_index = queue.pop(0)
        complete_node = complete_tree.get_node(node_id)

        if current_height >= intermediate_tree_height:
            continue

        dynamic_node = dynamic_tree.get_node(node_id)

        if dynamic_node and dynamic_tree.children(dynamic_node.identifier):
            for i, child in enumerate(dynamic_tree.children(dynamic_node.identifier)):
                child_identifier = f"{current_height}-{current_index * branch + i}"
                complete_tree.create_node(
                    tag=child_identifier,
                    identifier=child_identifier,
                    parent=node_id,
                    data=child.data
                )
                queue.append((child_identifier, current_height + 1, current_index * branch + i))
        else:
            parent_interval = complete_node.data.interval
            interval_length = (parent_interval[1] - parent_interval[0]) / branch

            for i in range(branch):
                child_identifier = f"{current_height}-{current_index * branch + i}"
                child_interval = np.array([
                    parent_interval[0] + i * interval_length,
                    parent_interval[0] + (i + 1) * interval_length
                ])
                child_frequency = complete_node.data.frequency / branch

                new_node_data = Nodex(
                    frequency=child_frequency,
                    divide_flag=False,
                    count=complete_node.data.count,
                    interval=child_interval,
                )
                complete_tree.create_node(
                    tag=child_identifier,
                    identifier=child_identifier,
                    parent=node_id,
                    data=new_node_data,
                )
                queue.append((child_identifier, current_height + 1, current_index * branch + i))

    return complete_tree

def smooth_tree(smooth_var_list, user_num_list, publish_tree_list, smooth_tree_list, timestap, IQ_tree_group, varience_of_OUE,complete_intermediate_tree ):
    for node in complete_intermediate_tree.all_nodes_itr():
        if node.identifier == 'root':
            continue

        last_group = IQ_tree_group.get(node.identifier).pop()
        n = len(last_group) + 1
        # Current group deviation: average absolute deviation from group mean
        mean = node.data.frequency
        now_group_frequency_list = [node.data.frequency]
        for element in last_group:
            mean += publish_tree_list[element].get_node(node.identifier).data.frequency
            now_group_frequency_list.append( publish_tree_list[element].get_node(node.identifier).data.frequency )
        mean = mean / (len(last_group)+1)
        dev = abs( node.data.frequency - mean )
        for element in last_group:
            dev += abs( publish_tree_list[element].get_node(node.identifier).data.frequency - mean )
        dev = dev / (len(last_group)+1)

        # deviation = τ
        var_xn = get_var(smooth_var_list[timestap],user_num_list[timestap])
        var_sum = var_xn
        for i in last_group:
            var_sum += get_var(smooth_var_list[i],user_num_list[timestap])
        deviation = var_xn - var_sum / (n * n)

        # If within deviation threshold, group values and apply smoothing
        if dev <= deviation :
            last_group.append(timestap)
            IQ_tree_group.get(node.identifier).append(last_group)
            now_group_value = get_median( now_group_frequency_list )
            for i in last_group:
                smooth_tree_list[i].get_node(node.identifier).data.frequency = now_group_value
        else:
            IQ_tree_group.get(node.identifier).append(last_group)
            IQ_tree_group.get(node.identifier).append([timestap])

def smooth( complete_tree_list, timestap, IQ_tree_group, deviation, complete_intermediate_tree, now_group_value, node, last_group ):
    for i in last_group:
        complete_tree_list[i].get_node(node.identifier).data.frequency = now_group_value

def get_median( now_group_frequency_list ):
    now_group_frequency_list.sort()
    i = len(now_group_frequency_list)
    if i % 2 == 0:
        i = i // 2
        return (now_group_frequency_list[i] + now_group_frequency_list[i-1])/2
    else:
        i = i // 2
        return now_group_frequency_list[i]

def main_func( domain_size, branch, intermediate_tree_height, time_length, data_path,IQ_path,start_time,end_time,w_list, epsilon_all_list):
    for w_item in w_list:
        w = w_item
        for epsilon_item in epsilon_all_list:
            epsilon_all = epsilon_item
            time_index = 0
            # Each mechanism at each timestamp gets a portion of the total privacy budget
            epsilon_m2_all = epsilon_all / 2  # Half the budget goes to M1 and the other half to M2
            epsilon_for_each_m1 = epsilon_for_each_m = epsilon_all / w / 2
            epsilon_for_each_time  = epsilon_all / w
            last_var = 0
            print(f"epsilon_for_each_m:{epsilon_for_each_m}")
            while time_index < time_length:
                print(f'Processing timestamp{time_index}')
                if time_index < w-1 :
                    epsilon_for_each_m = epsilon_all / w

                # M1 - Estimating approximate error (err_a)
                user_dataset_partition = user_record_partition(time_index, intermediate_tree_height, domain_size, data_path,start_time, end_time)#得到真实数据的直方图
                user_num_temp =  np.sum(user_dataset_partition)
                user_num =  np.sum(user_dataset_partition)
                varience_of_OUE = get_var(epsilon_for_each_m, user_num/intermediate_tree_height)
                last_var = varience_of_OUE
                # M1 - Build the complete tree
                intermediate_tree = Tree()
                intermediate_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size])))
                translation_vector = construct_translation_vector(domain_size, branch)

                if time_index == 0:
                    complete_tree_construction(intermediate_tree, intermediate_tree_height, branch, translation_vector,
                                               user_dataset_partition, epsilon_for_each_m)
                    publish_tree_list = [copy.deepcopy(intermediate_tree)]
                    intermediate_tree.get_node('root').data.frequency = 1
                    smooth_tree_list= [copy.deepcopy(intermediate_tree)]
                    smooth_var_list = [epsilon_for_each_m]
                    epsilon_m2_list = [epsilon_for_each_m/2]
                    err_a_list =[0.0]
                    user_num_list = [user_num_temp]
                    # Initialize IQ_tree_group for each node to track temporal grouping
                    IQ_tree_group = {}
                    for node in intermediate_tree.all_nodes_itr():
                        if node.identifier == 'Root':
                            continue
                        IQ_tree_group[node.identifier] = [[time_index]]
                    time_index += 1
                    continue

                if time_index < w-1 :
                    complete_tree_construction(intermediate_tree, intermediate_tree_height, branch, translation_vector,
                                               user_dataset_partition, epsilon_for_each_m)
                    err_a = cal_err_a(publish_tree_list[-1], intermediate_tree, varience_of_OUE)
                    err_a_list.append(max(err_a, 0.0))
                    epsilon_m2_list.append(epsilon_for_each_m/2)
                    publish_tree_list.append(copy.deepcopy(intermediate_tree))
                    intermediate_tree.get_node('root').data.frequency = 1
                    smooth_tree_list.append(copy.deepcopy(intermediate_tree))
                    smooth_var_list.append(epsilon_for_each_m)
                    user_num_list.append(user_num_temp)
                    for node in intermediate_tree.all_nodes_itr():
                        if node.identifier == 'Root':
                            continue
                        IQ_tree_group.get(node.identifier).append([time_index])
                    time_index += 1
                    continue

                #timpstamp>=w
                # Build a complete tree for M2 using the allocated epsilon
                complete_tree_construction(intermediate_tree, intermediate_tree_height, branch, translation_vector,
                                               user_dataset_partition, epsilon_for_each_m/2)
                err_a = cal_err_a(smooth_tree_list[-1], intermediate_tree, varience_of_OUE)
                err_a_list.append(err_a)

                print(f"remian epsilon：{epsilon_m2_all-sum(epsilon_m2_list[-(w):])}")
                print(epsilon_m2_list)
                print(err_a_list)


                #M2 - Optimal Privacy Budget Allocation
                optimal_privacy_budget = get_optimal_privacy_budget(err_a_list[-w:], epsilon_m2_all, w, user_num,
                                                                    intermediate_tree_height)
                used_budget = sum(epsilon_m2_list[-(w - 1):])
                can_use_privacy_budget = min(epsilon_m2_all - used_budget, optimal_privacy_budget)

                # M3 - adaptive tree publication
                if can_use_privacy_budget == 0:
                        publish_tree_list.append(copy.deepcopy(publish_tree_list[-1]))
                        smooth_tree_list.append(copy.deepcopy(intermediate_tree))
                        smooth_var_list.append(epsilon_for_each_m / 2)
                        varience_of_OUE = last_var
                        complete_tree = copy.deepcopy(publish_tree_list[-1])
                        user_num_list.append(copy.deepcopy(user_num_list[-1]))
                        epsilon_m2_list.append(0.0)

                else:
                    # 如果有预算则重新发布
                    varience_of_OUE = get_var(can_use_privacy_budget, user_num / intermediate_tree_height)
                    last_var = varience_of_OUE
                    # initialize the tree structure, set the root node
                    dynamic_tree = Tree()
                    dynamic_tree.create_node('Root', 'root', data=Nodex(1, True, 1, np.array([0, domain_size])))
                    translation_vector = construct_translation_vector(domain_size, branch)
                    dynamic_tree_construction(dynamic_tree, intermediate_tree_height, branch, translation_vector,
                                               user_dataset_partition, can_use_privacy_budget, user_num)
                    complete_tree = convert_to_complete_tree(dynamic_tree, branch, intermediate_tree_height)
                    publish_tree_list.append(copy.deepcopy(complete_tree))
                    complete_tree.get_node('root').data.frequency = 1
                    smooth_tree_list.append(copy.deepcopy(complete_tree))
                    smooth_var_list.append(can_use_privacy_budget)
                    user_num_list.append(user_num_temp)
                    epsilon_m2_list.append(can_use_privacy_budget)
                #Grouping and Smoothing
                smooth_tree( smooth_var_list,user_num_list, publish_tree_list, smooth_tree_list, time_index, IQ_tree_group, varience_of_OUE, complete_tree)
                time_index += 1

            # Save Results
            result_dir = os.path.join(IQ_path, 'IQtree_result')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            with open(os.path.join(result_dir, f'w_{w}_epsilon_{epsilon_for_each_time}_start_{start_time}_end_{end_time}_smooth_tree_list.pkl'),
                      'wb') as f:
                pickle.dump(smooth_tree_list, f)
            with open(os.path.join(result_dir
                    ,f'w_{w}_epsilon_{epsilon_for_each_time}_start_{start_time}_end_{end_time}_user_num.pkl'), 'wb') as f:
                pickle.dump(user_num_list, f)



if __name__ == "__main__":
    start_time = 0
    end_time = 138
    domain_size = 2048
    time_length = end_time - start_time + 1
    w_list = [ 10,20,30,40,50]
    epsilon_all_list = [0.5,1,1.5,2,5]

    data_path = './data_process/dataset/loan/installment'
    IQ_path = './data_process/dataset/loan/installment_answer/IQ_tree/'

    branch = 2

    intermediate_tree_height = int(math.log(domain_size, branch))


    # running
    main_func( domain_size, branch, intermediate_tree_height, time_length, data_path,IQ_path, start_time, end_time,w_list, epsilon_all_list)
