# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:56:08 2020

@author: Kevin
"""

import os
import math
import random
from networkx import grid_graph
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
#  import time as t
#import user-defined functions
from functions import define_grid_graph,define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, calculate_Vsource, calculate_Isource, update_edge_weigths
# from file_sel import file_sel

def dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps, idle_timesteps, digit_list, input_digit, pulse_amplitude):
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]
    train_pulse = [[[]for i in range(0, digit_cols)] for i in range(0, digit_rows)]

    bit_0 = [0]*(idle_timesteps + pulse_timesteps + idle_timesteps)
    bit_1 = [0]*idle_timesteps + [1]*pulse_timesteps + [0]*idle_timesteps
    
    for i in range(0, digit_rows):
        for j in range(0, digit_cols):
             digit_cell = digit_list[input_digit][i][j]
             cell_value = int(digit_cell == 1)
             train_pulse[i][j] = bit_0*(1-cell_value)+bit_1*(cell_value)
        train_pulse[i] = [element for item in train_pulse[i] for element in item] # make it a single list
    
    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if train_pulse[r][t] == 0:
                Vin_list_write[t] = list(Vin_list_write[t])+['f']
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[np.multiply(pulse_amplitude,(train_pulse[r][t]))]
    
    
    return train_pulse, Vin_list_write

def insert_R_to_graph(G, R, src_nodes, new_nodes, gnd):
    
    new_nodes_filtered = []
    gnd_filtered = []
    for node in new_nodes:
        if not G.has_node(node):
            new_nodes_filtered.append(node)
    for node in gnd:
        if not G.has_node(gnd):
            gnd_filtered.append(node)
    
    if len(src_nodes) != len(new_nodes):
        print("[Warning] New nodes lenght filtered is different from sources")
    G.add_nodes_from(new_nodes_filtered)
    G.add_nodes_from(gnd_filtered)
    
    for i in range(len(new_nodes_filtered)):    
        G.add_edge(new_nodes_filtered[i], src_nodes[i])
        
    for u in new_nodes_filtered:
        for v in src_nodes:  
            if G.has_edge(u, v):
                idx = new_nodes_filtered.index(u)
                G[u][v]['Y'] = 1/R[idx]
    return G

def remove_R_from_graph(G, src_nodes, new_nodes, gnd):
    
    G.remove_nodes_from(new_nodes)
    G.remove_nodes_from(gnd)
    
    for u in new_nodes:
        for v in src_nodes:  
            if G.has_edge(u, v):
                G.remove_edge(u, v)
    return G
