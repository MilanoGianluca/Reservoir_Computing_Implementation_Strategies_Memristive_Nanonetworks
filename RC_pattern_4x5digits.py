# -*- coding: utf-8 -*-
import os
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import user-defined functions
from functions import define_grid_graph_2,initialize_graph_attributes, mod_voltage_node_analysis, update_edge_weigths, calculate_Isource
from functions_reservoir import dataset_to_pulse, insert_R_to_graph, remove_R_from_graph

#%%
def plot_raw(H):
    fig3, ax = plt.subplots(figsize=(10, 10))
    plt.cla()
    pos=nx.get_node_attributes(H,'pos')
    
    nx.draw_networkx(H, pos,
                     node_size=60,
                      node_color='red',
                      with_labels=False   #Set TRUE to see node numbers
                      )
    
#%%
def plot(H, title):
    if read_config == "float":
        remove_R_from_graph(H, src_read, new_nodes_read, gnd_read)
    fig3, ax = plt.subplots(figsize=(10, 10))
    plt.cla()
    pos=nx.get_node_attributes(H,'pos')
    
    if case == "config_LR":
        node_color=[H.nodes[n]['V']*int(n not in src and n not in gnd_tmp)+5*int(n in src or n in gnd_tmp) for n in H.nodes()]
        node_size=[60+120*int(n in src or n in gnd_tmp) for n in H.nodes()]
    else:
        node_color=[H.nodes[n]['V']*int(n not in src_read and n not in gnd_read)+5*int(n in src_read or n in gnd_read) for n in H.nodes()]
        node_size=[60+120*int(n in src_read or n in gnd_read) for n in H.nodes()]

    nx.draw_networkx(H, pos,
                      #NODES
                      node_size=node_size,
                      node_color=node_color,
                      cmap=plt.cm.Blues,
                      vmin=0,
                      vmax=pulse_amplitude+V_read,
                      #EDGES
                      width=4,
                      edge_color=[H[u][v]['Y'] for u,v in H.edges()],
                      edge_cmap=plt.cm.Reds,
                      edge_vmin=g_min,
                      edge_vmax=g_max,
                      with_labels=False,   #Set TRUE to see node numbers
                      font_size=6,)
    
    ax.set_title(title)


#%% DIRECTORY DEFINITION

case = "config_LR" # available ---> "config_NESWO" , "config_LR"

out_folder_train = r'./output_digits/case_{}/train/'.format(case) 
if not os.path.exists(out_folder_train):
    os.makedirs(out_folder_train)  
    
#%% DATASET LOAD & DISPLAY

# to plot train conductance histogram in subplot grid: n_train x m_train
n_hist = 2 
m_hist = 5

digit_rows = 5
digit_cols = 4

color_n = ['--*r','--*g','--*b', '--*m', '--*c']

t_int_list = [0, 1, 2, 3, 4, 5] # int point state to save and plot evolution
t_int_read = 4 # timestep index for network output reading for reservoir computing task

########### Importing train digits ###########
file_to_train = 'digits'
file_train = file_to_train+'.txt'
file_train_class = file_to_train+'_class.txt'
digit_train = np.loadtxt('./dataset/'+file_train)
digit_train_class = np.loadtxt('./dataset/'+file_train_class)
total_rows_train = int(len(digit_train))
num_digits_train = len(digit_train_class)
digit_list_train = [[] for i in range(0, num_digits_train)]
# plt.figure()
for i in range(0, num_digits_train):
    digit_list_train[i] = digit_train[digit_rows*i:digit_rows*(i+1)][:]
    # plt.subplot(2, 5, i+1)
    # plt.imshow(digit_list_train[i], cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.colorbar()

#%% NETWORK DEFINITION

kp0 = 2.555173332603108574e-06
kd0 = 6.488388862524891465e+01
eta_p = 3.492155165334443012e+01
eta_d = 5.590601016803570467e+00
g_min = 1.014708121672117710e-03
g_max = 2.723493729125820492e-03
g0 = 5.602507668026886038e-04
g0 = g_min

xdim = 21  # graph dimension
ydim = 21
frame = 2 #number of frame rows/columns
seed = 3

left_pads = [(xdim-1)*(frame+1)-2-3*i for i in range(0, 5)] # (from top to bottom)
right_pads = [(xdim*xdim-1)-xdim*frame-frame-2-3*i for i in range(0, 5)] # (from top to bottom)
top_pads = [(xdim-1)+xdim*(frame+2)-frame+xdim*3*i for i in range(0, 5)] # (from left to right)
bottom_pads = [xdim*(frame+2)+frame+xdim*3*i for i in range(0, 5)] # (from left to right)

pad_N = top_pads[2]  
pad_E = right_pads[2]
pad_S = bottom_pads[2]  
pad_W = left_pads[2]
pad_O = (xdim*ydim-1)/2

if case == "config_NESWO":
    write_config = "float"
    read_config = "float"
    
    # WRITE
    src = [pad_N, pad_E, pad_S, pad_W, pad_O]
    new_nodes = [xdim*ydim+nn for nn in range(len(src))]
    gnd = [new_nodes[-1]+1]
    
    # READ
    src_read = src
    bias_nodes = [pad_O]
    read_nodes = [pad_N, pad_E, pad_S, pad_W]
    new_nodes_read = new_nodes
    gnd_read = gnd         
    neuron_name = ["N","E","S","W"]
    
elif case == "config_LR":
    write_config = "gnd"
    read_config = "float"
    
    # WRITE
    src = left_pads
    gnd_tmp = right_pads
    gnd = [xdim*ydim+nn for nn in range(len(gnd_tmp))]
    new_nodes = src
    
    # READ
    bias_nodes = [left_pads[2]]
    read_nodes = [right_pads[0], right_pads[1], right_pads[3], right_pads[4]]
    src_read = bias_nodes + read_nodes
    new_nodes_read = gnd
    gnd_read = [new_nodes_read[-1]+1] 
    neuron_name = ["R1","R2","R4","R5"]
    
else:
    raise Exception("Configuration not valid")
#%% NETWORK INPUTs

####### CUSTOMIZE YOUR PULSE SHAPE ###############
R_read = [82]*(len(new_nodes))
V_read = 100e-3
pulse_amplitude = 5 # Volts

delta_pot = 250e-6 # distance for potentiation points
delta_dep = 250e-6 # distance for depression points
delta_read = delta_dep
delta = 250e-6 # distance for transition from low to high signal and viceversa 

pulse_time = 10e-3-delta
idle_time = 5e-4-delta 
read_time = 5.5e-3 #  seconds between write and read

###################################################
read_timesteps = int(read_time/delta_read)-1
pulse_timesteps = int(pulse_time/delta_pot) 
idle_timesteps = int(idle_time/delta_dep)

one_pulse = 2*idle_timesteps+pulse_timesteps+3 # points of a single pulse

time_write_1 = np.linspace(0, idle_time, idle_timesteps+1)
time_write_2 = np.linspace(idle_time+delta, idle_time+delta+pulse_time, pulse_timesteps+1)
time_write_3 = np. linspace(idle_time+pulse_time+2*delta, idle_time+pulse_time+2*delta+idle_time, idle_timesteps+1)
time_write_tot = np.append(np.append(time_write_1, time_write_2), time_write_3)

time_write = time_write_tot

for i in range(1, digit_cols):
    time_write = np.append(time_write, time_write_tot+time_write[-1]+delta_dep)

timesteps_write = len(time_write)

wave_test = [0]*(idle_timesteps+1)+[1]*(pulse_timesteps+1)+[0]*(idle_timesteps+1)
wave_test = wave_test*digit_cols
wave_test = wave_test+[0]*(read_timesteps+1)

int_point = [[] for ip in range(digit_cols+2)]

int_point[0] = 0
for ip in range(1, digit_cols+2):
    int_point[ip] = one_pulse*ip+idle_timesteps 

int_point[-1] = timesteps_write+read_timesteps
field_save_point = [0, 3, 47, 91, 135]

time_test = [i*delta_pot for i in range(len(wave_test))]
plt.figure()
plt.plot(time_test, wave_test, '*--b')
plt.plot(np.array(time_test)[int_point], [0]*len(int_point), '*r')
plt.plot(np.array(time_test)[field_save_point], [1]*len(field_save_point), '*y')

#%% TRAIN DIGITS

for digit in range(num_digits_train):
        
    G = define_grid_graph_2(xdim, ydim, seed)
    G = initialize_graph_attributes(G, g0) 
    
    input_digit = int(digit_train_class[digit])
    
    H_list_write = [[] for t in range(0, timesteps_write+read_timesteps+1)]
    
    train_pulse, _ = dataset_to_pulse(digit_rows, digit_cols, timesteps_write, pulse_timesteps+1, idle_timesteps+1, digit_list_train, digit, pulse_amplitude)
    
    Vin_list_write = [[] for t in range(0, timesteps_write)]
    
    for t in range(0, timesteps_write):
        for r in range(0, digit_rows):
            if train_pulse[r][t] == 0:
                if write_config == "float":
                    Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*V_read]
                else:
                    Vin_list_write[t] = list(Vin_list_write[t])+['f']
            else:
                Vin_list_write[t] = list(Vin_list_write[t])+[int(src[r] in bias_nodes)*0+np.multiply(pulse_amplitude,(train_pulse[r][t]))]
        
    # WRITE
    if write_config == "float":
        insert_R_to_graph(G, R_read, src, new_nodes, gnd)
    else:
        insert_R_to_graph(G, R_read, gnd_tmp, gnd, gnd)
    
    gnd_filtered = []
    voltage_to_apply = Vin_list_write[0].copy()
    if write_config == "gnd":
        for v in range(len(Vin_list_write[0])):
            if Vin_list_write[0][v] != 'f':                    
                gnd_filtered += [gnd[v]]
    else:
        gnd_filtered = gnd
    if len(gnd_filtered) == 0:
        gnd_filtered = [gnd[0], gnd[1], gnd[3], gnd[4]]
        voltage_to_apply[2] = V_read # tmp
    H_list_write[0] = mod_voltage_node_analysis(G, voltage_to_apply, new_nodes, gnd_filtered)
    if 0 in int_point:
            fname = out_folder_train+'train_'+str(digit)+'_t_'+str(int_point.index(0))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
    for i in range(1, timesteps_write):
            
        delta_t = time_write[i] - time_write[i-1]
        
        if write_config == "float":
            remove_R_from_graph(G, src, new_nodes, gnd)
        else:
            remove_R_from_graph(G, gnd_tmp, gnd, gnd)
        
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = out_folder_train+'train_'+str(digit)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
        
        if i in field_save_point:
            fname = out_folder_train+'train_'+str(digit)+'_t_'+str(field_save_point.index(i))+'_voltages.txt'
            voltage_map = np.rot90(np.reshape(np.array([G.nodes[n]['V'] for n in G.nodes()]), (xdim, ydim)))
            plt.figure()
            plt.imshow(voltage_map, cmap='gray')
            np.savetxt(fname, voltage_map)
             
        # WRITE
        if write_config == "float":
            insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        else:
            insert_R_to_graph(G, R_read, gnd_tmp, gnd, gnd)
    
       
        gnd_filtered = []
        voltage_to_apply = Vin_list_write[i].copy()
        if write_config == "gnd":
            for v in range(len(Vin_list_write[i])):
                if Vin_list_write[i][v] != 'f':                    
                    gnd_filtered += [gnd[v]]
        else:
            gnd_filtered = gnd
        if len(gnd_filtered) == 0:
                gnd_filtered = [gnd[0], gnd[1], gnd[3], gnd[4]]
                voltage_to_apply[2] = V_read # tmp
        
        H_list_write[i] = mod_voltage_node_analysis(G, voltage_to_apply, new_nodes, gnd_filtered)

    for i in range(timesteps_write, timesteps_write+read_timesteps+1):
    
        delta_t = delta_read
        
        if write_config == "float":
            remove_R_from_graph(G, src, new_nodes, gnd) 
        else:
            remove_R_from_graph(G, gnd_tmp, gnd, gnd)
            
        G = update_edge_weigths(G, delta_t, g_min, g_max, kp0, eta_p, kd0, eta_d)                                  #update edges
        
        if i in int_point:
            fname = out_folder_train+'train_'+str(digit)+'_t_'+str(int_point.index(i))+'.txt'
            pickle.dump(G, open(fname, 'wb'))
                
        # WRITE
        if write_config == "float":
            insert_R_to_graph(G, R_read, src, new_nodes, gnd)
        else:
            insert_R_to_graph(G, R_read, gnd_tmp, gnd, gnd)
        
        gnd_filtered = []
        voltage_to_apply = Vin_list_write[-1].copy()
        if write_config == "gnd":
            for v in range(len(Vin_list_write[-1])):
                if Vin_list_write[-1][v] != 'f':                    
                    gnd_filtered += [gnd[v]]
        else:
            gnd_filtered = gnd
        if len(gnd_filtered) == 0:
            gnd_filtered = [gnd[0], gnd[1], gnd[3], gnd[4]]
            voltage_to_apply[2] = V_read # tmp
        H_list_write[i] = mod_voltage_node_analysis(G, voltage_to_apply, new_nodes, gnd_filtered)
    
    print('Train Digit '+str(input_digit)+' completed'+' ('+str(np.round((digit+1)/num_digits_train*100,2))+'%)')

#%% HIST GENERATION TRAIN

V_list_read = []
src_vec = []
for n in range(len(src_read)):
    if src_read[n] in bias_nodes:
        V_list_read += [V_read]
        src_vec += [new_nodes_read[n]]
    elif src_read[n] in read_nodes and read_config == "float":
        V_list_read += [0]
        src_vec += [new_nodes_read[n]]

print('\n')
print("######### Hist Generation Train dataset #########")
hist_train_list = [np.zeros((len(read_nodes)+1,)) for i in range(num_digits_train)]
plt.figure()
H_read_list_train = []
H_read_list_train_digit_9 = []
plt.suptitle("Train")
for digit in range(num_digits_train):
    for t_int in range(len(int_point)):
        hist_train = np.zeros((len(read_nodes)+1,))
    
        fname = out_folder_train+'train_'+str(digit)+'_t_'+str(t_int)+'.txt'
        G_read = pickle.load(open(fname, 'rb'))
    
        if read_config == "float":
            if case == "config_NESWO":
                insert_R_to_graph(G_read, R_read, src_read, new_nodes_read, gnd_read)
                H_read = mod_voltage_node_analysis(G_read, V_list_read, src_vec, gnd_read)
            elif case == "config_LR":
                insert_R_to_graph(G_read, R_read, src_read[1:], new_nodes_read[0:2]+new_nodes_read[3:], gnd_read)
                H_read = mod_voltage_node_analysis(G_read, V_list_read, [src_read[0]]+new_nodes_read[0:2]+new_nodes_read[3:], gnd_read)
        
        if t_int == t_int_read:
            H_read_list_train.append(H_read)
        if digit == 9 and t_int in t_int_list:
            H_read_list_train_digit_9.append(H_read)
        for n in range(len(read_nodes)):
            hist_train[n+1] = H_read.nodes[read_nodes[n]]['V']*(int(read_config == "float")) + calculate_Isource(H_read, read_nodes[n])*(int(read_config == "gnd"))
        
        hist_train[0] = digit_train_class[digit]
        if t_int == t_int_read:
            hist_train_list[digit] = hist_train
        if t_int == 0:
            print(str(digit)+'/'+str(num_digits_train))    
        
        if t_int == t_int_read:            
            plt.subplot(n_hist,m_hist,digit+1)
            plt.bar([i+1 for i in range(len(hist_train[1:]))], hist_train[1:], color='#1f77b4')
            plt.xticks([i+1 for i in range(len(hist_train[1:]))], neuron_name, fontsize = 12)
            plt.xlabel('Neuron', fontsize = 12)
            plt.ylabel('Output [a.u.]', fontsize = 12)
            plt.title('Digit: {}'.format(digit_train_class[digit]), fontsize = 15)
            
        np.savetxt(out_folder_train+'hist_train_digit_{}_t_int_{}.txt'.format(digit, t_int), hist_train)

hist_train_nodes_t_int = np.zeros((len(neuron_name), len(t_int_list)))
plt.figure()
plt.suptitle("Train")
for digit in range(num_digits_train):
    
    for t_int in range(len(t_int_list)):    
        fname = out_folder_train+'train_'+str(digit)+'_t_'+str(t_int_list[t_int])+'.txt'
        G_read = pickle.load(open(fname, 'rb'))

        if read_config == "float":
            if case == "config_NESWO":
                insert_R_to_graph(G_read, R_read, src_read, new_nodes_read, gnd_read)
                H_read = mod_voltage_node_analysis(G_read, V_list_read, src_vec, gnd_read)
            elif case == "config_LR":
                insert_R_to_graph(G_read, R_read, src_read[1:], new_nodes_read[0:2]+new_nodes_read[3:], gnd_read)
                H_read = mod_voltage_node_analysis(G_read, V_list_read, [src_read[0]]+new_nodes_read[0:2]+new_nodes_read[3:], gnd_read)
        for n in range(len(neuron_name)):
            hist_train_nodes_t_int[n][t_int] = H_read.nodes[read_nodes[n]]['V']*(int(read_config == "float")) + calculate_Isource(H_read, read_nodes[n])*(int(read_config == "gnd"))
        
    for n in range(len(neuron_name)):
        plt.subplot(n_hist,m_hist,digit+1)
        plt.plot([i+1 for i in range(len(hist_train_nodes_t_int[n][:]))], hist_train_nodes_t_int[n][:], color_n[n], label=neuron_name[n])
        
        plt.xticks([i+1 for i in range(5)], ["t0", "t1", "t2", "t3", "t4"], fontsize = 12)
        plt.title('Digit: {}'.format(digit_train_class[digit]), fontsize = 15)
    plt.legend()

#%% PLOT TRAIN GRAPH

for i in range(len(H_read_list_train)): 
    plot(H_read_list_train[i], "Train Digit {} t_int {}".format(int(digit_train_class[i]), int(t_int_read)))
    plt.savefig(out_folder_train + 'train_digit_{}_t_int_{}'.format(int(digit_train_class[i]), int(t_int_read)))

for i in range(len(H_read_list_train_digit_9)): 
    plot(H_read_list_train_digit_9[i], "Train Digit 9 t_int {}".format(i))
    plt.savefig(out_folder_train + 'train_digit_9_t_int_{}'.format(i))

#%% TRAINING 


train_in = np.array(hist_train_list)[:, 1:]
train_out = np.reshape(digit_train_class[:num_digits_train], (num_digits_train,1))

sc = StandardScaler()
ohe = OneHotEncoder()

scaler = sc.fit(train_in)

train_in = scaler.transform(train_in)
train_out = ohe.fit_transform(train_out).toarray()

model = Sequential()
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_in, train_out, epochs=3000)

plt.figure()
plt.plot(history.history['accuracy'], 'b', linewidth=2)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.ylim([0, 1.1])
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train'], loc='upper left')

np.savetxt(out_folder_train+"/accuracy_vs_epochs.txt", history.history['accuracy'])
    