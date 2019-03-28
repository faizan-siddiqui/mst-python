import numpy as np
import random
import heapq
import time

def generate_vertices(N):
    X = np.random.uniform(0,10,N)
    Y = np.random.uniform(0,10,N)
    return list(zip(X,Y))

def calc_weight(v_i, v_j):
    x_i, y_i = v_i
    x_j, y_j = v_j
    return np.sqrt(np.add(np.square(np.subtract(x_i, x_j)), np.square(np.subtract(y_i, y_j))))

def generate_weights(V):
    N = len(V)
    weights_matrix_output = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            weights_matrix_output[i,j] = calc_weight(V[i],V[j])
            
    return weights_matrix_output

class Vertex:
    def __init__(self, index, pred, key=float('inf')):
        self.index = index
        self.key = key
        self.pred = pred
    
    def __lt__(self, other):
        return self.key < other.key

def generate_vertices_list(V):
    N = len(V)
    vertices_list_output = []
    
    for i in range(N):
        vertices_list_output.append(Vertex(i,None))
    return vertices_list_output

def parent(i):
    return int((i+1)/2)-1

def heapDecreaseKey(heap, i):
    while i>0 and heap[parent(i)].key > heap[i].key:
        temp = heap[i]
        heap[i] = heap[parent(i)]
        heap[parent(i)] = temp
        i = parent(i)
    
        
def prim(V, E):
    vertices_list = generate_vertices_list(V)
    MST = []
    
    r = vertices_list[0]
    r.key = 0
    r.pred = r
    heapq.heapify(vertices_list)
    
    while (len(vertices_list) != 0):
        u = heapq.heappop(vertices_list)
        MST.append((u.pred,u))
        for v in vertices_list:
            if (E[u.index,v.index] < v.key):
                v.pred = u
                v.key = E[u.index,v.index]
                heapDecreaseKey(vertices_list, vertices_list.index(v))

    return MST

def sum_edge(G):
    distance = 0
    
    for (x,y) in G:
        distance += y.key
    return distance


def generate_subgraph(V,proportion):
    np.random.shuffle(V)
    n = len(V)
    subgraph_size = int(np.ceil(proportion*n))
    sub_nodes = V[0:subgraph_size]
    subgraph = generate_weights(sub_nodes)
    
    return (sub_nodes,subgraph)

def generate_p_MSTS_w_subgraph(p,N):
    vertices_set = []
    graph_weights_set = []
    subgraph_weights_set = []
    mst_set = []
    total_times = []
    subgraph_set = []
    sum_graph_edges = []
    sum_subgraph_edges_1 = []
    sum_subgraph_edges_2 = []
    sum_subgraph_edges_3 = []
    for i in range(0,p):
        tmp_vertices = generate_vertices(N)
        vertices_set.append(tmp_vertices)
        subnodes_1, subgraph_1 = generate_subgraph(tmp_vertices,0.75)
        subnodes_2, subgraph_2 = generate_subgraph(tmp_vertices,0.5)
        subnodes_3, subgraph_3 = generate_subgraph(tmp_vertices,0.25)
        graph_weights_set.append(generate_weights(vertices_set[i]))
        start_time = time.time()
        mst_graph = prim(vertices_set[i],graph_weights_set[i])
        total_times.append(time.time() - start_time)
        sum_graph_edges.append(sum_edge(mst_graph))
        subgraph_mst_graph_1 = prim(subnodes_1,subgraph_1)
        subgraph_mst_graph_2 = prim(subnodes_2,subgraph_2)
        subgraph_mst_graph_3 = prim(subnodes_3,subgraph_3)
        sum_subgraph_edges_1.append(sum_edge(subgraph_mst_graph_1))
        sum_subgraph_edges_2.append(sum_edge(subgraph_mst_graph_2))
        sum_subgraph_edges_3.append(sum_edge(subgraph_mst_graph_3))
    print("Average time of size {:d} is {:f}".format(N,np.average(total_times)))
    print("Average L(n) of size {:d} is {:f}".format(N,np.average(sum_graph_edges)))
    #print(sum(sum_graph_edges))
    #print(sum(sum_subgraph_edges))
    ratio_1 = sum(sum_graph_edges)/sum(sum_subgraph_edges_1)
    ratio_2 = sum(sum_graph_edges)/sum(sum_subgraph_edges_2)
    ratio_3 = sum(sum_graph_edges)/sum(sum_subgraph_edges_3)
    print("Ratio given by sum of edges of complete graph / sum of edges of subgraph")
    print("Ratio with {:d} and U_1 is {:f}".format(N,ratio_1))
    print("Ratio with {:d} and U_2 is {:f}".format(N,ratio_2))
    print("Ratio with {:d} and U_3 is {:f}".format(N,ratio_3))
    return 0

def calc_average_MST(mst_set):
    sum_edges = []
    for i in range(len(mst_set)):
        sum_edges.append(sum_edge(mst_set[i]))
    sum_msts = sum(sum_edges)
    return (np.average(sum_edges), sum_msts)

mst_set_200 = generate_p_MSTS_w_subgraph(20,200)
mst_set_400 = generate_p_MSTS_w_subgraph(20,400)
mst_set_800 = generate_p_MSTS_w_subgraph(20,800)
mst_set_1600 = generate_p_MSTS_w_subgraph(20,1600)
    