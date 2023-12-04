"""
 _._     _,-'""`-._
(,-.`._,'(       |\`-/|
    `-.-' \ )-`( , o o)
          `-    \`_`"'-
            CAT.
"""

import io
import time
import subprocess
import json
from collections import defaultdict, ChainMap

from Misc.graph_edit import *
from Misc.utils import list_of_lists_to_list

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, sort_edge_index

#
# Constants
#
label_ham_cycle = 0
label_articulation_vertex = 1
label_pooling_vertex = 2
label_block_vertex = 3
label_original_Vertex = 4
label_global_pool = 5
label_spider_pool = 6

pos_edge_type = 0
# Hamiltonian cycle distance = 0 -> not in a hamiltonian cycle
pos_ham_dis = 1

label_edge_original = 0
label_edge_ham_cycle = 1
label_edge_ham_pool = 2
label_edge_pool_block = 3
label_edge_pool_art = 4

# Edge from articulation node to original node or articulation node
label_edge_art_original = 5
label_edge_block_ham = 6
label_edge_pool_ham = 7
label_edge_shortcut = 8 # or spiderweb

#
# CODE
#

def maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label, nr_edges = 1):
    if has_edge_attr:
        new_feat = torch.zeros([nr_edges, e_shape])
        new_feat[:, pos_edge_type] = label
        return torch.cat((edge_attr, new_feat), dim=0)

def get_hamiltonian_cycles(g: Data, config): 
    from Misc.run_converter import to_aids_format

    graphstring = ''
    
    # graph conversion to textual input format for all graphs in the current batch
    # assumes undirected graphs and ignores labels
    file = io.BytesIO()
    to_aids_format(g, 0, file, undirected=True)
    graphstring = file.getvalue().decode('latin1')
    
    # the actual computation of outerplanarity and Hamiltonian cycles in a subprocess
    cmd = [config['binfile'], '-sw', '-']
    proc = subprocess.run(args=cmd, capture_output=True, input=graphstring.encode("utf-8"))

    # parsing of the results (directly from stdout of the process)
    jstr = proc.stdout.decode("utf-8")
    jsobjects = json.loads(jstr)

    return jsobjects

def edge_in_ham_cycles(edge, list_vertices_in_ham_cycle):
    return int(edge[0]) in list_vertices_in_ham_cycle and int(edge[1]) in list_vertices_in_ham_cycle

def get_block_idx(edge, ham_cycle_dict):
    """
    Returns id of the block of an edge
    """
    blocks1 = ham_cycle_dict[int(edge[0])]
    blocks2 = ham_cycle_dict[int(edge[1])]
    
    # Intersection between the two list of cycles
    blocks = list(set(blocks1) & set(blocks2))
    
    # An edge can be in at most one hamiltonian cycle
    assert len(blocks) <= 1
    
    return blocks[0] if (len(blocks) == 1) else None
    
    
    
class CyclicAdjacencyTransform(BaseTransform):
    """
            
    """
    def __init__(self, debug = False, spiderweb = True):
        self.config = {'binfile': './outerplanaritytest', 'verbose': True}
        self.debug = debug
        self.spiderweb = spiderweb
        
    def __call__(self, data: Data):
        edge_index, x, edge_attr = data.edge_index, data.x, data.edge_attr 
        has_edge_attr = edge_attr is not None 
        has_vertex_feat = x is not None
        
        if (len(edge_attr.shape) == 1):
            edge_attr = torch.unsqueeze(edge_attr, 1)

        nr_vertices_in_og_graph = get_nr_vertices(data)

        # Increment features by 1 to make room for new "empty" edges / vertices
        # This assumes that all features are categorical
        if has_vertex_feat:
            x += 1
            x_shape = x.shape[1]+1

        # Shift features (see: incrementing of vertex features)
        if has_edge_attr:
            edge_attr += 1
            e_shape = edge_attr.shape[1]+2

            # Make space for labels and hamiltonian cycle distance
            edge_attr = torch.cat((torch.zeros([edge_attr.shape[0], 2]), edge_attr), dim = 1)


        ham_cycle_info = get_hamiltonian_cycles(data, self.config)[0]
        # print(f"ham_cycle_info: {ham_cycle_info}")
        blocks_dict = dict(ChainMap(*list(map(lambda x: x["blocks"], ham_cycle_info['ccs']))))        
        ham_cycles_dict = dict(ChainMap(*list(map(lambda x: x["hamiltonianCycles"], ham_cycle_info['ccs']))))
              
        # print("\nBefore renaming")
        # print(f"ham_cycles_dict: {ham_cycles_dict}")
        # print(f"blocks_dict: {blocks_dict}")
                
        
                
        keys = list(blocks_dict.keys())
        
        # Maps original (negative) indies to the new (positive) indices
        og_block_idx_to_new_idx = {}

        # Maps new indices to whether the block has Hamiltonian cycle
        is_ham_cycle = {}
        
        # Change from the negative index to an index starting at 0 and increasing
        for key in keys:
            new_key = abs(int(key)) - 1
            og_block_idx_to_new_idx[int(key)] = new_key
            
            # Rename
            blocks_dict[new_key] = blocks_dict[key]
            del blocks_dict[key]
            
            is_ham_cycle[new_key] = key in ham_cycles_dict
            
            if is_ham_cycle[new_key]:
                ham_cycles_dict[new_key] = ham_cycles_dict[key]
                del ham_cycles_dict[key]
                
                
        # print(f"og_block_idx_to_new_idx: {og_block_idx_to_new_idx}")
        # print("\nAfter renaming")
        # print(f"ham_cycles_dict: {ham_cycles_dict}")
        # print(f"blocks_dict: {blocks_dict}")
        # print("\n")
  
        # shortcut_edges = list_of_lists_to_list(list(map(lambda x: x["shortcutEdges"], ham_cycle_info['ccs'])))
        vertex_to_spiderweb_pool = dict(ChainMap(*list(map(lambda x: x["spiderweb"], ham_cycle_info['ccs']))))
        # print(f"vertex_to_spiderweb_pool: {vertex_to_spiderweb_pool}")
        spiderweb_pooling_to_vertex_list = defaultdict(lambda: [])
        for (vertex, spider_web_pooling_vertex) in vertex_to_spiderweb_pool.items():
            spiderweb_pooling_to_vertex_list[spider_web_pooling_vertex].append(int(vertex))
        
        # print(f"spiderweb_pooling_to_vertex_list: {spiderweb_pooling_to_vertex_list}")
        articulation_vertices = []
        
        # Dict to map vertices in ham cycle to id of cycle
        vertices_to_block_idx = defaultdict(lambda: [])
     
        for (key, block) in blocks_dict.items():
            for vertex in block:
                vertices_to_block_idx[(vertex)].append(key)
              
        # print(f"vertices_to_block_idx: {vertices_to_block_idx}")
        
        vertices_in_ham_cycles = list(set(vertices_to_block_idx.keys()))
        vertices_in_ham_cycles.sort()       
        new_edge_index = torch.clone(edge_index)
        nr_vertices_in_og_graph = get_nr_vertices(data)
        
        # Duplicate and orient hamiltonian cycles
        # also collect articulation vertices

        # (Vertex in og graph, block idx) -> vertex in new graph
        vertex_in_block_to_vertex_idx = {}
        vertex_to_duplicate_dict = {}

        created_vertices = 0
        already_seen_vertices = []
        if has_vertex_feat:
            labels = [label_original_Vertex for _ in range(nr_vertices_in_og_graph)]
            
        for (block_idx, block) in blocks_dict.items():
            # Do not need to duplicate nodes if they are not in a Hamiltonian cycle
            if not is_ham_cycle[block_idx]:
                for vertex in block:
                    vertex_in_block_to_vertex_idx[(vertex, block_idx)] = vertex
                    already_seen_vertices.append(vertex)
                continue
            
            for vertex in block:
                # Create original vertex entry
                if vertex in already_seen_vertices:
                    idx = nr_vertices_in_og_graph + created_vertices
                    created_vertices += 1
                    
                    if has_vertex_feat:
                        x = torch.cat((x, torch.unsqueeze(x[vertex], 0)), dim=0)
                        labels.append(label_ham_cycle)
                else:
                    idx = vertex
                    already_seen_vertices.append(vertex)
                    if has_vertex_feat:
                        labels[idx] = label_ham_cycle
                vertex_in_block_to_vertex_idx[(vertex, block_idx)] = idx

                # Duplicate
                idx = nr_vertices_in_og_graph + created_vertices
                created_vertices += 1
                vertex_to_duplicate_dict[(vertex, block_idx)] = idx
                if has_vertex_feat:
                    x = torch.cat((x, torch.unsqueeze(x[vertex], 0)), dim=0)
                    labels.append(label_ham_cycle)
        
        # print(f"vertex_to_duplicate_dict: {vertex_to_duplicate_dict}\n")
        if has_vertex_feat:
            x = torch.cat((torch.unsqueeze(torch.tensor(labels), 1), x), dim= 1)
            
        edges_outside_blocks = []
        get_duplicate_vertex = lambda vertex, block_idx: vertex_to_duplicate_dict[(vertex, block_idx)]
        nr_vertices_in_graph_with_duplication = nr_vertices_in_og_graph + created_vertices

        # Duplicate and orient Hamiltonian cycles
        for i in range(edge_index.shape[1]):
            block_idx = get_block_idx(edge_index[:, i], vertices_to_block_idx)
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            
            # Case: not a block
            if block_idx is None:
                # Check if vertices incident to edge are articulation vertices
                articulation_vertices += [int(edge[j]) for j in [0, 1] if (int(edge[j]) in vertices_in_ham_cycles)]
                edges_outside_blocks += [i]
                continue
            
            # Check for articulation vertex
            for j in [0, 1]:
                if len(vertices_to_block_idx[int(edge[j])]) > 1:
                    articulation_vertices += [int(edge[j])] 
            
            # Case: non-outerplanar
            if not is_ham_cycle[block_idx]:
                continue
                
            if has_edge_attr:
                edge_attr[i, pos_edge_type] = label_edge_ham_cycle    
                
            ham_cycle = blocks_dict[block_idx]
            pos_in_ham_cycle1, pos_in_ham_cycle2 = ham_cycle.index(edge[0]), ham_cycle.index(edge[1])
            modulus = len(ham_cycle) 

            # This checks if the vertices are part of the hamiltonian cycle in a clockwise / counter clockwise fashion
            # Clockwise (+1): move edge if some of the vertices are not original
            if (pos_in_ham_cycle1 + 1) % modulus == pos_in_ham_cycle2:
                # Potentially change incident vertices to duplicate vertex
                for j in [0, 1]:
                    vertex_idx = vertex_in_block_to_vertex_idx.get((edge[j], block_idx))
                    if vertex_idx is not edge[j]:
                        new_edge_index[j, i] = vertex_idx
                
                if has_edge_attr:
                    edge_attr[i, pos_ham_dis] = 1

            # Counter clockwise (-1): create new vertices / edges 
            elif (pos_in_ham_cycle1 + modulus - 1) % modulus == pos_in_ham_cycle2:     
                # Change incident vertices to duplicate vertex              
                new_edge_index[0, i] = get_duplicate_vertex(edge[0], block_idx)
                new_edge_index[1, i] = get_duplicate_vertex(edge[1], block_idx)
                           
                if has_edge_attr:
                    edge_attr[i, pos_ham_dis] = 1

            # Neither (these are "diagonal" edges that are not part of the hamiltonian cycle): duplicate
            else:
                v1 = get_duplicate_vertex(edge[0], block_idx)
                v2 = get_duplicate_vertex(edge[1], block_idx)
                new_edge_index = add_dir_edge(new_edge_index, v1, v2)
                
                if has_edge_attr:
                    new_feat = torch.clone(edge_attr[i, :]).unsqueeze(0)    
                    
                    new_feat[:, pos_edge_type] = label_edge_original
                    edge_attr[i, pos_edge_type] = label_edge_original
                    
                    # Add distance in hamiltonian cycle to new and old edge
                    distance_1_to_2 = (pos_in_ham_cycle2 + modulus - pos_in_ham_cycle1) % modulus
                    distance_2_to_1 = modulus - distance_1_to_2
                    new_feat[0, pos_ham_dis] = distance_2_to_1
                    edge_attr[i, pos_ham_dis] = distance_1_to_2
                    
                    assert (distance_1_to_2 + distance_2_to_1) == modulus 
                    edge_attr = torch.cat((edge_attr, new_feat), dim=0)                      
            
       
        articulation_vertices = list(set(articulation_vertices))

        # Create vertices that pool the representation for a single vertex
        created_vertices = 0
        vertex_to_pooling = {}
        for (block_idx, block) in blocks_dict.items():
            
            # No pooling vertices for non-outerplanar blocks
            if not is_ham_cycle[block_idx]:
                for vertex in block:
                    vertex_to_pooling[(vertex, block_idx)] = vertex
                continue
            
            for vertex in block:
                idx = nr_vertices_in_graph_with_duplication + created_vertices
                created_vertices += 1
                
                vertex_to_pooling[(vertex, block_idx)] = idx
                vertex_og = vertex_in_block_to_vertex_idx[(vertex, block_idx)]
                vertex_duplicate = vertex_to_duplicate_dict[(vertex, block_idx)]
                
                new_edge_index = add_undir_edge(new_edge_index, vertex_og, idx)             
                new_edge_index = add_undir_edge(new_edge_index, vertex_duplicate, idx)
                
                edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_ham_pool, 4)

                if has_vertex_feat:
                    new_feat = torch.cat((torch.tensor([label_pooling_vertex]), x[vertex, 1:]))
                    x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
                        
        nr_vertices_in_graph_with_pooling = nr_vertices_in_graph_with_duplication + created_vertices  
        
        # print(f"vertex_to_pooling: {vertex_to_pooling}")
        
        # Create block vertices
        created_vertices = 0
        block_to_block_vertex_idx = {}
        for (block_idx, block) in blocks_dict.items():
            idx = nr_vertices_in_graph_with_pooling + created_vertices
            created_vertices += 1
            block_to_block_vertex_idx[block_idx] = idx       
                 
            if has_vertex_feat:
                new_feat = torch.cat((torch.tensor([label_block_vertex]), torch.zeros(x_shape-1)))
                x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
            
            for vertex in block:      
                if is_ham_cycle[block_idx]:
                    pooling_vertex = vertex_to_pooling[(int(vertex), block_idx)]
                    new_edge_index = add_undir_edge(new_edge_index, pooling_vertex, idx)
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_block_ham, 2)
                    
                    # if vertex in articulation_vertices:
                    #     new_edge_index = add_undir_edge(new_edge_index, pooling_vertex, idx)
                    #     edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_block, 2)
                else:
                    new_edge_index = add_undir_edge(new_edge_index, vertex_in_block_to_vertex_idx[(vertex, block_idx)], idx)
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_block_ham, 2)
                    
        # print(f"block_to_block_vertex_idx: {block_to_block_vertex_idx}")

        nr_vertices_in_graph_with_block = nr_vertices_in_graph_with_pooling + created_vertices
        
        # Create articulation vertices
        created_vertices = 0
        vertex_to_articulation = {}
        for i, articulation_vertex in enumerate(articulation_vertices):
            idx = nr_vertices_in_graph_with_block + i
            vertex_to_articulation[articulation_vertex] = idx
            created_vertices += 1
            if has_vertex_feat:
                new_feat = torch.cat((torch.tensor([label_articulation_vertex]), x[articulation_vertex, 1:]))
                x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)

            for (block_idx, block) in blocks_dict.items():
                # if not is_ham_cycle[block_idx]:
                #     continue
                
                if articulation_vertex in block:
                    # Edge from articulation vertex to pooling vertex
                    pooling_vertex = vertex_to_pooling[(int(articulation_vertex), block_idx)]
                    new_edge_index = add_undir_edge(new_edge_index, pooling_vertex, idx)
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_art, 2)
                    
        nr_vertices_in_new_graph = nr_vertices_in_graph_with_block + created_vertices
        
        # Add a virtual node pooling blocks:
        idx = nr_vertices_in_new_graph
        
        # print(f"Virtual pooling idx: {idx}")
        if has_vertex_feat:
            new_feat = torch.cat((torch.tensor([label_global_pool]), torch.zeros(x_shape-1)))
            x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
                
        for block in block_to_block_vertex_idx.values():
            new_edge_index = add_undir_edge(new_edge_index, block, idx) 
            edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_ham, 2)
            
        nr_vertices_in_new_graph += 1
        
        # Add shortcut edges
        # print(f"raw_id_to_block_vertex: {raw_id_to_block_vertex}")
        # for edge in shortcut_edges:
            # print(edge)
            # p, q = edge[0], edge[1]
            # print(f"p: {p}, q: {q}")
            # # If incident vertices are block vertices then map them to the newly created block vertices
            # if p < 0:
            #     p = block_to_block_vertex_idx[og_block_idx_to_new_idx[p]]
            # if q < 0:
            #     q = block_to_block_vertex_idx[og_block_idx_to_new_idx[q]]
                
            # if p < 0 or q < 0:
            #     continue
                
            # print(f"p: {p}, q: {q}")
            # new_edge_index = add_undir_edge(new_edge_index, p, q) 
            # edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)
        
        # Spiderweb Shorcuts
        if self.spiderweb:
            # print("\n\n")
            # print(f"spiderweb_pooling_to_vertex_list: {spiderweb_pooling_to_vertex_list}")
            for created_vertices, vertices_ls in enumerate(spiderweb_pooling_to_vertex_list.values()):
                new_feat = torch.cat((torch.tensor([label_spider_pool]), torch.zeros(x_shape-1)))
                x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
                spider_web_vertex_idx = nr_vertices_in_new_graph + created_vertices
                
                for vertex_idx in vertices_ls:
                    # Block vertices
                    if vertex_idx < 0: 
                        vertex_idx = block_to_block_vertex_idx[og_block_idx_to_new_idx[vertex_idx]]
                        new_edge_index = add_undir_edge(new_edge_index, vertex_idx, spider_web_vertex_idx) 
                        edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)
                    else:
                        if vertices_to_block_idx[vertex_idx] != []:     
                            for block_idx in vertices_to_block_idx[vertex_idx]:
                                v1 = vertex_in_block_to_vertex_idx[(vertex_idx, block_idx)]
                                new_edge_index = add_undir_edge(new_edge_index, v1, spider_web_vertex_idx) 
                                edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)
                                
                                if (vertex_idx, block_idx) in vertex_to_duplicate_dict:
                                    v2 = vertex_to_duplicate_dict[(vertex_idx, block_idx)]
                                    new_edge_index = add_undir_edge(new_edge_index, v2, spider_web_vertex_idx) 
                                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)
                        else:
                            new_edge_index = add_undir_edge(new_edge_index, vertex_idx, spider_web_vertex_idx) 
                            edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)

        # print(f"vertices_to_block_idx: {vertices_to_block_idx}")
        
        # Clean up edges: move edges to articulation vertices 
        for i in edges_outside_blocks:
            for j in [0, 1]:
                if new_edge_index[j, i] in articulation_vertices:
                    new_edge_index[j, i] = vertex_to_articulation[int(new_edge_index[j, i])]

        data.edge_index = new_edge_index.type(data.edge_index.type())
        if has_vertex_feat:
            data.x = x.type(data.x.type())
        if has_edge_attr:
            data.edge_attr = edge_attr.type(data.edge_attr.type())
        data.num_nodes = x.shape[0]

        assert data.edge_index.shape[1] == data.edge_attr.shape[0]
        if data.edge_index.shape[1]> 0:
            assert data.x.shape[0] >= (torch.max(data.edge_index) + 1)
            assert torch.min(data.edge_index) >= 0
            
        # quit()
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.spiderweb})'

"""
END of CAT
"""