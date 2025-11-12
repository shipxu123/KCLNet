import os
import json
import networkx as nx
import matplotlib.pyplot as plt

from networkx.readwrite import json_graph


def write_graph_as_json(out_dir, data_type, filename, graph):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not data_type:
        sub_dir = out_dir + "/" + data_type
    else:
        sub_dir = out_dir

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    with open(sub_dir + "/" + filename + ".json", 'w') as f:
        f.write(json.dumps(json_graph.node_link_data(graph)))
    print("Graph saved in circuit_graphs directory")


def write_graph_as_image(graph, filename="./figs/network_graph.pdf"):
    # top = nx.bipartite.sets(graph)[0]
    if nx.is_connected(graph):
        if not nx.is_bipartite(graph):
            plt.figure(figsize=(32, 24))
            nx.spring_layout(graph)
            nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray')
            plt.savefig(filename)
            plt.close()
        else:
            top = nx.bipartite.sets(graph)[0]
            pos = nx.bipartite_layout(graph, top)
            plt.figure(figsize=(32, 24))
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
            plt.savefig(filename)
            plt.close()

    else:
        print("Bipartite sets could not be determined due to a disconnected graph.")
        import pdb
        pdb.set_trace()
        for i, component in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(component)

            top = nx.bipartite.sets(subgraph)[0]

            pos = nx.bipartite_layout(graph, top)

            plt.figure(figsize=(32, 24))
            nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')

            plt.savefig('.'.join(filename.split('.')[:-1]) + str(i) + '.pdf')
            plt.close()
