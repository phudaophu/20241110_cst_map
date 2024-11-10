# Python program to convert JSON to Python
import os

def getDir(file_name):
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    dir = os.path.join(path, file_name)
    print(dir)
    return dir

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(str(line) + "\n")
    return

def getMap(nodes):
    max_lon = float('-inf')
    max_lat = float('-inf')
    min_lon = float('inf')
    min_lat = float('inf')
    for i in nodes:
        max_lon = max(max_lon, i['long'])
        min_lon = min(min_lon, i['long'])
        max_lat = max(max_lat, i['lat'])
        min_lat = min(min_lat, i['lat'])
    print(max_lon, min_lon, max_lat, min_lat)
    print((max_lon + min_lon)/2, (max_lat + min_lat)/2, (max_lon - min_lon)/2, (max_lat - min_lat)/2)
    return max_lon, min_lon, max_lat, min_lat

def smartParse(str):
    str = str.replace("\n","")
    #print(str)
    try:
        i = int(str)
        return i
    except:
        pass
    try:
        f = float(str)
        return f
    except:
        pass
    return str

def extractCSV(txt, filter=[]):
    title = None
    data = []
    for line in txt:
        line_data = line.split(',')
        for i in filter:
            line_data.pop(i)
        if not title:
            title = [smartParse(i) for i in line_data]
        new_data = {}
        for i in range(len(line_data)):
            new_data.update({title[i]: smartParse(line_data[i])})
        #print(new_data)
        data.append(new_data)
    return data[1:]

def extractData():
    dir_edges = getDir('train.csv')
    data_edges = open(dir_edges, encoding='utf8')
    data_edges = extractCSV(data_edges, [13, 10, 9, 5, 4, 3, 2, 1, 0]) # pop filter from last to 1st
    data_edges = sorted([dict(t) for t in {tuple(d.items()) for d in data_edges}], key=lambda d: (d['s_node_id'], d['e_node_id'])) # remove duplicates
    #for i in data_edges:
    #    print(i)
    print('Number of edges: ', len(data_edges))
    saveToTxt(data_edges, 'log/edges.txt') # save to log

    dir_nodes = getDir('nodes.csv')
    data_nodes = open(dir_nodes, encoding='utf8')
    data_nodes = extractCSV(data_nodes)
    data_nodes = sorted([dict(t) for t in {tuple(d.items()) for d in data_nodes}], key=lambda d: d['_id']) # remove duplicates
    #for i in data_nodes:
    #    print(i)
    print('Number of nodes: ', len(data_nodes))
    saveToTxt(data_nodes, 'log/nodes.txt') # save to log
    #print(data_nodes[0])
    
    # TRAIN.CVS ==========================================
    # s_node_id, e_node_id, length, street_id, street_name, long_snode, lat_snode, long_enode, lat_enode
    # [
    #    [..],
    #    [..],
    #    [..],
    #    ...
    #    [..],    
    # ]
    
    # NODES.CVS ==========================================
    # _id, long, lat
    # [
    #    [..],
    #    [..],
    #    [..],
    #    ...
    #    [..],    
    # ]
    
    getMap(data_nodes)
    return data_nodes, data_edges
