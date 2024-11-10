import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from extract import getDir
from load_map import loadMap
from collections import deque
from algorithms import fordFulkerson, dinics, reset_map, estimate_max_capacity
from edmonds_karp import find_maximum_flow_using_edmonds_karp

ratio = 111196.2878

class ImageCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Canvas")
        #1220 x 1019  self.dim = {'width': 1344, 'height': 686}
        self.dim = {'width': 1220, 'height': 1019}
        self.canvas = tk.Canvas(root, width=self.dim['width'], height=self.dim['height'], bg='white')
        self.canvas.pack()

        self.image = None
        self.image_layer = None
        self.node_layer = []
        self.edge_layer = []
        
        #self.map_name = 'newgraph_conso.osm'
        self.map_name = 'map_3_areas.osm'
        self.graph = loadMap(self.map_name)
        # create capacity for edges
        for edge in self.graph.edges(data=True):
            #print(f'old edge: {edge}')
            edge[2]['capacity'] = estimate_max_capacity(edge[2])
            #print(f'new edge: {edge}')
        # 106.65416 - 106.71553    ;  10.81754 - 10.76729
        # self.bbox = {
        #     'max_lon': 106.71535,
        #     'min_lon': 106.64738,
        #     'max_lat': 10.81864,
        #     'min_lat': 10.78786,
        #     'range_lon': 106.71535 - 106.64738,
        #     'range_lat': 10.81864 - 10.78786
        #     }
        print(self.graph)
        self.bbox = {
            'max_lon': 106.71553,
            'min_lon': 106.65416,
            'max_lat': 10.81754,
            'min_lat': 10.76729,
            'range_lon': 106.71553 - 106.65416,
            'range_lat': 10.81754 - 10.76729
            }
        self.ratio = {'x': self.dim['width']/self.bbox['range_lon'], 'y': self.dim['height']/self.bbox['range_lat']}
        self.dist = 3
        self.start_node = None
        self.end_node = None

        # Load an image
        self.load_image(getDir('map_3_areas.png'))
        self.load_image_button = tk.Button(root, text="Print flow/paths", command=self.print_flow_paths)
        self.load_image_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Create an entry widget for user input
        #self.entry = tk.Entry(self.root, width=40)
        #self.entry.pack(side=tk.LEFT, padx=10, pady=10)  # Pack the entry on the left

        # Create a button that will call the print_text function when clicked
        self.print_button = tk.Button(self.root, text="Toggle nodes", command=self.toggle_node)
        self.print_button.pack(side=tk.LEFT, padx=10, pady=10)  # Pack the button next to the entry

        self.canvas.bind("<Button-1>", self.on_mouse_lclick)
        self.canvas.bind("<Button-2>", self.on_mouse_rclick)
        self.canvas.bind("<Button-3>", self.on_mouse_rclick)
        self.draw_map()
        
        self.test()
        return

    def load_image(self, img_file):
        self.image = Image.open(img_file)
        #self.image.thumbnail((800, 600))  # Resize image to fit the canvas
        self.tk_image = ImageTk.PhotoImage(self.image)
        if self.image_layer is not None:
            self.canvas.delete(self.image_layer)  # Remove previous image if any
        
        self.image_layer = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        return
    
    def convert_coord_2_pixl(self, lon, lat):
        x = (lon - self.bbox['min_lon']) * self.ratio['x']
        y = (lat - self.bbox['min_lat']) * self.ratio['y']
        return x, y
    
    def convert_pixl_2_Coord(self, x, y):
        lon = x / self.ratio['x'] + self.bbox['min_lon']
        lat = y / self.ratio['y'] + self.bbox['min_lat']
        return lon, lat
    
    def draw_map(self):
        for node in self.graph.nodes(data=True):
            if 'lon' in node[1]:
                #print(node[1]['lon'])
                x, y = self.convert_coord_2_pixl(node[1]['lon'], node[1]['lat'])
                #x, y = int(x), int(y)
                #print(x, y)
                rect = self.canvas.create_rectangle(x-self.dist, y-self.dist, x+self.dist, y+self.dist, outline='red')
                self.node_layer.append([node[0], rect])
        return
    
    def on_mouse_lclick(self, event, color='blue'):
        x, y = event.x, event.y
        lon, lat = self.convert_pixl_2_Coord(x, y)
        if self.start_node:
            self.canvas.itemconfig(self.start_node[1], outline="red")
        self.start_node = self.find_closest_point(lon, lat, color=color)
        if self.start_node:
            self.canvas.itemconfig(self.start_node[1], outline="blue")
            print(f"Left clicked at: <<{self.start_node[0]}>> ({x}, {y}) => ({lon}, {lat}) => node: {self.start_node}")
        return self.start_node
    
    def on_mouse_rclick(self, event, color='green'):
        x, y = event.x, event.y
        lon, lat = self.convert_pixl_2_Coord(x, y)
        if self.end_node:
            self.canvas.itemconfig(self.end_node[1], outline="red")
        self.end_node = self.find_closest_point(lon, lat, color=color)
        if self.end_node:
            self.canvas.itemconfig(self.end_node[1], outline="green")
            print(f"Right clicked at: <<{self.end_node[0]}>> ({x}, {y}) => ({lon}, {lat}) => node: {self.end_node}")
        return self.end_node
        
    def save_as(self):
        file_name = self.entry.get()  # Get text from the entry widget
        print(file_name)  # Print the text to the terminal
        return
    
    def find_closest_point(self, lon, lat, color='red'):
        for node_rect in self.node_layer:
            node_id = node_rect[0]
            node = self.graph.nodes[node_id]
            if 'lon' not in node:
                continue
            if abs(node['lon'] - lon) < self.dist/10000 and abs(node['lat'] - lat) < self.dist/10000:
                return node_rect
        return None
    
    def print_flow_paths(self, algo=fordFulkerson, reset=reset_map):
        if not self.start_node:
            print('No start node picked')
            return
        if not self.end_node:
            print('No end node picked')
            return
        
        max_flow, paths_list, true_paths_list, true_level_graph = algo(self.graph, self.start_node[0], self.end_node[0])
        for path in paths_list:
            # print path
            print(f'paths found: {path}')
            
        for path in true_paths_list:
            # print path
            print(f'true paths found: {path}')
        
        print(f'number of paths: {len(paths_list)}')
        print(f'number of true paths: {len(true_paths_list)}')
        if true_level_graph:
            print(len(true_level_graph['nodes']), len(true_level_graph['edges']))
        #for node in true_level_graph:
        #    # draw nodes
        #    for node_rect in self.node_layer:
        #        if 
        
        print(f'max flow: {max_flow}')
        
        # reset_map
        if algo == dinics:
            reset(self.graph, true_level_graph)
        return
    
    def toggle_node(self):
        # Toggle the visibility of the drawing layer
        if self.node_layer:
            for node in self.node_layer:
                current_state = self.canvas.itemcget(node[1], "state")
                new_state = "hidden" if current_state == "normal" else "normal"
                self.canvas.itemconfigure(node[1], state=new_state)

    def test(self):
        #for node in self.graph.nodes:
        #    print(f'node: {self.graph.nodes[node]}')
        #    print(f'edge: {self.graph.edges(node, data=True)}')
        #    print(f'edge i: {self.graph.in_edges(node)}')
        #    print(f'edge o: {self.graph.out_edges(node)}')
        return

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCanvasApp(root)
    root.mainloop()
