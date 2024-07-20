import networkx as nx 
from shapely import GEOSException 
import geopandas as gpd 
import numpy as np
from functools import partial
import json
from shapely.geometry import Polygon, MultiPolygon, shape

class InvalidGeojsonError(Exception):
    pass

class PolygonGraph(object):
    """
    PolygonGraph class to create a networkx graph from a geojson file

    Parameters
    ---
    - geojson_path: str, path to the geojson file

    Attributes
    ---
    - G: networkx.Graph, the graph of the geojson file

    Methods
    ---
    - graphml: Save the graph as a graphml file
    - get_adjacency_matrix: Get the adjacency matrix of the graph. The nodes are sorted in the adjacency matrix by their region ids.

    Note
    ---
    The nodes of the graph are the region ids of the polygons in the geojson file. So even if there are multiple polygons
    associated with a region id, there will be only one node in the graph for that region id and the edges will be added 
    to the graph based on the proximity of the polygons in the geojson file.


    """
    def __init__(self, geojson_path):
        self.geojson_path = geojson_path
        # Load the geojson file and parse the properties of the features
        with open(geojson_path, "r") as f:
            self.json = json.load(f)
        
        if not isinstance(self.json, dict):
            raise InvalidGeojsonError("Invalid geojson file")

        self.prop_keys = []
        for feature in self.json["features"]:
            for k in feature["properties"]["data"].keys():
                if k not in self.prop_keys:
                    self.prop_keys.append(k)

        self.prop_keys_ = []
        for k in self.prop_keys:
            if k == "id":
                self.prop_keys_.append("region_id")
                continue
            elif " " in k:
                k_ = k.split(" ")[0]
                self.prop_keys_.append(k_)
            else:
                self.prop_keys_.append(k)

        # Parse the geojson file and create a geodataframe
        self._parse_geojson()

        # Create a networkx graph
        self.G = nx.Graph()

        # Add nodes to the graph
        self.geodf["region_id"] = self.geodf["region_id"].astype(str)
        region_ids = list(set(self.geodf["region_id"].values.tolist()))
        self.G.add_nodes_from([str(rid) for rid in region_ids])

        # Add Node Attributes
        attribute_columns = [
            "region_id",
            "name",
            "acronym",
            "type",
            "parent_structure_id",
            "color_hex_triplet",
        ]
        self.node_attributes = self.geodf[attribute_columns].to_dict()
        for attr in attribute_columns:
            attr_dict = self.node_attributes[attr]
            attr_dict = {str(k):v for k,v in attr_dict.items()}
            attr_name = attr
            if attr == "color_hex_triplet":
                attr_name = "color"
            nx.set_node_attributes(self.G, attr_dict, name=attr_name)

        # Add edges to the graph
        self._add_edges()

    @staticmethod
    def _parse_feature_properties(row):
        """Convert the properties string to a dictionary"""
        props = row["data"]
        prop_data = json.loads(props)
        row["data"] = prop_data
        return row

    @staticmethod
    def _expand_data_column(row, prop_keys, prop_keys_):
        """Expand the properties dictionary to columns"""
        props = row["data"]
        for k, v in zip(prop_keys, prop_keys_):
            try:
                row[v] = props[k]
            except KeyError:
                row[v] = None
        return row

    @staticmethod
    def _tree_idx_to_region_id(geodf, tree_idx: int) -> str:
        """Map the tree index to the region id"""
        return geodf.loc[geodf["tree_idx"] == tree_idx, "region_id"].values[0]

    @staticmethod
    def add_edges_based_on_proximity(row, G, tree, region_id_mapper):
        """Add edges to the graph based on the proximity of the polygons"""
        try:
            overlaps = tree.query(row["geometry"], predicate="overlaps")
        except GEOSException:
            overlaps = []
        intersects = tree.query(row["geometry"], predicate="intersects")
        try:
            touches = tree.query(row["geometry"], predicate="touches")
        except GEOSException:
            touches = []
        connexions = set(overlaps).union(set(intersects).union(set(touches)))
        connexions = list(connexions)
        connexions.remove(row["tree_idx"])
        for c in connexions:
            if row["region_id"] != region_id_mapper(c):
                G.add_edge(row["region_id"], region_id_mapper(c))
        return row

    def _parse_geojson(self):
        """Parse the geojson file and create a geodataframe"""
        data = gpd.read_file(self.geojson_path, on_invalid='warn')
        data = data.dropna(subset=["geometry"])
        expander = partial(
            self._expand_data_column,
            prop_keys=self.prop_keys,
            prop_keys_=self.prop_keys_,
        )
        data = data.apply(self._parse_feature_properties, axis=1)
        data = data.apply(expander, axis=1)
        data["tree_idx"] = np.arange(data.shape[0])
        strTree = data["geometry"].sindex
        data = data.drop(columns=["data"])
        self.geodf = data
        self.geo_tree = strTree

    def _add_edges(self):
        """Add edges to the graph based on the proximity of the polygons"""
        region_idx_mapper = partial(self._tree_idx_to_region_id, self.geodf)
        add_edges = partial(
            self.add_edges_based_on_proximity,
            G=self.G,
            tree=self.geo_tree,
            region_id_mapper=region_idx_mapper,
        )
        self.geodf = self.geodf.apply(add_edges, axis=1)
    
    def graphml(self, path):
        """Save the graph as a graphml file"""
        nx.write_graphml(self.G, path)
    
    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph"""
        sorted_node_list = sorted(self.G.nodes())
        return nx.adjacency_matrix(self.G,nodelist=sorted_node_list).todense()


    
 #     #  #####     #     #####  ####### 
 #     # #     #   # #   #     # #       
 #     # #        #   #  #       #       
 #     #  #####  #     # #  #### #####   
 #     #       # ####### #     # #       
 #     # #     # #     # #     # #       
  #####   #####  #     #  #####  ####### 


# pg = PolygonGraph("B_222_FB74-SL_125-ST_NISL-SE_373_.geojson")
# adj_matrix = pg.get_adjacency_matrix()
