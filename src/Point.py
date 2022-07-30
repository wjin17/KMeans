import numpy as np

class Point:
    def __init__(self, label):
        self.label = label

    def get_label(self):
        return self.label
    
    def set_features(self, features):
        self.features = features
    
    def get_features(self):
        return self.features
    
    def get_sorted_features(self):
        return {k: v for k, v in sorted(self.features.items(), key=lambda item: item[1])}

    def get_coordinates(self):
        return np.array([x for x in self.features.values()])