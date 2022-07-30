from cmath import inf
from random import random
import numpy as np

from src.Centroid import Centroid

test_centroids = [
    Centroid(0, {'x': 2.8891352610454177, 'y': 2.897209939975002}),
    Centroid(1, {'x': 1.2189567534989494, 'y': 1.8251834047361815}),
    Centroid(2, {'x': 1.5201237544815864, 'y': 1.5660825173109565})
]

class KMeans:
    @staticmethod
    def generate_centroids(numCentroids, points, is_random):
        random_centroids = []
        if is_random:
            maxValues = {}
            minValues = {}

            for point in points:
                for feature in point.features:
                    if feature not in maxValues or maxValues[feature] < point.features[feature]:
                        maxValues[feature] = point.features[feature]

                    if feature not in minValues or minValues[feature] > point.features[feature]:
                        minValues[feature] = point.features[feature]
            
            for i in range(numCentroids):
                random_features = {}
                for feature in maxValues:
                    random_features[feature] = random() * (maxValues[feature] - minValues[feature]) + minValues[feature]
                random_centroids.append(Centroid(i, random_features))
        else:
            random_centroids = test_centroids

        return random_centroids

    @staticmethod
    def nearest_centroid(point, centroids, distance):
        nearest = None
        min_distance = inf
        for centroid in centroids:
            current_distance = distance(point.get_coordinates(), centroid.get_coordinates())
            if current_distance < min_distance:
                min_distance, nearest = current_distance, centroid
        return nearest

    @staticmethod
    def assign_to_clusters(clusters, point, centroid):
        if centroid not in clusters:
            clusters[centroid] = [point]
        else:
            clusters[centroid].append(point)
    
    @staticmethod
    def relocate_centroids(clusters):
        new_clusters = {}
        for centroid in clusters.keys():
            average = np.array([point.get_coordinates() for point in clusters[centroid]]).mean(axis=0)
            new_centroid = Centroid(centroid.get_label(), {"x": average[0], "y": average[1]})
            new_clusters[new_centroid] = clusters[centroid]
        return new_clusters

    @staticmethod
    def fit(points, num_clusters, iterations, similarity):
        clusters = {}
        last_state = {}
        centroids = KMeans.generate_centroids(num_clusters, points, False)

        for i in range(iterations):
            is_last_iteration = i == iterations - 1

            for point in points:
                centroid = KMeans.nearest_centroid(point, centroids, similarity)
                KMeans.assign_to_clusters(clusters, point, centroid)

            should_terminate = is_last_iteration or clusters == last_state
            last_state = clusters
            if should_terminate:
                #print(f"terminating early: {i}")
                break

            centroids = KMeans.relocate_centroids(clusters)
            clusters = {}

        return last_state