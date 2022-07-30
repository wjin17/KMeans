from random import random, shuffle, choice
from src.KMeans import KMeans
from src.Point import Point
import matplotlib.pyplot as plt
import numpy as np
from src.Distance import euclidean_distance

# points = []
# #features = {x: int, y: int}
# for i in range(3):
#     for j in range(5):
#         new_point = Point(f'{i}:{j}')
#         new_point.set_features({"x": random() + i, "y": random() + i})
#         points.append(new_point)

# shuffle(points)
test_points = [
    {"name": "0:3", "features": {'x': 0.8134111128420548, 'y': 0.9317979028082289}},
    {"name": "0:1", "features": {'x': 0.9388977291334022, 'y': 0.26760564001055476}},
    {"name": "2:2", "features": {'x': 2.3221638658604937, 'y': 2.0543408725270833}},
    {"name": "1:3", "features": {'x': 1.2936559309270137, 'y': 1.2621546133481392}},
    {"name": "1:0", "features": {'x': 1.2854683897538406, 'y': 1.5861322619190865}},
    {"name": "1:1", "features": {'x': 1.4573312444782367, 'y': 1.1547732910678175}}, 
    {"name": "1:2", "features": {'x': 1.0833472170994891, 'y': 1.0357415845191953}},
    {"name": "2:3", "features": {'x': 2.066619929737822, 'y': 2.095052596725555}},
    {"name": "0:4", "features": {'x': 0.8861707005963045, 'y': 0.33742424312075237}},
    {"name": "1:4", "features": {'x': 1.926570852867097, 'y': 1.1725190652235242}},
    {"name": "0:0", "features": {'x': 0.4406653574600551, 'y': 0.2650805902509631}},
    {"name": "0:2", "features": {'x': 0.1262628496886694, 'y': 0.5112860238320611}},
    {"name": "2:1", "features": {'x': 2.1804809343163156, 'y': 2.748267538995033}},
    {"name": "2:4", "features": {'x': 2.9089321703937183, 'y': 2.99761597822963}},
    {"name": "2:0", "features": {'x': 2.5337216597875, 'y': 2.7729695023403127}}
]

points = []

for point in test_points:
    new_point = Point(point["name"])
    new_point.set_features(point["features"])
    points.append(new_point)

clusters = KMeans.fit(points, 3, 4, euclidean_distance)

i = 0
for centroid in clusters:
    colors = ["#424b54", "#a53860", "#00a5e0", "#e2b4bd", "#9b6a6c", "#61c9a8"]
    print(f"\nCluster: {centroid.get_label()} {centroid.get_features()}")
    centroid_x = [centroid.get_coordinates()[0]]
    centroid_y = [centroid.get_coordinates()[1]]

    plt.scatter(centroid_x, centroid_y, color = colors[i])

    for point in clusters[centroid]:
        print(f"Point: {point.get_label()} {point.get_features()}")
    coords = np.array([point.get_coordinates() for point in clusters[centroid]])
    xpoints = coords[:,0]
    ypoints = coords[:,1]
    plt.scatter(xpoints, ypoints, color=colors[i], alpha=0.8)
    
    i += 1

plt.show()
