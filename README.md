# KMeans Clustering In Python

## Usage

```
points = []

for point in test_points:
    new_point = Point(point["name"])
    new_point.set_features(point["features"])
    points.append(new_point)

clusters = KMeans.fit(points, 3, 4, euclidean_distance)
```

### KMeans.py

**Example**

```
KMeans.fit(points: Point[], num_clusters: Int, iterations: Int, similarity: func (a, b))
```

**Inputs**

points: list of Points with n features

num_clusters: number of clusters you want to create

iterations: max number of iterations you want to run

similarity: function that returns the similarity between 2 points

**Outputs**

clusters containing list of points

### Point.py

**Example**

```
new_point = Point(label)
new_point.set_features(features)
```

Input to KMeans.fit

#### Methods

```
get_label() # Returns point label
```

```
set_features(features: {"feature":value})
```

```
get_features() # Returns dict of feature vector
```

```
get_sorted_features() # Returns dict of features from highest to lowest
```

```
get_coordinates() # Returns list of values for each feature without label
```

### Centroid.py

**Example**

```
newCentroid = Centroid("label", initial_features)
```

Used to generate initial centroids for KMeans

#### Methods

```
get_label() # Returns point label
```

```
set_features(features: {"feature":value})
```

```
get_features() # Returns dict of feature vector
```

```
get_sorted_features() # Returns dict of features from highest to lowest
```

```
get_coordinates() # Returns list of values for each feature without label
```

### Distance.py

**Example**

```
from src.Distance import euclidean_distance
clusters = KMeans.fit(points, 3, 4, euclidean_distance)

from src.Distance import cosine_similarity
clusters = KMeans.fit(points, 3, 4, cosine_similarity)
```

Calculates similarity between point and centroid

#### Methods

```
euclidean_distance(a, b) # Returns sqrt(a^2 + b^2)
```

```
cosine_similarity(a, b) # Returns a*b / |a| * |b|
```

### Clustering

- KMeans clustering

  - Generates initial 3 clusters by randomly generating doubles between 0 and the max for each feature across the feature set

- KMeans++ Clustering
  - Selects 3 random documents from each dataset as the initial clusters
