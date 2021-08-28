# cortical-layers
Automatically detect the boundaries between cortical layers of MICrONS
segmentation data.

It is available through pip

`pip install cortical-layers`

This package provides interface for easily determining the layer of a
given list of points.

```
from cortical_layers.LayerPredictor import LayerClassifier

points_nm = ...  # soma locations, synapse locations, etc. 
c = LayerClassifier(data="minnie65_phase3")  # an aligned volume name
layers = c.predict(points_nm)  # np.array(["L4", "L23", "WM", ...]) 
```

For more detailed information about the layer boundaries for the
specified aligned volume, look at the LayerPrediction object in the
classifier, `c.pred`, which has has several attributes that will be
useful.

`LayerPredictor.BoundaryPredictor` has the capability to generate new
layer boundary predictions using a Hidden Markov Model for a given
dataset, provided the necessary features are available.
