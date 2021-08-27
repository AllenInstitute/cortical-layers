# cortical-layers
Automatically detect the boundaries between cortical layers of MICrONS
segmentation data.

Available through pip

`pip install cortical-layers`

This package provides interface for easily determining the layer of a
given list of points.



```
from cortical_layers.LayerPredictor import LayerClassifier

points_nm = ...  # soma locations, synapse locations, etc. 
c = LayerClassifier(data="minnie65_phase3") 
layers = c.predict(points_nm)  # np.array(["L4", "L23", "WM", ...]) 
```

It also has the capability to generate new layer boundary predictions
using a Hidden Markov Model for a given dataset, provided the features
are available.
