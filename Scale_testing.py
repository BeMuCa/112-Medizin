import numpy as np
from wettbewerb import load_references
from features_112 import features
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Wir nehmen die verteilung von 
# (0). testen ob der array werte <1 hat weil sonst log trafo negative werte rausspuckt, wir wollen aber 
#
#
#
#
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale










### Log trafo für                       skewed data         : 

#import numpy as np
#
#def log_transform(data):
#    # replace all negative or zero values with a small positive number to avoid math errors
#    #data[data <= 0] = np.min(data[data > 0])  -- klappt net für listen stuff - aber nötig zu testen
#    log_transformed = np.log(data)
#    return log_transformed
#
## Example usage
#skewed_data = np.array([0.1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#log_transformed = log_transform(skewed_data)
#
#print("Skewed data: ", skewed_data)
#print("Log transformed data: ", log_transformed)
