from scipy.fft import fft, fftfreq
from wettbewerb import load_references
import math
import features_112
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np

ecg_leads,ecg_labels,fs,ecg_names = load_references()
print('Messungen wurden geladen!')

plt.plot(range(0, ecg_leads[1].size), ecg_leads[1])
plt.show()
print('Ungefilterte Daten wurden verarbeitet!')
a = ecg_leads[1]
b = ecg_leads[3]
measurements = ecg_leads[2]
print(ecg_leads[1].size)
print(a)
print(b)
print(ecg_leads[3].size)
test = np.array([[a],[b]])
#Initialisierung eines Kalman-Filters
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
#Schätzung der Parameter des Kalman-Filters
print('Parameter für Kalman-Filter werden gefittet!')
c = kf.em(measurements)
print('Parameter für Kalman-Filter wurden gefittet!')
(filtered_state_means, filtered_state_covariances) = c.filter(ecg_leads[3])
(smoothed_state_means, smoothed_state_covariances) = c.smooth(ecg_leads[3])
print(filtered_state_means.size)

figure, axis = plt.subplots(2, 2)
  
# For Sine Function
axis[0, 0].plot(range(0, filtered_state_means.size), filtered_state_means)
axis[0, 0].set_title("filtered")
  
# For Cosine Function
axis[0, 1].plot(range(0, ecg_leads[3].size), ecg_leads[3])
axis[0, 1].set_title("normal")
  
# For Tangent Function
axis[1, 0].plot(range(0, smoothed_state_means.size), smoothed_state_means)
axis[1, 0].set_title("smoothend")

# Combine all the operations and display
plt.show()
print('#')
print('#')
print('#')
