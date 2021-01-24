import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# import pandas as pd

x_values = [2, 4, 8, 16, 32]
y_axis_labels = ['GMM', 'WNLSTM']

wnlstm_values = [87.35, 74.68, 71.17, 58.32, 57.06]
gmm_values = [76.22,
              68.32,
              64.78,
              56.18,
              47.98]
dummy_values = [100.0 / class_no for class_no in x_values]

# x_new = np.linspace(1, 100, 60)
# fit = np.polyfit(x_values, wnlstm_values, 3)
# fit_fn = np.poly1d(fit)
# polyval = np.polyval(fit_fn, x_new)

patches = [mpatches.Patch(color='blue', label='WNLSTM'),
           mpatches.Patch(color='red', label='GMM'),
           mpatches.Patch(color='green', label='Baseline')
           ]

plt.legend(handles=patches)
plt.xlabel("Number of singers")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy on test data")
plt.plot(x_values, wnlstm_values, '-bo')
plt.plot(x_values, gmm_values, '-ro')
plt.plot(x_values, dummy_values, '-go')
plt.show()
