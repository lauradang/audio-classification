"""The functions that plot data in the visualization notebook."""

import matplotlib.pyplot as plt

def plot_data(data, title, cmap=None, interpolation=None):
    """
    Plots the given data.

    Arguments:
        data {dict} -- A dictionary with keys that contain the classes and values with numpy arrays.
        title {str} -- The title of the plot.

    Keyword Arguments:
        cmap {str} -- Defines the cmap type of the plot (default: {None}).
        interpolation {str} -- Defines the interpolation type of the plot (default: {None}).
    """
    num_rows = 2
    num_columns = 5

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(20, num_columns))
    fig.suptitle(title, size=16)
    class_idx = 0

    for x in range(num_rows):
        for y in range(num_columns):
            axes[x,y].set_title(list(data.keys())[class_idx])
            values = list(data.values())[class_idx]
            
            if cmap:
                axes[x,y].imshow(values, cmap=cmap, interpolation=interpolation)
            else:
                if len(values) == 2:
                    Y, freq = values[0], values[1]
                    axes[x,y].plot(freq, Y)
                else:
                    axes[x,y].plot(values)

            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            class_idx += 1
            