import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def custumize_colormap(colors):
    values = range(len(colors))
    value_max = np.ceil(np.max(values))
    
    colors_val = []
    for value in values:
        colors_val.append(value/value_max)
    
    color_name = []
    for color in colors:
        color_name.append(color)
    
    colors_list = []
    for i in range(len(colors)):
        colors_list.append([colors_val[i], color_name[i]])
    #print(colors_list)
    #for value, color in zip(values, colors):
    #    colors_list.append([value/value_max, color])
    #print(colors_list)
    
    
    return LinearSegmentedColormap.from_list('custumized_cmap', colors_list)



#custumize_colormap(['mediumblue', 'orangered'])

