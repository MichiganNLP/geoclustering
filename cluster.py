
from itertools import chain
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap as Basemap
import numpy
import operator
import os
from random import randint, uniform
from sklearn import metrics
import sklearn.cluster
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.cluster.mean_shift_ import estimate_bandwidth, MeanShift
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import sys
from time import time
import us

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import plotter
import pylab as pl
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster.spectral import SpectralClustering


def genColors(states):
    """
    Given a map of <ClusterID,State>, it generates a RGB map that corresponds to each cluster, so that they can be displayed on a map
    For example: (<1,TX>,<1,CA>,<2,NY>) might become (<(.04,04.09),TX>, <(.04,04.09),TX>, <(.03,00.41),NY>,
    """
    s = set()
    for val in states.values():
        s.add(val)
    colors = [0]*len(s)
    for i in range(0, len(s)):
        h = randint(0, 255) # Select random green'ish hue from hue wheel
        s = uniform(.6, 1)
        v = uniform(0.3, 1)

        r, g, b = hsv_to_rgb(h, s, v)
        colors[i] = (r,g,b)
    return colors

def hsv_to_rgb(h, s, v):
    """Converts HSV value to RGB values
    Hue is in range 0-359 (degrees), value/saturation are in range 0-1 (float)

    Direct implementation of:
    http://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_from_HSV_to_RGB
    """
    h, s, v = [float(x) for x in (h, s, v)]

    hi = (h / 60) % 6
    hi = int(round(hi))

    f = (h / 60) - (h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if hi == 0:
        return v, t, p
    elif hi == 1:
        return q, v, p
    elif hi == 2:
        return p, v, t
    elif hi == 3:
        return p, q, v
    elif hi == 4:
        return t, p, v
    elif hi == 5:
        return v, p, q

def getSaveFiles(dir):
    files = []
    while True:
            id = randint(0, 9999) # Select random green'ish hue from hue wheel
            newpath = os.path.join(dir, str(id))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                print("Saving results in" + newpath)
                files.append(os.path.join(newpath, 'cluster.png'))
                files.append(os.path.join(newpath, 'map.png'))
                return files


#Runs k-means clustering on the data from the linguistic ethnogrpahy tool
#results are saved in a folder called results/ in working_dir
def main():

    """CONFIGURATION"""
    num_clusters = 5; #Number of clusters
    random = False #If true, it will randomly assign clusters to the states w/ equal prob. If false, it will actually computer the clusters.
    working_dir = "/home/jmaxk/proj/geoloc/cluster/fb1/" #The input working_dir, which has 1 file per class. Each file contains the results of the linguistic ethnography tool

    """END CONFIGURATION"""

    if random:
        saveFiles = getSaveFiles(working_dir + 'results/random')
    else:
        saveFiles = getSaveFiles(working_dir + 'results/real')

    clusterFile = saveFiles[0]
    mapFile = saveFiles[1]
    featureIndeces = dict()
    classIndeces = []
    counter =0
    vecs = []


    #Turn each file into a vector to be clustered. Note
    for root, dirs, files in os.walk(working_dir):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.txt':
                with open(fullpath) as fp:
                    lines = fp.readlines()
                    vec = [0.0]*(len(lines) + 1)
                    for line in lines:
                        featVals = line.split(' ')
                        key = featVals[0]
                        val = featVals[1]
                        if not featureIndeces.has_key(key):
                            featureIndeces[key] = counter
                            counter = counter + 1
                        index = featureIndeces.get(key);
                        vec[index] = float(val)
                    vecs.append(vec)
                    abbr = os.path.basename(fullpath).split(".")[0]

                    #we only want to save actual states
                    if (us.states.lookup(abbr) != None):
                        st = (str(us.states.lookup(abbr).name))
                        classIndeces.append(st)

        #transform data into numpy array
        mylist = []
        for item in vecs:
            mylist.append(numpy.array(item))
        data = numpy.array(mylist)

        #cluster with kmeans, and save the clusters
        km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=10,
                verbose=False)
        raw_results =  km.fit_predict(data)
        results = dict(zip(classIndeces, raw_results))
        saveClusters(data,km, clusterFile) #this doesn't working_dir with random

#   save the map
    if random:
        random_results = dict()
        for key in results:
            random_results[key] = randint(0,5)
        colors = genColors(random_results)
        saveMap(random_results,colors, mapFile)
    else:
        colors = genColors(results)
        saveMap(results,colors, mapFile)



def saveClusters(mat, kmeans, cluster_file):
        reduced_data = PCA(n_components=2).fit_transform(mat)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
        y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plotter
        Z = Z.reshape(xx.shape)
        pl.figure(1)
        pl.clf()
        pl.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=pl.cm.get_cmap('Paired'),
                  aspect='auto', origin='lower')
        pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        pl.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)
        pl.title('K-means clustering on PCA reduced data')
        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())
        #pl.show()
        pl.savefig(cluster_file)

def saveMap(states, colorlist, map_file):
    plt.clf()
    current_dir=os.getcwd() + "/st99_d00"
    print current_dir 
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    shp_info = m.readshapefile(current_dir, 'states', drawbounds=True)
    colors = {}
    statenames = []
    cmap = cm.get_cmap('hot')
     # use 'hot' colormap
    vmin = 0; vmax = 450  # set range.
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['Puerto Rico']:
            cluster_id = states.get(statename)
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            #colors[statename] = cmap(1. - np.sqrt((pop - vmin) / (vmax - vmin)))[:3]
            colors[statename] = colorlist[cluster_id]
        statenames.append(statename)
    ax = plt.gca()  # get current axes instance
    for nshape, seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax.add_patch(poly)
    # draw meridians and parallels.
    m.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])
    plt.title('States with colored clusters')
    plt.savefig(map_file)

if __name__ == "__main__":
    main()
