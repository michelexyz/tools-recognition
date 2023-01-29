import numpy as np

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy.linalg as la

def mapData(dist_matrix, X, y, images, metric, title):
    mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)


    # Get the embeddings
    pts = mds.fit_transform(dist_matrix)

    # Print the stress value
    stress = mds.stress_
    print(f"stress value: {stress}")

    # Plot the embedding, colored according to the class of the points
    fig = plt.figure(2, (15, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1],
                         hue=y, palette=['r', 'g', 'b', 'c','violet','gold','lightcoral','chocolate','k','y'])

    # Add the second plot
    ax = fig.add_subplot(1, 2, 2)
    # Plot the points again
    plt.scatter(pts[:, 0], pts[:, 1])

    # Annotate each point by its corresponding image
    for x, ind in zip(images, range(pts.shape[0])):

        imagebox = OffsetImage(x, zoom=0.3, cmap=plt.cm.gray)
        i = pts[ind, 0]
        j = pts[ind, 1]
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)
    plt.title(title)
    plt.show()


def plot_stress_values(dist_matrix, metric):
    stress = []
    # Max value for n_components
    max_range = 21
    for dim in range(1, max_range):
        # Set up the MDS object
        mds = MDS(n_components=dim, metric= False, dissimilarity='precomputed', random_state=0, normalized_stress=True)
        # Apply MDS
        pts = mds.fit_transform(dist_matrix)
        # Retrieve the stress value
        stress.append(mds.stress_)
        # Plot stress vs. n_components
    plt.plot(range(1, max_range), stress)
    plt.xticks(range(1, max_range, 2))

    plt.xlabel('n_components')
    plt.ylabel('stress')
    plt.show()

def plot_eigenvalues(A):
    # square it
    A = A ** 2

    # centering matrix
    n = A.shape[0]
    J_c = 1. / n * (np.eye(n) - 1 + (n - 1) * np.eye(n))

    # perform double centering
    B = -0.5 * (J_c.dot(A)).dot(J_c)

    # find eigenvalues and eigenvectors
    eigen_val = la.eig(B)[0]
    eigen_vec = la.eig(B)[1].T

    # select top 2 dimensions (for example)
    PC1 = np.sqrt(eigen_val[0]) * eigen_vec[0]
    PC2 = np.sqrt(eigen_val[1]) * eigen_vec[1]

    new_n = 20
    rel_eigen_val = eigen_val / eigen_val.sum(dtype=float)*100
    first_values = rel_eigen_val[0:new_n]

    range = np.arange(1, new_n+1)
    plt.bar(range, height=first_values)
    plt.xticks(range, range);
    plt.title('Variazione per componente')
    plt.xlabel('Componenti')
    plt.ylabel('% della variazione')
    plt.show()

    cum_sum = np.cumsum(first_values)

    ax =plt.bar(range, height=cum_sum)
    plt.xticks(range, range);
    plt.title('Variazione cumulativa')
    plt.xlabel('Componenti')
    plt.ylabel('% della variazione')
    plt.text(1,90,f'{float(cum_sum[-1])} %')
    plt.show()

    # Trova

def analize_data(descriptors_file='data.npy'):
    data = np.load(descriptors_file, allow_pickle=True)


    X = data[0]
    Y = data[1]
    file_names = data[2]
    category_legend = data[3]
    images = data[4]

    dist_euclid = euclidean_distances(X)
    mapData(dist_euclid, X, Y, images, True,
            'Metric MDS with Euclidean')

    plot_eigenvalues(dist_euclid)
    plot_stress_values(dist_euclid,True)



analize_data()