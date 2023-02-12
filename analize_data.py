import numpy as np

from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy.linalg as la

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import KernelPCA


def mapData(dist_matrix, X, y, images, metric, title, legend=None):
    mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)

    # Get the embeddings
    pts = mds.fit_transform(dist_matrix)

    # Print the stress value
    stress = mds.stress_
    print(f"stress value: {stress}")

    plotPCA(y, pts, images, title, legend)

    return pts, mds


def plotPCA(y, pts, images, title, legend=None):
    # Plot the embedding, colored according to the class of the points
    fig = plt.figure(2, (15, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1],
                         hue=y, palette=['r', 'g', 'b', 'c', 'violet', 'gold', 'lightcoral', 'chocolate', 'k', 'y'])

    plt.title('Scatter plot')
    ax.set(xlabel="1st PCo", ylabel="2nd PCo")

    if not (legend is None):
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles, legend)

    # Add the second plot
    ax = fig.add_subplot(1, 2, 2)
    ax.set(xlabel="1st PCo", ylabel="2nd PCo")
    # Plot the points again
    plt.scatter(pts[:, 0], pts[:, 1])

    # Annotate each point by its corresponding image
    for x, ind in zip(images, range(pts.shape[0])):
        imagebox = OffsetImage(x, zoom=0.3, cmap=plt.cm.gray)
        i = pts[ind, 0]
        j = pts[ind, 1]
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)
    plt.suptitle(title)
    ax.set_title('Scatter with images')
    plt.show()


def PCAmapData(X, y, images, kernel='rbf', title='PCA', legend=None):
    # mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)
    pca = KernelPCA(n_components=None, kernel=kernel)

    # Get the embeddings
    pts = pca.fit_transform(X)

    # Print the stress value
    # stress = mds.stress_
    # print(f"stress value: {stress}")

    plotPCA(y, pts, images, title, legend)

    return pts, pca


def compute_eigenvalues(A):
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

    # # select top 2 dimensions (for example)
    # PC1 = np.sqrt(eigen_val[0]) * eigen_vec[0]
    # PC2 = np.sqrt(eigen_val[1]) * eigen_vec[1]

    return eigen_val


def KPCA_explained_ratios(PCA):
    values = PCA.eigenvalues_

    rel_eigen_val = values / values.sum(dtype=float)

    return rel_eigen_val


def MDS_plot_ratios(A, n=20):
    eigen_val = compute_eigenvalues(A)

    rel_eigen_val = eigen_val / eigen_val.sum(dtype=float)
    first_values = rel_eigen_val[0:n]

    plot_ratios(first_values)


# def PCA_plot_ratios(explained_variance):
#
#
#     rel_eigen_val = eigenvalues / eigenvalues.sum(dtype=float) * 100
#
#     plot_ratios(rel_eigen_val)


def plot_ratios(values):
    values = values * 100

    n = values.shape[0]
    range = np.arange(1, n + 1)
    x_ticks = np.arange(1, n + 1, step=int(n / 10))

    plt.bar(range, height=values)
    plt.xticks(x_ticks, x_ticks)
    plt.title('Variazione per componente')
    plt.xlabel('Componenti')
    plt.ylabel('% della variazione')
    plt.show()

    cum_sum = np.cumsum(values)

    ax = plt.bar(range, height=cum_sum)
    plt.xticks(x_ticks, x_ticks)
    plt.title('Variazione cumulativa')
    plt.xlabel('Componenti')
    plt.ylabel('% della variazione')
    plt.text(1, 90, f'{float(cum_sum[-1])} %')
    plt.show()


# trova il numero ottimale di componenti pca
def optimal_components_n(dist_matrix, goal_perc=99):
    eigen_val = compute_eigenvalues(dist_matrix)
    rel_eigen_val = eigen_val / eigen_val.sum(dtype=float) * 100
    cum_sum = np.cumsum(rel_eigen_val)
    num_pca = 0
    for i in range(cum_sum.shape[0]):
        if cum_sum[i] >= goal_perc:
            num_pca += 1
            print("The optimal number of PCA's is: {}".format(num_pca))
            break
        else:
            num_pca += 1
            continue

    return num_pca

    # Trova


def KPCA_oplimal_components(X, goal_perc=99, kernel='rbf'):
    pca = KernelPCA(n_components=None, kernel=kernel)

    # Get the embeddings
    pts = pca.fit_transform(X)
    ratios = KPCA_explained_ratios(pca)
    cum_ratios = (np.cumsum(ratios))

    num_pca = 0
    num_found = False

    for i in range(cum_ratios.shape[0]):
        if cum_ratios[i] * 100 >= goal_perc:
            num_pca += 1
            print("The optimal number of PCA's is: {}".format(num_pca))
            break
        else:
            num_pca += 1
            continue

    return num_pca, pca, pts


def analize_data(descriptors_file='data.npy', output_file='data_extracted.npy',extraction_file='extraction_data.npy', draw=True):  # TODO correggi xticks grafici
    data = np.load(descriptors_file, allow_pickle=True)
    scaler = StandardScaler()
    X = data[0]
    Y = data[1]
    scaler.fit(X)
    X = scaler.transform(X)

    file_names = data[2]
    category_legend = data[3][0]
    print(category_legend)

    images = data[4]

    if draw:
        dist_euclid = euclidean_distances(X)

        _, _ = mapData(dist_euclid, X, Y, images, True,
                       'MDS', legend=category_legend)

        _, _ = PCAmapData(X, Y, images=images, legend=category_legend)

        optimal_n = optimal_components_n(dist_euclid, 99)

    pca_optimal_n, pca, pts = KPCA_oplimal_components(X, goal_perc=99)

    if draw:
        MDS_plot_ratios(dist_euclid, n=optimal_n)
        plot_ratios(KPCA_explained_ratios(pca)[0:pca_optimal_n])
    if not (__name__ == '__main__'):
        pts = pts[:, 0:pca_optimal_n]
        X = pts
        data[0] = X
        #data.resize(data.shape[0]+1)
        #data[-1] = (scaler, pca, pca_optimal_n)
        extraction_data = np.empty(1, dtype=object)
        extraction_data[0] = (scaler, pca, pca_optimal_n)
        np.save(extraction_file,extraction_data)
        np.save(output_file, data)
        print('Feature extracted with PCA and saved to file')

    # plot_stress_values(dist_euclid,True, max_range=optimal_n)

    return scaler, pca, pca_optimal_n


if __name__ == '__main__':
    analize_data('data.npy')
