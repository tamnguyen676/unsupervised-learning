import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, completeness_score, homogeneity_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

from sklearn.tree import DecisionTreeClassifier


def get_clusters_em(X, components=2):
    em = GaussianMixture(n_components=components)
    em.fit(X)
    clusters = em.predict(X)

    return clusters

    diff = Y - clusters
    res = np.sum(np.abs(diff))
    accuracy = max(res / len(X), 1 - res / len(X))
    if components == 2:
        print(f'EM with 2 components results in {accuracy} accuracy if used to classify')

def get_clusters_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    return clusters

    diff = Y - clusters
    res = np.sum(np.abs(diff))
    accuracy = max(res / len(X), 1 - res / len(X))

    if n_clusters == 2:
        print(f'K means with 2 clusters results in {accuracy} accuracy if used to classify')

def elbow_method(X, dataset):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(2, 25)

    for k in K:
        # Building and fitting the model
        model = KMeans(n_clusters=k)
        model.fit(X)

        distortions.append(sum(np.min(cdist(X, model.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(model.inertia_)

        mapping1[k] = sum(np.min(cdist(X, model.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = model.inertia_

    print('Distortion')
    for key, val in mapping1.items():
        print(str(key) + ' : ' + str(val))

    xlabel = 'Values of K'
    plt.plot(K, distortions, 'bx-')
    plt.xticks(K)
    plt.xlabel(xlabel)
    plt.ylabel('Distortion')
    plt.title(f'Distortion vs {xlabel}')
    plt.savefig(f'{dataset} Distortion {xlabel}')
    plt.clf()


    print('Inertia')
    for key, val in mapping2.items():
        print(str(key) + ' : ' + str(val))

    plt.plot(K, inertias, 'bx-')
    plt.xticks(K)

    plt.xlabel(xlabel)
    plt.ylabel('Inertia')
    plt.title(f'Inertia vs {xlabel}')
    plt.savefig(f'{dataset} Inertia {xlabel}')
    plt.clf()

def get_colored_silhouette_plot(X, use_kmeans, dataset):
    range_n_clusters = [2, 3, 4, 5, 6] if use_kmeans else range(2, 30)

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10) if use_kmeans else GaussianMixture(n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, .8, 1])

        plt.suptitle(("Silhouette analysis"
                      "n_clusters = %d" % n_clusters),
                     fontsize=20, fontweight='bold')

        plt.savefig(f'{dataset} Silhouette Analysis {n_clusters} {use_kmeans}')


def get_completeness_plot(X, Y_truth, dataset, use_kmeans=True):

    k_range = range(2, 20)
    completeness_scores = []
    for i in k_range:
        clusters = get_clusters_kmeans(X, i) if use_kmeans else get_clusters_em(X, i)
        completeness_scores.append(completeness_score(Y_truth, clusters))
    plt.plot(k_range, completeness_scores, '-x')
    plt.title(f'{dataset} Completeness Plot')
    plt.xlabel('Value of K' if use_kmeans else 'Number of Components')
    plt.ylabel('Completeness Score')
    plt.savefig(f'{dataset} completeness plot {use_kmeans}')
    plt.clf()


def get_silhouette_plot(X, use_kmeans, dataset):
    scores = []
    k_range = range(2, 45) if '1' in dataset else range(2, 15)
    print('getting silhouette plot')
    highest_score = argmax = 0
    for i in k_range:
        print(i)
        model = KMeans(n_clusters=i) if use_kmeans else GaussianMixture(n_components=i)
        model.fit(X)
        cluster_labels = model.predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        scores.append(silhouette_avg)

        if silhouette_avg > highest_score:
            highest_score = silhouette_avg
            argmax = i

    xlabel = 'Values of K' if use_kmeans else 'Number of Components'

    plt.plot(k_range, scores, '-x')
    plt.title(f'Average Silhouette Score vs {xlabel} {dataset}')
    plt.xlabel(xlabel)
    plt.ylabel('Average Silhouette Score')

    plt.savefig(f'{dataset} Silhouette {xlabel}')
    plt.clf()

    return highest_score, argmax

def get_pca_plot(X, Y, dataset):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(Y, columns=['target'])], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(f'2 component PCA {dataset}', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(f'{dataset} PCA Clusters')
    plt.clf()


def get_pca_scree_plot(X, dataset):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=6)
    pca.fit_transform(X)
    pc_list = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

    plt.bar(pc_list, pca.explained_variance_ratio_)
    pca.explained_variance_
    print(pca.explained_variance_ratio_)
    plt.title(f'Scree Plot {dataset}')
    plt.xlabel('Principle Components')
    plt.ylabel('Explained Variance Ratio')
    plt.savefig(f'{dataset} PCA Scree Plot')
    plt.clf()

def randomized_projection(X, dataset):
    components_range = range(1, 10) if dataset == 'Dataset 2' else range(2, 170, 2)

    reconstruction_error = []
    times = []
    for i in components_range:
        transformer = GaussianRandomProjection(n_components=i)

        start = time.time()
        X_transformed = transformer.fit_transform(X)
        end = time.time()
        times.append(end - start)

        inverse_data = np.linalg.pinv(transformer.components_.T)
        X_projected = X_transformed.dot(inverse_data)
        loss = ((X - X_projected) ** 2).mean().mean()
        reconstruction_error.append(loss)

    plt.plot(components_range, reconstruction_error, '-o')
    plt.title(f'Reconstruction Error vs Number of Components {dataset}')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.savefig(f'{dataset} Randomized Projection Reconstruction Error')
    plt.clf()

    plt.plot(components_range, times, '-o')
    plt.title(f'Time to Fit vs Number of Components {dataset}')
    plt.xlabel('Number of Components')
    plt.ylabel('Time to Fit (s)')
    plt.savefig(f'{dataset} Randomized Projection Time to fit')
    plt.clf()


def get_ica_plot(X, dataset):
    kurtosis_list = []
    reconstruction_error = []
    times = []

    kurtosis_range = range(2, 170, 2) if dataset == 'Dataset 1' else range(2, 10)

    for i in kurtosis_range:
        print(i)
        transformer = FastICA(n_components=i, random_state=0, max_iter=500)

        start = time.time()
        X_transformed = transformer.fit_transform(X)
        end = time.time()
        times.append(end - start)

        kurt = kurtosis(X_transformed)
        avg_kurtosis = np.mean(np.abs(kurt))
        kurtosis_list.append(avg_kurtosis)
        X_projected = transformer.inverse_transform(X_transformed)
        squared_error = (X - X_projected) ** 2
        loss = squared_error.mean().mean()
        reconstruction_error.append(loss)

    plt.plot(kurtosis_range, kurtosis_list, '-o')
    plt.title(f'Avg Kurtosis vs Number of Components {dataset}')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Kurtosis')
    plt.savefig(f'Kurtosis {dataset}')
    plt.clf()

    plt.plot(kurtosis_range, reconstruction_error, '-o')
    plt.title(f'Reconstruction Error vs Number of Components {dataset}')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.savefig(f'Reconstruction Error {dataset}')
    plt.clf()

    plt.plot(kurtosis_range, times, '-o')
    plt.title(f'Time to Fit vs Number of Components {dataset}')
    plt.xlabel('Number of Components')
    plt.ylabel('Time to Fit (s)')
    plt.savefig(f'Time to Fit {dataset}')
    plt.clf()

    print(max(range(len(kurtosis_list)), key=lambda i: kurtosis_list[i]))


def get_rfe_plot(X_train, Y_train, dataset):
    estimator = DecisionTreeClassifier(min_samples_split=14, max_depth=48, criterion='entropy', random_state=1)
    # estimator = AdaBoostClassifier(n_estimators=148, learning_rate=0.35346938775510206, algorithm='SAMME')
    # The "accuracy" scoring is proportional to the number of correct
    # classifget_ica_plottions
    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(3),
                  scoring='accuracy', verbose=True)
    rfecv.fit(X_train, Y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.title(f'RFE CV Score vs Number of Features Selected {dataset}')
    plt.xlabel("Number of features selected")
    plt.ylabel("CV Score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, '-o')
    plt.savefig(f'RFE CV {dataset}')
    plt.clf()


def get_plots(X_train, Y_train, dataset):
    elbow_method(X_train, dataset)
    get_silhouette_plot(X_train, False, dataset)
    get_silhouette_plot(X_train, True, dataset)
    get_colored_silhouette_plot(X_train, False, dataset)
    get_colored_silhouette_plot(X_train, True, dataset)
    get_completeness_plot(X_train, Y_train, dataset, False)
    get_completeness_plot(X_train, Y_train, dataset, True)
    get_pca_scree_plot(X_train, dataset)
    get_pca_plot(X_train, Y_train, dataset)
    get_ica_plot(X_train, dataset)
    randomized_projection(X_train, dataset)
    get_rfe_plot(X_train, Y_train, dataset)


def get_reduced_pca(X, num_components):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=num_components, random_state=0)
    return pca.fit_transform(X)


def get_reduced_ica(X, num_components):
    transformer = FastICA(n_components=num_components, random_state=0, max_iter=500)
    return transformer.fit_transform(X)


def get_reduced_rp(X, num_components):
    transformer = GaussianRandomProjection(n_components=num_components, random_state=0)
    return transformer.fit_transform(X)


def get_reduced_rfe(X, cols_to_drop):
    return X.drop(columns=X.columns[np.where(cols_to_drop != 1)])


def get_reduced_k_means(X, k):
    clusters = get_clusters_kmeans(X, k)
    df = pd.DataFrame(data={'clusters': clusters})
    return df

def get_reduced_em(X, k):
    clusters = get_clusters_em(X, k)
    df = pd.DataFrame(data={'clusters': clusters})
    return df

def optimize_model(X_train, Y_train, name):

    options = {
        'hidden_layer_sizes': range(2, 100, 2),
        'activation': ['logistic', 'tanh', 'relu']
    }

    random_search = RandomizedSearchCV(MLPClassifier(), cv=5, param_distributions=options, n_jobs=-1,
                                       n_iter=20, verbose=True, return_train_score=True)
    random_search.fit(X_train, Y_train)

    df = pd.DataFrame(random_search.cv_results_)
    print(f'Random Search Optimization results for {name}')
    argmax = df['mean_test_score'].argmax()
    print('Highest Value ', df['mean_test_score'][argmax])
    print('Best Param', df['params'][argmax])

    return df['mean_test_score'][argmax], df['params'][argmax]


def test_model(X_train, X_test, Y_train, Y_test, name, options):
    print(name)

    clf = MLPClassifier(**options)
    start = time.time()
    clf.fit(X_train, Y_train)
    end = time.time()
    train_time = end - start
    print(f'Train time: {end - start}')

    start = time.time()
    y_predict = clf.predict(X_test)
    end = time.time()
    query_time = end - start
    print(f'Query time: {end - start}')

    accuracy = accuracy_score(Y_test, y_predict, normalize=True)
    print(f'Accuracy: {accuracy}')
    print()

    return train_time, query_time, accuracy


def run_models(X, Y, dataset, k_pca, k_ica, k_rp, k_rfe, k_kmeans, k_em):

    algs = ['PCA', 'ICA', 'RP', 'RFE', 'K MEANS', 'EM', 'None']

    train_times = []
    query_times = []
    accuracies = []

    pca_reduced_X = get_reduced_pca(X, k_pca)
    X_train, X_test, Y_train, Y_test = train_test_split(pca_reduced_X, Y, train_size=.9, random_state=123)
    # _, options = optimize_model(X_train, Y_train, f'pca {dataset}')
    options = {'hidden_layer_sizes': 92, 'activation': 'tanh'}
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'pca {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    ica_reduced_X = get_reduced_ica(X, k_ica)
    X_train, X_test, Y_train, Y_test = train_test_split(ica_reduced_X, Y, train_size=.9, random_state=123)
    # options = {'hidden_layer_sizes': 2, 'activation': 'tanh'}
    _, options = optimize_model(X_train, Y_train, f'ica {dataset}')
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'ica {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    rp_reduced_X = get_reduced_rp(X, k_rp)
    X_train, X_test, Y_train, Y_test = train_test_split(rp_reduced_X, Y, train_size=.9, random_state=123)
    options = {'hidden_layer_sizes': 38, 'activation': 'logistic'}
    _, options = optimize_model(X_train, Y_train, f'rp {dataset}')
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'rp {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    rfe_reduced_X = get_reduced_rfe(X, k_rfe)
    X_train, X_test, Y_train, Y_test = train_test_split(rfe_reduced_X, Y, train_size=.9, random_state=123)
    _, options = optimize_model(X_train, Y_train, f'rfe {dataset}')
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'ica {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    kmeans_reduced_x = get_reduced_k_means(X, k_kmeans)
    X_train, X_test, Y_train, Y_test = train_test_split(kmeans_reduced_x, Y, train_size=.9, random_state=123)
    _, options = optimize_model(X_train, Y_train, f'kmeans {dataset}')
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'kmeans {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    em_reduced_x = get_reduced_em(X, k_em)
    X_train, X_test, Y_train, Y_test = train_test_split(em_reduced_x, Y, train_size=.9, random_state=123)
    _, options = optimize_model(X_train, Y_train, f'em {dataset}')
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'em {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    options1 = {'learning_rate': 'invscaling', 'hidden_layer_sizes': 5, 'activation': 'relu'}
    options2 = {'learning_rate': 'adaptive', 'hidden_layer_sizes': 90, 'activation': 'logistic'}

    options = options1 if dataset == 'Dataset 1' else options2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.9, random_state=123)
    train_time, query_time, accuracy = test_model(X_train, X_test, Y_train, Y_test, f'em {dataset}', options)
    train_times.append(train_time)
    query_times.append(query_time)
    accuracies.append(accuracy)

    plt.bar(algs, train_times)
    plt.title(f'Train Times {dataset}')
    plt.ylabel('Train Time (s)')
    plt.savefig(f'Train Times {dataset}')
    plt.clf()

    plt.bar(algs, query_times)
    plt.title(f'Query Times {dataset}')
    plt.ylabel('Query Time (s)')
    plt.savefig(f'Query Times {dataset}')
    plt.clf()

    plt.bar(algs, accuracies)
    plt.title(f'Test Set Accuracy {dataset}')
    plt.ylabel('Accuracy')
    plt.savefig(f'Test Set Accuracy {dataset}')
    plt.clf()


def cluster_reductions(X, dataset, k_pca, k_ica, k_rp, k_rfe, k_kmeans, k_em):
    print('running cluster reduction')
    algos = {
        # 'PCA': get_reduced_pca(X, k_pca),
        # 'ICA': get_reduced_ica(X, k_ica),
        'RP': get_reduced_rp(X, k_rp),
        # 'RFE': get_reduced_rfe(X, k_rfe)
    }

    for algo, X in algos.items():
        # score, argmax = get_silhouette_plot(X, True, f'{dataset} {algo}')
        # print(f'K means {algo}, k = {argmax}')
        score, argmax = get_silhouette_plot(X, False, f'{dataset} {algo}')
        print(f'EM {algo}, k = {argmax}')



def run_dataset_1():
    print('Running dataset 1 (fight data)')
    data = pd.read_csv('data/fight-data.csv')
    X, Y = data.drop(columns='Winner'), data['Winner']
    Y = np.copy(Y)
    Y[Y == 'Red'] = 0.
    Y[Y == 'Blue'] = 1.
    Y = Y.astype(float)
    dataset = 'Dataset 1'


    # get_plots(X_train, Y_train, dataset)
    filtered_cols = [124, 48, 60, 75, 74, 61, 53, 43, 1, 28, 1, 86, 1, 14, 1, 44, 94, 1,
                     33, 46, 1, 102, 35, 50, 1, 78, 36, 1, 1, 121, 20, 1, 32, 13, 38, 1,
                     1, 1, 90, 1, 15, 6, 92, 84, 64, 9, 25, 62, 4, 1, 59, 70, 72, 29,
                     1, 55, 49, 10, 100, 108, 23, 68, 16, 104, 65, 18, 31, 51, 47, 73, 95, 34,
                     1, 109, 1, 42, 3, 116, 1, 122, 58, 40, 17, 1, 11, 56, 45, 26, 1, 1,
                     1, 52, 1, 1, 1, 12, 63, 1, 27, 1, 80, 21, 2, 41, 1, 37, 30, 5,
                     1, 82, 22, 24, 1, 1, 1, 93, 57, 83, 8, 7, 106, 1, 110, 19, 85, 103,
                     105, 101, 107, 119, 79, 118, 39, 117, 1, 1, 97, 99, 89, 88, 91, 98, 87, 81,
                     96, 77, 76, 67, 69, 71, 66, 54, 111, 112, 113, 114, 115, 120, 123]

    run_models(X, Y, dataset, 3, 128, 149, filtered_cols, 2, 18)
    cluster_reductions(X, dataset, 3, 128, 149, filtered_cols, 2, 18)


def run_dataset_2():
    print('Running dataset 2 (vehicle MPG data)')

    data = pd.read_csv('data/auto-mpg.csv')
    X, Y = data.drop(columns=['mpg', 'car name']), data['mpg']  # car name and horsepower cause crash
    Y = np.copy(Y)
    Y[Y >= 25] = 1
    Y[Y != True] = 0

    dataset = 'Dataset 2'

    get_plots(X, Y, dataset)

    run_models(X, Y, dataset, 2, 3, 6, [3, 1, 1, 1, 1, 1, 2], 2, 3)
    cluster_reductions(X, dataset, 2, 3, 6, [3, 1, 1, 1, 1, 1, 2], 2, 3)


run_dataset_2()
run_dataset_1()



