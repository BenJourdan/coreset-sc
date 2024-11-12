# test_coreset_sc.py
# test

import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy
from coreset_sc import CoresetSpectralClustering, gen_sbm

# from maturin_import_hook.settings import MaturinSettings
# import maturin_import_hook
# maturin_import_hook.install(
# enable_project_importer=True,
# enable_rs_file_importer=True,
# settings=MaturinSettings(
#     release=True,
#     strip=True,
# ),
# show_warnings=False,
# )
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", message="KMeans is known to have a memory leak on Windows with MKL"
)


def test_default_coreset():
    """Test the default coreset sampler"""
    resolution = 2
    rounds = 1

    ks = numpy.linspace(10, 20, resolution, dtype=int)

    construction_times = numpy.zeros((resolution, rounds))
    coreset_time = numpy.zeros((resolution, rounds))
    labelling_time = numpy.zeros((resolution, rounds))

    coreset_aris = numpy.zeros((resolution, rounds))
    full_aris = numpy.zeros((resolution, rounds))

    for i, k in tqdm(list(enumerate(ks))):
        k = int(k)
        for r in range(rounds):
            # Generate some data
            n = 1000
            p = 0.5
            q_factor = 1.0
            q = (q_factor / n) / k
            coreset_fraction = 0.1
            # Generate the data
            t0 = time.time()
            A, y = gen_sbm(n, k, p, q)
            construction_times[i, r] = time.time() - t0
            # Sample, extract, cluster and label the coreset graph
            t0 = time.time()
            csc = CoresetSpectralClustering(
                num_clusters=k, coreset_ratio=coreset_fraction
            )
            csc.fit(A)
            coreset_time[i, r] = time.time() - t0
            # Get the coreset ARI
            coreset_aris[i, r] = (
                adjusted_rand_score(y[csc.coreset_indices_], csc.coreset_labels_)
                if coreset_fraction < 1.0
                else 0.0
            )
            # label the full graph
            t0 = time.time()
            csc.label_full_graph()
            labelling_time[i, r] = time.time() - t0
            # Get the full ARI
            full_aris[i, r] = adjusted_rand_score(y, csc.labels_)
            if coreset_fraction == 1.0:
                coreset_aris[i, r] = full_aris[i, r]
            assert full_aris[i, r] > 0.5, f"ARI {full_aris[i, r]} is below 0.9"

    # save the data to a json file
    data_dict = {
        "construction_times": construction_times.tolist(),
        "coreset_time": coreset_time.tolist(),
        "labelling_time": labelling_time.tolist(),
        "coreset_aris": coreset_aris.tolist(),
        "full_aris": full_aris.tolist(),
        "ks": ks.tolist(),
        "rounds": rounds,
        "title": f"SBM with N=1000k, p={p}, q={q_factor}/nk, coreset_fraction={coreset_fraction*100}%",
    }

    with open("test_default_coreset.json", "w") as f:
        json.dump(data_dict, f)


def plot_coreset_test():

    # Load the data
    with open("test_default_coreset.json", "r") as f:
        data_dict = json.load(f)

    construction_times = numpy.array(data_dict["construction_times"])
    coreset_time = numpy.array(data_dict["coreset_time"])
    labelling_time = numpy.array(data_dict["labelling_time"])
    coreset_aris = numpy.array(data_dict["coreset_aris"])
    full_aris = numpy.array(data_dict["full_aris"])
    ks = numpy.array(data_dict["ks"])
    rounds = data_dict["rounds"]
    title = data_dict["title"]

    # plot results on two subplots.

    # On the first subplot, plot the construction time and stack the corset time under the full labelling time
    # On the second subplot, plot the coreset and full ARI

    fontsize = 20

    # Means:
    construction_time_means = construction_times.mean(axis=1)
    coreset_time_means = coreset_time.mean(axis=1)
    labelling_time_means = labelling_time.mean(axis=1)
    coreset_ari_means = coreset_aris.mean(axis=1)
    full_ari_means = full_aris.mean(axis=1)

    # Std errors:
    construction_time_se = construction_times.std(axis=1) / numpy.sqrt(rounds)
    coreset_time_se = coreset_time.std(axis=1) / numpy.sqrt(rounds)
    labelling_time_se = labelling_time.std(axis=1) / numpy.sqrt(rounds)
    coreset_ari_se = coreset_aris.std(axis=1) / numpy.sqrt(rounds)
    full_ari_se = full_aris.std(axis=1) / numpy.sqrt(rounds)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # first subplot
    ax[0].plot(ks, construction_times.mean(axis=1), label="Construction time")
    # Stack the mean coreset time under the mean full labelling time

    ax[0].fill_between(
        ks,
        0.0,
        coreset_time_means,
        label="coreset labelling time",
        color="orange",
        alpha=0.5,
    )
    ax[0].fill_between(
        ks,
        coreset_time_means,
        coreset_time_means + labelling_time_means,
        label="Full labelling time",
        color="green",
        alpha=0.5,
    )

    # error bars:
    ax[0].errorbar(
        ks,
        construction_time_means,
        yerr=construction_time_se,
        fmt="none",
        ecolor="black",
    )
    ax[0].errorbar(
        ks,
        coreset_time_means,
        yerr=coreset_time_se,
        fmt="none",
        ecolor="black",
    )
    ax[0].errorbar(
        ks,
        coreset_time_means + labelling_time_means,
        yerr=labelling_time_se,
        fmt="none",
        ecolor="black",
    )

    ax[0].set_yscale("log")

    ax[0].tick_params(axis="both", which="major", labelsize=fontsize)

    ax[0].set_xlabel("k", fontsize=fontsize)
    ax[0].set_ylabel("Time (s)", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize)

    # second subplot

    ax[1].plot(ks, coreset_aris.mean(axis=1), label="Coreset ARI")
    ax[1].plot(ks, full_aris.mean(axis=1), label="Full ARI")

    # error bars:
    ax[1].errorbar(
        ks,
        coreset_ari_means,
        yerr=coreset_ari_se,
        fmt="none",
        ecolor="black",
    )

    ax[1].errorbar(
        ks,
        full_ari_means,
        yerr=full_ari_se,
        fmt="none",
        ecolor="black",
    )

    ax[1].tick_params(axis="both", which="major", labelsize=fontsize)

    ax[1].set_xlabel("k", fontsize=fontsize)
    ax[1].set_ylabel("ARI", fontsize=fontsize)

    ax[1].legend(fontsize=fontsize)
    fig.suptitle(title, fontsize=20)

    fig.tight_layout()
    plt.savefig("test_default_coreset.png")


def test_sbm():
    """
    Test the stochastic block model generator
    """
    import time

    import coreset_sc
    import networkx as nx

    n = 20
    k = 5
    p = 0.5
    q = (1.0 / n) / k

    print(n, k, p, q)

    t = time.time()
    A, labels = coreset_sc.gen_sbm(n, k, p, q)
    print(f"Time: {time.time()-t}")

    # convert to networkx graph
    G = nx.from_scipy_sparse_array(A)

    # remove all self loops:
    G.remove_edges_from(nx.selfloop_edges(G))

    # draw and save the graph
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    nx.draw(G)
    plt.savefig("test_sbm.png")
    # print pwd:


# def test_both_sbms():

#     import time

#     import coreset_sc

#     n = 5000
#     k = 50
#     p = 0.1
#     q = (1.0 / n) / k

#     t0 = time.time()
#     _fast_sbm = coreset_sc.gen_sbm(n, k, p, q)
#     fast_time = time.time() - t0

#     t0 = time.time()
#     _slow_sbm = coreset_sc.stag_sbm(n, k, p, q)
#     slow_time = time.time() - t0

#     print(f"fast time: {fast_time:.5f}s\t slow time: {slow_time:.5f}s")

#     raise Exception("stop")


if __name__ == "__main__":
    test_default_coreset()
    plot_coreset_test()
