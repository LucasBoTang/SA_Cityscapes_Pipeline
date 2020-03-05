#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
import cv2
import networkx as nx
import numpy as np

def solve(graph, lambd=0.03, psi=1, phi=0.02, stop=None, propagation=True, attr="feat"):
    """
    gradient minimization algorithm
    based on paper "Fast and Effective L0 Gradient Minimization by Region Fusion"
    the algorithm is modified to deal with scribble label
    """
    iter = 0
    gamma = 2.2
    iterations = 50
    print("Running heuristic gradient...")

    # first stage: gradually increase lambda
    # ========================================================================
    # region fusion until 200 iterations
    for iter in range(1000):

        # terminate when all pixels are labelled
        if not check_none_label(graph):
            break

        # increase beta for each iteration
        beta = (iter / iterations) ** gamma * lambd
        # region fusion
        region_fuse(graph, beta, psi, phi, propagation, attr)

        if iter and iter % iterations == 0:
            print("{} iterations: {} groups of pixels".format(iter, len(graph)))

        # stop on fisrt stage
        if stop and iter == stop:
            return graph

    # second stage: assign unlabelled node
    # ========================================================================
    # merge neighbors with same label
    for i in list(graph.nodes):
        # skip contracted node
        if i not in graph.nodes:
            continue
        # if the group has label, contract the neighbor with same label
        for j in list(graph.neighbors(i)):
            # get label of groups
            label = graph.nodes[i]["label"]
            if graph.nodes[j]["label"] == label:
                graph.contract(j, i)
                i = j

    # get rid off inner unlabelled
    for i in list(graph.nodes):
        # skip contracted node
        if i not in graph.nodes:
            continue
        if len(list(graph.neighbors(i))) == 1 and not graph.nodes[i]["label"]:
            j = list(graph.neighbors(i))[0]
            graph.contract(j, i)
        # skip contracted node
        if i not in graph.nodes:
            continue

    # assign unlabelled to similiar
    for i in list(graph.nodes):
        # skip contracted node
        if i not in graph.nodes:
            continue
        if not graph.nodes[i]["label"]:
            color = graph.nodes[i]["mean_color"]
            inds = []
            colors = []
            neighors = list(graph.neighbors(i))
            for j in neighors:
                inds.append(j)
                colors.append(graph.nodes[j]["mean_color"])
            colors = np.array(colors)
            # get min l1 distance
            k = np.argmin(np.sum(np.abs(colors - color)))
            j = neighors[k]
            graph.contract(i, j)

    if iter % 100 != 0:
        print("{} iterations: {} groups of pixels".format(iter+1, len(graph)))

    # show labels
    labels = []
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        labels.append(label)
    # print("Labels includes:", ", ".join(labels))

    return graph


def scribble_center_solve(graph, lambd=0.03):
    """
    old version heuristic in ILP paper
    """

    # search labelled nodes
    roots = {}
    for i in list(graph.nodes):
        # skip if i has been contracted
        if not i:
            continue
        label = graph.nodes[i]["label"]
        # skip unlabelled node
        if not label:
            continue
        if label not in roots:
            roots[label] = i
        else:
            root = roots[label]
            graph.contract(root, i)

    iter = 0
    lambd = 0.1
    gamma = 2.2
    iterations = 50
    print("Running heuristic gradient...")

    while len(graph) > len(roots):
        # increase beta for each iteration
        beta = (iter / iterations) ** gamma * lambd

        for root in roots.values():
            # flag if root is merged
            merged = True

            while merged:
                merged = False

                # get attributes of root
                wi = graph.nodes[root]["weight"]
                Yi = graph.nodes[root]["mean_color"]

                # loop through neighbors
                for j in list(graph.neighbors(root)):

                    # get attributes of j
                    wj = graph.nodes[j]["weight"]
                    Yj = graph.nodes[j]["mean_color"]
                    # get attributes of (i, j)
                    cij = graph.edges[root, j]["connections"]

                    # modification: skip when labels conflict
                    label_i = graph.nodes[root]["label"]
                    label_j = graph.nodes[j]["label"]

                    # avoid conflication
                    if label_i and label_j and label_i != label_j:
                        continue

                    # fusion criterion
                    if wi * wj * (np.linalg.norm(Yi - Yj) ** 2) <= beta * cij * (wi + wj):
                        # contract nodes
                        graph.contract(root, j)
                        merged = True
        iter += 1
    return graph

def region_fuse(graph, beta, psi, phi, propagation=True, attr="feat"):
    """
    fuse regions with current auxiliary parameter beta
    """
    # loop through all nodes
    for i in list(graph.nodes):

        # skip contracted node
        if i not in graph.nodes:
            continue

        # get attributes of i
        wi = graph.nodes[i]["weight"]
        Yi = graph.nodes[i]["mean_color"]
        Zi = graph.nodes[i][attr]

        # loop through neighbors
        for j in list(graph.neighbors(i)):

            # get attributes of j
            wj = graph.nodes[j]["weight"]
            Yj = graph.nodes[j]["mean_color"]
            Zj = graph.nodes[j][attr]
            # get attributes of (i, j)
            cij = graph.edges[i, j]["connections"]

            # modification: skip when labels conflict
            label_i = graph.nodes[i]["label"]
            label_j = graph.nodes[j]["label"]
            # aviod label propagation
            if not propagation:
                if label_i != label_j:
                    continue
            # make a label propagation
            else:
                #if label_i and label_j and label_i != label_j:
                if label_i and label_j and label_i != label_j:
                    continue

            # fusion criterion
            # print(np.linalg.norm(Yi - Yj) ** 2 / np.linalg.norm(Zi - Zj) ** 2)
            if wi * wj * (psi * np.linalg.norm(Yi - Yj) ** 2 + phi * np.linalg.norm(Zi - Zj) ** 2)<= beta * cij * (wi + wj):
                # contract nodes
                graph.contract(j, i)
                i = j

    return graph


def check_none_label(graph):
    """
    check if the graph has None label
    """
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        if not label:
            return True
    return False
