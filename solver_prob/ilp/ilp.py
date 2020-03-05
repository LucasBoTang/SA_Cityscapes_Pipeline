#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
import cv2
import networkx as nx
import numpy as np
import cplex
from .callback import connectivityCallback
from utils.data import class_info

def build_model(graph, lambd):
    """
    build mip model from graph
    """
    # get label map
    label_map = get_label_map(graph)

    # initialize model
    ilp = cplex.Cplex()

    # set mip gap
    ilp.parameters.mip.tolerances.mipgap.set(1.0e-2)
    # emphasis
    #ilp.parameters.emphasis.mip.set(1)

    # set sense
    ilp.objective.set_sense(ilp.objective.sense.minimize)

    # objective funtion
    colnames, obj, types = get_obj(graph, lambd, label_map)
    ilp.variables.add(obj=obj, types=types, names=colnames)

    # constraints
    rows, senses, rhs = get_constraints(graph, label_map)
    ilp.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs)

    # parallel
    ilp.parameters.parallel.set(-1)
    ilp.parameters.threads.set(32)

    # register callback
    ilp.register_callback(connectivityCallback)

    # associate additional data
    connectivityCallback._graph = graph.copy()
    connectivityCallback._names = ilp.variables.get_names()[:len(graph)*len(label_map)]
    connectivityCallback._label_map = label_map

    print("Original MIP has {} rows and {} columns.".format(len(rows), len(colnames)))

    return ilp


def get_label_map(graph):
    """
    get label map for calculate unary cost
    """
    # collect information of labels
    labels = {}
    for i in graph.nodes:
        label = graph.nodes[i]["label"]
        # skip unlabelled
        if not label:
            continue
        label_id = class_info[label.split("_")[0]].id
        if label_id in labels:
            labels[label_id].append(i)
        else:
            labels[label_id] = [i]

    label_map = {}
    Label = namedtuple("Label", ["root", "nodes", "label", "color", "feat"])
    for l in labels:
        nodes = labels[l]
        root = nodes[0]
        label = graph.nodes[root]["label"]
        color = np.zeros_like(graph.nodes[root]["mean_color"])
        feat = np.zeros_like(graph.nodes[root]["feat"])

        cnt = 0
        for i in nodes:
            color += graph.nodes[i]["mean_color"]
            feat += graph.nodes[i]["feat"]
            cnt += 1
        label_map[l] = Label(root, nodes, label, color/cnt, feat/cnt)

    return label_map


def get_obj(graph, lambd, label_map):
    """
    get columns name and coefficients of objective function
    get types of variables
    """
    # unary part
    u_colnames = []
    u_obj = []
    u_types = ""
    # interate nodes
    for i in graph.nodes:
        # weight of superpixel node
        wi = graph.nodes[i]["weight"]
        # interate label
        for l in label_map:
            # variable name
            u_colnames.append("x_" + str(i) + "_" + str(l))
            # coefficient
            fi = graph.nodes[i]["feat"]
            Yl = np.zeros((19,))
            Yl[l%19] =  1
            cil = np.linalg.norm(fi - Yl) ** 2
            #print(cil)
            u_obj.append(wi * cil)
            # variable type
            u_types += "B"

    # pairwise part
    p_colnames = []
    p_obj = []
    p_types = ""
    # interate edges
    for i, j in graph.edges:
        # weight of superpixel edge
        wij = graph.edges[(i, j)]["connections"]
        # pairwise penalty
        fi = graph.nodes[i]["mean_color"]
        fj = graph.nodes[j]["mean_color"]
        pij = np.exp(- np.linalg.norm(fi - fj) ** 2)
        # interate label
        for l in label_map:
            # variable name
            name = "e_" + str(i) + "_" + str(j) + "_" + str(l)
            p_colnames += [name+"+", name+"-"]
            # coefficient
            coef = lambd * wij * pij
            #print(wij * pij)
            p_obj += [coef, coef]
            # variable type
            p_types += "CC"

    # concatenate
    colnames = u_colnames + p_colnames
    obj = u_obj + p_obj
    types = u_types + p_types

    return colnames, obj, types


def get_constraints(graph, label_map):
    """
    get constraints
    """
    rows = []
    rhs = []
    senses = ""

    # hard assignment
    # interate nodes
    for i in graph.nodes:
        vars, coefs = [], []
        # interate label
        for l in label_map:
            # variable name
            vars.append("x_" + str(i) + "_" + str(l))
            # coefficient
            coefs.append(1)
        # add constraint
        assert len(vars) == len(coefs)
        rows.append([vars, coefs])
        rhs.append(1)
        senses += "E"

    # absolute value
    # interate nodes
    for i, j in graph.edges:
        # interate label
        for l in label_map:
            vars = ["x_" + str(i) + "_" + str(l), \
                    "x_" + str(j) + "_" + str(l), \
                    "e_" + str(i) + "_" + str(j) + "_" + str(l) + "+", \
                    "e_" + str(i) + "_" + str(j) + "_" + str(l) + "-"]
            coefs = [1, -1, -1, 1]
            # add constraint
            assert len(vars) == len(coefs)
            rows.append([vars, coefs])
            rhs.append(0)
            senses += "E"

    # label fixation
    # interate label
    cnt = 0
    for l in label_map:
        nodes = label_map[l].nodes
        for i in nodes:
            vars = ["x_" + str(i) + "_" + str(l)]
            coefs = [1]
            # add constraint
            assert len(vars) == len(coefs)
            rows.append([vars, coefs])
            rhs.append(1)
            senses += "E"

    return rows, senses, rhs


def warm_start(ilp, pred, superpixels):
    """
    start from heuristic solution
    """
    print("Adding MIP start from heuristic...")

    # get all labels in current image
    labels = []
    for name in ilp.variables.get_names():
        l = int(name.split("_")[-1])
        # avoid duplication
        if l in labels:
            break
        labels.append(l)

    h, w = pred.shape
    added = set()
    names = []
    vars = []
    for y in range(h):
        for x in range(w):
            superpixel = superpixels[y, x]
            # skip if superpixel is not selected
            if not superpixel:
                continue
            # skip if superpixel is added
            if superpixel in added:
                continue
            added.add(superpixel)
            for l in labels:
                name = "x_" + str(superpixel) + "_" + str(l)
                names.append(name)
                # decide binary state
                if l == pred[y, x]:
                    vars.append(1)
                else:
                    vars.append(0)

    assert len(names) == len(vars)
    ilp.MIP_starts.add(cplex.SparsePair(ind=names, val=vars), ilp.MIP_starts.effort_level.repair, "heuristic")
