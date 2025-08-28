"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from tree.utils import *
from tree.add_utils import *

np.random.seed(42)


class Node:
    def __init__(
        self,
        attribute=None,
        threshold=None,
        children=None,
        left=None,
        right=None,
        value=None,
        samples=None,     # number of samples in this node
        entropy=None,     # entropy (classification only)
        mse=None,         # mean squared error (regression only)
        output=None       # distribution of y values at this node
    ):
        self.attribute = attribute      # str: feature name
        self.threshold = threshold      # float: split value (real features)
        self.children = children
        self.left = left                # Node: left child (<= threshold)
        self.right = right              # Node: right child (> threshold)
        self.value = value
        self.samples = samples          # number of samples in this node
        self.entropy = entropy          # entropy for classification
        self.mse = mse                  # MSE for regression


@dataclass
class DecisionTree:
    # criterion won't be used for regression
    criterion: Literal["DD", "DR", "RD", "RR"]
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5, save_plot=True, plot_name="decision_tree"):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.save_plot = save_plot
        self.plot_name = plot_name

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:

        if self.criterion == "DD":
            if depth >= self.max_depth or len(y.unique()) == 1 or X.empty:
                return Node(
                    value=y.mode()[0],
                    samples=len(y),
                    entropy=entropy(y),
                )

            attr, _ = best_attr_split(X, y, "DD")
            if attr is None:

                return Node(
                    value=y.mode()[0],
                    samples=len(y),
                    entropy=entropy(y),
                )

            children = {}
            for val in X[attr].unique():
                mask = X[attr] == val

                X_child = X[mask].drop(columns=[attr])
                y_child = y[mask]
                children[val] = self._build_tree(X_child, y_child, depth + 1)

            return Node(
                attribute=attr,
                children=children,
                samples=len(y),
                entropy=entropy(y),
            )

        elif self.criterion == "DR":

            if depth >= self.max_depth or len(y.unique()) == 1 or X.empty:
                return Node(
                    value=y.mean(),
                    samples=len(y),
                    mse=MSE(y),
                )

            attr, _ = best_attr_split(X, y, "DR")
            if attr is None:

                return Node(
                    value=y.mean(),
                    samples=len(y),
                    mse=MSE(y),
                )

            children = {}
            for val in X[attr].unique():
                mask = X[attr] == val

                X_child = X[mask].drop(columns=[attr])
                y_child = y[mask]
                children[val] = self._build_tree(X_child, y_child, depth + 1)

            return Node(
                attribute=attr,
                children=children,
                samples=len(y),
                mse=MSE(y),
            )
        elif self.criterion == "RD":

            if depth >= self.max_depth or len(y.unique()) == 1 or X.empty:
                return Node(
                    value=y.mode()[0],
                    samples=len(y),
                    entropy=entropy(y),
                )

            attr, split_index = best_attr_split(X, y, "RD")
            if attr is None or split_index is None:
                return Node(
                    value=y.mode()[0],
                    samples=len(y),
                    entropy=entropy(y)
                )

            sorted_vals = sorted(X[attr].unique())

            if split_index + 1 < len(sorted_vals):
                threshold = (sorted_vals[split_index] +
                             sorted_vals[split_index + 1]) / 2
            else:
                threshold = sorted_vals[split_index]
            
            # if len(sorted_vals) == 1:
            #     return Node(
            #         value=y.mode()[0],
            #         samples=len(y),
            #         entropy=entropy(y)
            #     )
            # else:
            #     split_index = min(split_index, len(sorted_vals) - 2)
            #     threshold = (sorted_vals[split_index] + sorted_vals[split_index + 1]) / 2

            left_mask = X[attr] <= threshold
            right_mask = X[attr] > threshold

            left_child = self._build_tree(
                X[left_mask], y[left_mask], depth + 1)
            right_child = self._build_tree(
                X[right_mask], y[right_mask], depth + 1)

            return Node(
                attribute=attr,
                threshold=threshold,
                left=left_child,
                right=right_child,
                samples=len(y),
                entropy=entropy(y)
            )
        elif self.criterion == "RR":
            if depth >= self.max_depth or len(y.unique()) == 1 or X.empty:
                return Node(
                    value=y.mean(),
                    samples=len(y),
                    mse=MSE(y),
                )

            attr, split_index = best_attr_split(X, y, "RR")
            if attr is None or split_index is None:
                return Node(
                    value=y.mean(),
                    samples=len(y),
                    mse=MSE(y)
                )

            sorted_vals = sorted(X[attr].unique())

            if split_index + 1 < len(sorted_vals):
                threshold = (sorted_vals[split_index] +
                             sorted_vals[split_index + 1]) / 2
            else:
                threshold = sorted_vals[split_index]
            
            # if len(sorted_vals) == 1:
            #     return Node(
            #         value=y.mean(),
            #         samples=len(y),
            #         mse=MSE(y)
            #     )
            # else:
            #     split_index = min(split_index, len(sorted_vals) - 2)
            #     threshold = (sorted_vals[split_index] + sorted_vals[split_index + 1]) / 2


            left_mask = X[attr] <= threshold
            right_mask = X[attr] > threshold

            left_child = self._build_tree(
                X[left_mask], y[left_mask], depth + 1)
            right_child = self._build_tree(
                X[right_mask], y[right_mask], depth + 1)

            return Node(
                attribute=attr,
                threshold=threshold,
                left=left_child,
                right=right_child,
                samples=len(y),
                mse=MSE(y)
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        self.root = self._build_tree(X, y, depth=0)
        if self.save_plot:
            dot = self.plot()
            dot.render(self.plot_name, format="pdf", cleanup=True)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        def traverse(node, row):
            # leaf → return stored value
            if node.value is not None and node.children is None and node.left is None and node.right is None:
                return node.value

            # Categorical split
            if node.children is not None:
                val = row[node.attribute]
                if val in node.children:
                    return traverse(node.children[val], row)
                else:
                    return node.value

            # Numerical split
            if node.threshold is not None:
                if row[node.attribute] <= node.threshold:
                    return traverse(node.left, row)
                else:
                    return traverse(node.right, row)
            # CatchCase
            return Node.value 
    
        return X.apply(lambda row: traverse(self.root, row), axis=1)

    def plot(self) -> None:
        dot = Digraph()

        def add_nodes_edges(node, parent=None, edge_label=""):
            if node is None:
                return

            if node.value is not None and (node.left is None and node.right is None and node.children is None):
                # leaf node
                colour = "lightblue"  # Leaf nodes
                label = f"Leaf\nvalue={node.value:.3f}\nsamples={node.samples}"
                if hasattr(node, "entropy") and node.entropy is not None:
                    label += f"\nentropy={node.entropy:.3f}"
                if hasattr(node, "mse") and node.mse is not None:
                    label += f"\nmse={node.mse:.3f}"

            else:
                # decision node
                if node.threshold is not None:
                    label = f"{node.attribute} <= {node.threshold:.3f}\nsamples={node.samples}"
                    if hasattr(node, "entropy") and node.entropy is not None:
                        label += f"\nentropy={node.entropy:.3f}"
                    if hasattr(node, "mse") and node.mse is not None:
                        label += f"\nmse={node.mse:.3f}"
                elif node.children is not None:
                    label = f"{node.attribute}\nsamples={node.samples}"
                    if hasattr(node, "entropy") and node.entropy is not None:
                        label += f"\nentropy={node.entropy:.3f}"
                    if hasattr(node, "mse") and node.mse is not None:
                        label += f"\nmse={node.mse:.3f}"

                colour = "lightyellow"

            node_id = str(id(node))
            dot.node(node_id, label, style="filled", fillcolor=colour)

            if parent is not None:
                dot.edge(parent, node_id, label=edge_label)
            if node.children is not None:  # categorical splits
                for val, child in node.children.items():
                    add_nodes_edges(child, node_id, str(val))
            else:  # numerical splits
                if node.left is not None:
                    add_nodes_edges(node.left, node_id, "Yes")
                if node.right is not None:
                    add_nodes_edges(node.right, node_id, "No")

        add_nodes_edges(self.root)
        return dot
