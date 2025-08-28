"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from tree.add_utils import *


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    pass


def check_ifreal(y: pd.Series) -> bool:
    """
    Check if target y is discrete (classification) or real (regression).
    """
    # Object/categorical
    if pd.api.types.is_categorical_dtype(y) or y.dtype == object:
        return "False"

    # Numeric: Very few unique values relative to length ->classification(Tune always)
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = y.nunique()
        if unique_vals <= 5:
            return "False"
        else:
            return "True"

    return "False"


#Not Using
def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    pass


def information_gain(Y: pd.Series, attr: pd.Series, index: int, criterion: str) -> float:
    """Calculate the information gain using criterion (entropy or MSE)

    Args:
        Y (pd.Series): Output
        attr (pd.Series): Any one of the attribute
        index (int): Split at index for real attributes
        criterion (str): {DD,DR,RD,RR} D-> Discrete R->Real

    Returns:
       float: information_gain
    """
    if criterion not in {"DD", "DR" ,"RD", "RR"}:
        raise ValueError(f"Invalid criterion '{criterion}'. Choose 'DD' or 'DR' or 'RD' or 'RR'.")
    
    if criterion == "DD":
        return info_gain_entropy_disc_disc(attr, Y)
    
    if criterion == "DR":
        return info_gain_mse_disc_real(attr, Y)
    
    if criterion == "RD":
        return info_gain_entropy_real_disc(attr, Y, index)
    
    if criterion == "RR":
        return info_gain_mse_real_real(attr, Y, index)



#Not Required
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


#Not Required
def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass


def best_attr_split(X: pd.DataFrame, y: pd.Series, criterion:str) -> tuple[any,any]:
    """Return best attr and best split(if required)

    Args:
        X (pd.DataFrame): DataFrame
        y (pd.Series): Output
        criterion (str): {DD,DR,RD,RR} D-> Discrete R->Real

    Returns:
        tuple[any,any]: attr, split
    """
    if criterion not in {"DD", "DR" ,"RD", "RR"}:
        raise ValueError(f"Invalid criterion '{criterion}'. Choose 'DD' or 'DR' or 'RD' or 'RR'.")
    
    if criterion == "DD":
        return best_attr_disc_disc(X, y), "NoSplit"
    
    if criterion == "DR":
        return best_attr_disc_real(X, y), "NoSplit"
    
    if criterion == "RD":
        return best_attr_split_real_disc(X, y)
    
    if criterion == "RR":
        return best_attr_split_real_real(X, y)
    

