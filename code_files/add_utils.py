import numpy as np
import pandas as pd


def probability(x1: pd.Series, x: pd.Series) -> float:
    """Probability Function, returns zero if complete 'x' is empty

    Args:
        x1 (pd.Series): Subset of x
        x (pd.Series):

    Returns:
        float: _description_
    """
    if x.size == 0:
        return 0
    else:
        return x1.size / x.size


def entropy(y: pd.Series) -> float:
    """Entropy - for discrete data

    Args:
        y (pd.Series): dtype = "category"

    Returns:
        float
    """
    classes = y.cat.categories
    counts = y.value_counts()
    ent = 0.0
    for cls in classes:
        count_for_cls = counts.get(cls, 0)
        if y.size > 0:
            prob = count_for_cls / y.size
        else:
            prob = 0.0
            return 0

        if prob > 0:
            ent += -prob * np.log2(prob)

    return ent


def MSE(y: pd.Series) -> float:
    """Mean Square Error - for real data

    Args:
        y (pd.Series): Real Data

    Returns:
        float
    """
    if y.size == 0:
        return 0
    else:
        m = y.mean()
        sum_of_square_errors = 0
        for y_i in y:
            sum_of_square_errors = sum_of_square_errors + np.square(y_i - m)

        return sum_of_square_errors / y.size


def info_gain_entropy_disc_disc(x: pd.Series, y: pd.Series) -> float:
    """Information gain using entropy for Discrete Input Discrete Output

    Args:
        x (pd.Series): dtype="category"
        y (pd.Series): dtype="category"

    Returns:
        float
    """
    S = entropy(y)
    classes_x = x.cat.categories
    df = pd.DataFrame({"x": x, "y": y})
    s = 0
    for cls in classes_x:
        df_temp = df[df["x"] == cls]
        prob = probability(df_temp["x"], x)
        # prob = (df_temp["x"].size) / (x.size)
        s = s + prob * entropy(df_temp["y"])

    return S - s


def info_gain_mse_disc_real(x: pd.Series, y: pd.Series) -> float:
    """Information gain using entropy for Discrete Input Real Output aka Mean Square Error Reduction

    Args:
        x (pd.Series): dtype="category"
        y (pd.Series): realdata

    Returns:
        float
    """
    x_cat = x.cat.categories
    df = pd.DataFrame({"x": x, "y": y})
    M = MSE(y)
    m = 0
    for cls in x_cat:
        df_temp = df[df["x"] == cls]
        prob = probability(df_temp["x"], x)
        # prob = (df_temp["x"].size) / (x.size)
        m = m + prob * MSE(df_temp["y"])

    return M - m


def info_gain_entropy_real_disc(x: pd.Series, y: pd.Series, index: int) -> float:
    """Information gain, computed using entropy, for the case of Real Input and Discrete Output, evaluated after splitting the sorted feature values at position x_sort[index] (in ascending order).

    Args:
        x (pd.Series): _description_
        y (pd.Series): dtype = "category"
        index (int): _description_

    Returns:
        float: _description_
    """
    S = entropy(y)
    df = pd.DataFrame({"x": x, "y": y})
    df_sort = df.sort_values(by=["x"], ascending=True)
    set1 = df_sort["y"].iloc[0: index + 1]
    set2 = df_sort["y"].iloc[index + 1:]
    prob1 = probability(set1, y)
    # prob1 = set1.size / y.size
    prob2 = probability(set2, y)
    # prob2 = set2.size / y.size
    w_entropy = prob1 * entropy(set1) + prob2 * entropy(set2)
    return S - w_entropy


def info_gain_mse_real_real(x: pd.Series, y: pd.Series, index: int) -> float:
    """Information gain, computed using MSE, for the case of Real Input and Real Output, evaluated after splitting the sorted feature values at position x_sort[index] (in ascending order).

    Args:
        x (pd.Series): _description_
        y (pd.Series): _description_
        index (int): _description_

    Returns:
        float: _description_
    """
    mse_y = MSE(y)
    df = pd.DataFrame({"x": x, "y": y})
    df_sort = df.sort_values(by="x", ascending=True)
    set1 = df_sort["y"].iloc[0: index + 1]
    prob1 = probability(set1, y)
    # prob1 = set1.size / y.size
    set2 = df_sort["y"].iloc[index + 1:]
    prob2 = probability(set2, y)
    # prob2 = set2.size / y.size

    w_mse = prob1 * MSE(set1) + prob2 * MSE(set2)

    return mse_y - w_mse



def best_split_real_basedon_entropy_real_disc(
    x: pd.Series, y: pd.Series
) -> tuple[int, float]:
    """Getting the best split for a given attribute using entropy

    Args:
        x (pd.Series): real
        y (pd.Series): dtype = "category"

    Returns:
        tuple[int, float]: index, info_gain
    """
    InfoGainarr = [info_gain_entropy_real_disc(x, y, i) for i in range(y.size)]
    return InfoGainarr.index(max(InfoGainarr)), max(InfoGainarr)


def best_split_real_basedon_mser_real_real(
    x: pd.Series, y: pd.Series
) -> tuple[int, float]:
    """Getting the best split for a given attribute using mser

    Args:
        x (pd.Series): real
        y (pd.Series): real

    Returns:
        tuple[int, float]: index, mser
    """
    MSERarr = [info_gain_mse_real_real(x, y, i) for i in range(y.size)]
    return MSERarr.index(max(MSERarr)), max(MSERarr)


def best_attr_disc_disc(X: pd.DataFrame, y: pd.Series) :
    """Best attribute is the one which gives maximum Information Gain using Entropy

    Args:
        X (pd.DataFrame): Discrete
        y (pd.Series): Discrete

    Returns:
        attribute
    """
    information_gain_array = []
    attr_array = []
    for attr in X.columns:
        information_gain_array.append(info_gain_entropy_disc_disc(X[attr], y))
        attr_array.append(attr)

    return attr_array[information_gain_array.index(max(information_gain_array))]

def best_attr_disc_real(X: pd.DataFrame, y: pd.Series):
    """Best attribute is the one which gives maximum Information Gain using MSER

    Args:
        X (pd.DataFrame): Discrete
        y (pd.Series): Real

    Returns:
        attribute
    """
    mser_array = []
    attr_array = []
    for attr in X.columns:
        mser_array.append(info_gain_mse_disc_real(X[attr], y))
        attr_array.append(attr)
        
    return attr_array[mser_array.index(max(mser_array))]


def best_attr_split_real_disc(X: pd.DataFrame, y: pd.Series) -> tuple[any, any]:
    """Best attribute is the one which gives maximum Information Gain using Entropy

    Args:
        X (pd.DataFrame): Real
        y (pd.Series): Discrete

    Returns:
        tuple[any, any]: attribute, index
    """
    InfoGainAttrArray = []
    split_array = []
    attr_array = []
    for attr in X.columns:
        InfoGainAttrArray.append(
            best_split_real_basedon_entropy_real_disc(X[attr], y)[1]
        )
        attr_array.append(attr)
        split_array.append(
            best_split_real_basedon_entropy_real_disc(X[attr], y)[0])

    return (
        attr_array[InfoGainAttrArray.index(max(InfoGainAttrArray))],
        split_array[InfoGainAttrArray.index(max(InfoGainAttrArray))],
    )


def best_attr_split_real_real(X: pd.DataFrame, y: pd.Series) -> tuple[any, any]:
    """Best attribute is the one which gives maximum Information Gain using MSER

    Args:
        X (pd.DataFrame): Real
        y (pd.Series): Real

    Returns:
        tuple[any, any]: attribute, index
    """
    MSERAttrArray = []
    split_array = []
    attr_array = []
    for attr in X.columns:
        MSERAttrArray.append(
            best_split_real_basedon_mser_real_real(X[attr], y)[1])
        attr_array.append(attr)
        split_array.append(
            best_split_real_basedon_mser_real_real(X[attr], y)[0])

    return (
        attr_array[MSERAttrArray.index(max(MSERAttrArray))],
        split_array[MSERAttrArray.index(max(MSERAttrArray))],
    )
