import numpy as np
import sklearn.metrics as sk


def _eval_cls(y_pred, y_true):
    """
    Compute the precision, recall and confusion matrix in one-VS-rest mode for each class.

    Parameters
    ----------
    y_true : np array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : np array-like of shape (n_samples,)
        pred label.

    Returns
    -------
    precision_dict: dict(class_id, class_precision)
    recall_dict: dict(class_id, class_recall)
    cm_dict: dict(class_id, class_confusion_matrix)

    Examples
    --------
    y_pred = [0, 2, 1, 0, 1, 2]
    y_true = [0, 0, 1, 1, 2, 2]
    precisions, recalls, cms = self._eval_cls(y_pred, y_true)
    # precisions = {0: 0.5, 1: 0.5, 2: 0.5}
    # recalls = {0: 0.5, 1: 0.5, 2: 0.5}
    # cms = {0: [[1 1], [1 3]], 1: [[1 1], [1 3]], 2: [[1 1], [1 3]]}
    """

    precision_dict, recall_dict, cm_dict = {}, {}, {}
    y_label = np.unique(y_true).tolist()

    for class_id in y_label:
        binary_y_true = y_true.copy()
        class_mask = y_true == class_id
        binary_y_true[class_mask] = 1
        binary_y_true[~class_mask] = 0
        binary_y_pred = y_pred.copy()
        class_mask = y_pred == class_id
        binary_y_pred[class_mask] = 1
        binary_y_pred[~class_mask] = 0

        precision = sk.precision_score(binary_y_true, binary_y_pred)
        recall = sk.recall_score(binary_y_true, binary_y_pred)
        tn, fp, fn, tp = sk.confusion_matrix(binary_y_true, binary_y_pred).ravel()
        cm = np.array([[tp, fn], [fp, tn]])

        precision_dict[class_id] = precision
        recall_dict[class_id] = recall
        cm_dict[class_id] = cm

    return precision_dict, recall_dict, cm_dict


if __name__ == '__main__':
    y_pred = np.array([0, 2, 1, 0, 1, 2])
    y_true = np.array([0, 0, 1, 1, 2, 2])
    precisions, recalls, cms = _eval_cls(y_pred, y_true)
