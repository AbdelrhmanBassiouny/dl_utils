import copy
import numpy as np
import matplotlib.pyplot as plt

def get_common_idxs(a, b, a_not_in_b=False):
    common_idxs = []
    if a_not_in_b:
        idxs_of_a_not_in_b = np.ones_like(a, dtype=bool)
    for i, a_val in enumerate(a):
        for j, b_val in enumerate(b):
            if a_val == b_val:
                common_idxs.append(a_val)
                if a_not_in_b:
                    idxs_of_a_not_in_b[i] = False
    if a_not_in_b:
        return common_idxs, a[idxs_of_a_not_in_b]
    else:
        return common_idxs


def calc_metrics(preds, labels, class_idx):
    pred_p_idxs = np.flatnonzero(preds == class_idx)
    act_p_idxs = np.flatnonzero(labels == class_idx)

    pred_n_idxs = np.flatnonzero(preds != class_idx)
    act_n_idxs = np.flatnonzero(labels != class_idx)

    tp_idxs, fp_idxs = get_common_idxs(a=pred_p_idxs, b=act_p_idxs, a_not_in_b=True)

    tn_idxs, fn_idxs = get_common_idxs(a=pred_n_idxs, b=act_n_idxs, a_not_in_b=True)

    return tp_idxs, fp_idxs, tn_idxs, fn_idxs


def get_top_k_contributor(a, k=1):
    # Get a tuple of unique values & their frequency in numpy array
    uniqueValues, occurCount = np.unique(a, return_counts=True)
    k = min(k, len(uniqueValues))
    result = sorted(list(zip(uniqueValues, occurCount)), key=lambda x: x[1], reverse=True)
    # total = np.sum(occurCount)
    return result[:k]


def calc_recall(tps, fns):
    return tps / (tps + fns)


def calc_precision(tps, fps):
    return tps / (tps + fps)


def calc_f1score(recall, precision):
    return 2 * precision * recall / (precision + recall)


def analyze_predictions(labels, true_labels, class_names):

    nc = len(class_names)
    tps = dict(zip(class_names, [0.0] * len(class_names)))
    fps = dict(zip(class_names, [0.0] * len(class_names)))
    tns = dict(zip(class_names, [0.0] * len(class_names)))
    fns = dict(zip(class_names, [0.0] * len(class_names)))
    top_k_class_in_fps_of = dict(zip(class_names, [0.0] * len(class_names)))
    top_k_class_in_fns_of = dict(zip(class_names, [0.0] * len(class_names)))

    for y, c in enumerate(class_names):
        tp_idxs, fp_idxs, tn_idxs, fn_idxs = calc_metrics(labels, true_labels, class_idx=y)

        tps[c], fps[c] = len(tp_idxs), len(fp_idxs)
        tns[c], fns[c] = len(tn_idxs), len(fn_idxs)

        top_k_class_in_fps_of[c] = \
            get_top_k_contributor(true_labels[fp_idxs], k=2) if len(fp_idxs) >= 1 else [(-1, -1)]
        top_k_class_in_fns_of[c] = \
            get_top_k_contributor(labels[fn_idxs], k=2) if len(fn_idxs) >= 1 else [(-1, -1)]

    precision, recall, acc, f1_score = dict(), dict(), dict(), dict()
    for c in class_names:
        precision[c] = round(calc_precision(tps[c], fps[c]), 2)
        recall[c] = round(calc_recall(tps[c], fns[c]), 2)
        f1_score[c] = round(calc_f1score(recall[c], precision[c]), 2)

        for i, (k, freq) in enumerate(top_k_class_in_fps_of[c]):
            top_k_class_in_fps_of[c][i] = None if k == -1 else (class_names[k], freq)
        for i, (k, freq) in enumerate(top_k_class_in_fns_of[c]):
            top_k_class_in_fns_of[c][i] = None if k == -1 else (class_names[k], freq)

    for c in class_names:
        print("top_fps of {}: {}\n"
              "top_fns of {}: {}".format(c, top_k_class_in_fps_of[c],
                                         c, top_k_class_in_fns_of[c]))

    metrics = {'pre': precision, 'rec': recall, 'f1s': f1_score}
    for name, val in metrics.items():
        print("{} = {}, avg = {}".format(name, val, round(sum([v/nc for v in val.values()]), 2)))


def visualize_predictions(images, labels, num_classes, class_names, samples_per_class=None, true_labels=None):
    plt.figure(figsize=(25, 25))

    if samples_per_class is None:
        rows = 0
        for y in range(num_classes):
            if true_labels is not None:
                rows = max(len(np.flatnonzero(true_labels == y)), rows)
            else:
                rows = max(len(np.flatnonzero(labels == y)), rows)
    else:
        rows = samples_per_class

    for y in range(num_classes):

        draw_idxs = np.flatnonzero(true_labels == y) if true_labels is not None else np.flatnonzero(labels == y)

        if draw_idxs.shape[0] >= 1:

            if samples_per_class is not None:
                draw_idxs = np.random.choice(draw_idxs, samples_per_class, replace=False)

            for i, idx in enumerate(draw_idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(rows, num_classes, plt_idx)
                plt.imshow(images[idx, :, :, :], aspect='auto')
                plt.axis('off')
                if true_labels is not None:
                    if i == 0:
                        plt.title('Column\nTrue\nLabel\nis\n{}\n\n\n{}'.
                                  format(class_names[true_labels[idx]], class_names[labels[idx]]),
                                  fontsize=20, fontweight='bold')
                    else:
                        plt.title('{}'.format(class_names[labels[idx]]),
                                  fontsize=20, fontweight='bold')
                elif i == 0:
                    plt.title('Column\nLabel\nis\n{}'.format(class_names[labels[idx]]),
                              fontsize=20, fontweight='bold')





