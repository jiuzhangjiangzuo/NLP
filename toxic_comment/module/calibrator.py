import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    ISOTONIC_REGRESSION = "isotonic_regression"
    PLATT_SCALING = "platt_scaling"
    def __init__(self, num_bins=10, model_type=None):
        self.num_bins = num_bins
        if model_type == self.ISOTONIC_REGRESSION:
            self.model = IsotonicRegression()
        else:
            self.model = LogisticRegression()

    def _get_bin_sizes(self, y_prob, num_bins):
        bins = np.linspace(0., 1. + 1e-8, num_bins +1)
        # 告诉每个概率属于哪一个区间
        bin_indies = np.digitize(y_prob, bins) - 1
        # 计算每个区间的样本个数
        bin_sizes = np.bincount(bin_indies, minlength=len(bins))
        # 跳过空的区间
        bin_sizes = [i for i in bin_sizes if i != 0]
        return bin_sizes

    def plot_reliability_diagrams(self, truth_label, pred_prob, label, output_path):
        truth_label, pred_prob = calibration_curve(truth_label, pred_prob, self.num_bins)
        plt.figure(figsize=(10, 10))
        plt.gca()

        plt.plot([0, 1], [0, 1], color="r", linestyle=":", label="Prefect Calibration")
        plt.plot(pred_prob, truth_label, label=label)
        plt.ylabel("Accuracy", fontsize=16)
        plt.xlabel("Confidence", fontsize=16)

        plt.grid(True, color="b")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_path, "{}.png".format(label))
        plt.savefig(fname=save_path, format="png")

    def cal_ece(self, truth_label, pred_prob, bin_sizes):
        ece = np.float32(0)
        total_samples = sum(bin_sizes)
        for m in range(len(bin_sizes)):
            ece = ece + (bin_sizes[m] / total_samples) * np.abs(truth_label[m] - pred_prob[m])
        return ece.item()

    def fit(self, truth_label, pred_prob):
        bin_sizes = self._get_bin_sizes(pred_prob, self.num_bins)
        uncalibrated_ece = self.cal_ece(truth_label, pred_prob, bin_sizes)
        expanded_pred_prob = np.expand_dims(pred_prob, axis=1)
        self.model.fit(expanded_pred_prob, truth_label)

        calibrated_prob = self.model.predict_proba(expanded_pred_prob)
        bin_sizes = self._get_bin_sizes(calibrated_prob.flatten(), self.num_bins)
        calibrated_ece = self.cal_ece(truth_label, calibrated_prob.flatten(), bin_sizes)
        return uncalibrated_ece, calibrated_ece

    def calibrate(self, pred_prob):
        expanded_pred_prob = np.expand_dims(pred_prob, axis=1)
        return self.model.predict_proba(expanded_pred_prob)
