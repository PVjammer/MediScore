#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
#import scipy.stats as st
#import sys
#import time


class detMetrics:
    """Class representing a set of trials and containing:
       - FNR array      (False Negative Rate)
       - FPR array      (Falst Positive Rate)
       - TPR array      (True Positive Rate)
       - EER metric     (Equal Error Rate)
       - AUC metric     (Area Under the Curve)
       - Confidence Interval for AUC
    """

    def __init__(self, score, gt, fpr_stop=1, isCI=False, ciLevel=0.9, dLevel=0.0, total_num=1, sys_res='all'):
        """Constructor for detMetrics class.
        Arguments:
         - score:       1D pandas dateframe containing all the scores for a given system output
         - gt:          1D pandas dataframe containing the ground truth values for each probe (Y/N strings for "IsTarget")
         - fpr_stop:    the stop point of FAR for calculating partial AUC (boolean or multi-type bool and float?) (Typo?)
         - isCI:        boolean that, if TRUE, will cause the lower and upper confidence interval to be calculated for AUC.  Hurts performace.
         - ciLevel:     float value specifying what confidence level should be used for calculating the interval.
         - dLevel:      the lower and upper exclusions for d-prime calculation
         - total_num:   total number of probes in dataset (why do we need to pass this in??)
         - sys_res:     sting specifying "tr" (for truncated) or "all" (for the whole dataset).  Used to spcify if optOut should be considered.
        """

        self.fpr, self.tpr, self.fnr, self.thres, self.t_num, self.nt_num = Metrics.compute_points_sk(
            score, gt)

        print("Total# ({}),  Target# ({}),  NonTarget# ({}) \n".format(
             self.t_num+self.nt_num, self.t_num, self.nt_num))
        self.trr = round(float((self.t_num + self.nt_num)) / total_num, 2)

        self.eer = Metrics.compute_eer(self.fpr, self.fnr)
        self.auc = Metrics.compute_auc(self.fpr, self.tpr, 1.0)
        self.auc_at_fpr = Metrics.compute_auc(self.fpr, self.tpr, fpr_stop)
        self.d, self.dpoint, self.b, self.bpoint = Metrics.compute_dprime(
            self.fpr, self.tpr, dLevel)

        inter_point = Metrics.linear_interpolated_point(self.fpr, self.tpr, fpr_stop)
        self.tpr_at_fpr = inter_point[0][1]

        self.ci_lower = 0
        self.ci_upper = 0
        self.ci_tpr = 0

        if isCI:
            self.ci_lower, self.ci_upper, self.ci_tpr = Metrics.compute_ci(
                score, gt, ciLevel, fpr_stop)

        self.fpr_stop = fpr_stop
        self.sys_res = sys_res

    def __repr__(self):
        """Print from interpretor"""
        return "DetMetrics: eer({}), auc({}), ci_lower({}), ci_upper({}), ci_tpr({})".format(self.eer, self.auc, self.ci_lower, self.ci_upper, self.ci_tpr)

    def write(self, file_name):
        """ Save the Dump files (formatted in a binary) that contains
        a list of FAR, FPR, TPR, threshold, AUC, and EER values.
        file_name: Dump file name
        """
        import pickle
        dmFile = open(file_name, 'wb')
        pickle.dump(self, dmFile)
        dmFile.close()

    def render_table(self):
        """ Render CSV table using Pandas Data Frame
        """
        from collections import OrderedDict
        from pandas import DataFrame
        data = OrderedDict([('AUC', self.auc), ('EER', self.eer), ('FAR_STOP', self.fpr_stop), ('AUC@FAR', self.auc_at_fpr), ('CDR@FAR', self.tpr_at_fpr), (
            'AUC_CI_LOWER@FAR', self.ci_lower), ('AUC_CI_UPPER@FAR', self.ci_upper), ('TRR', self.trr), ('SYS_RESPONSE', self.sys_res)])
        my_table = DataFrame(data, index=['0'])

        return my_table.round(6)

    def get_eer(self):
        if self.eer == -1:
            self.eer = Metrics.compute_eer(self)
        return self.eer

    def get_auc(self):
        if self.auc == -1:
            self.auc = Metrics.compute_auc(self)
        return self.auc

    def get_ci(self):
        self.ci_lower, self.ci_upper, self.ci_tpr = Metrics.compute_ci(self)
        return self.ci_lower, self.ci_upper, self.ci_tpr

    def set_eer(self, eer):
        self.eer = eer

    def set_auc(self, auc):
        self.aux = auc


def load_dm_file(path):
    """ Load Dump (DM) files
        path: DM file name along with the path
    """
    import pickle
    file = open(path, 'rb')
    myObject = pickle.load(file)
    file.close()
    return myObject


class Metrics:

    @staticmethod
    def compute_points_sk(score, gt):
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their ground-truth using the sklearn package.
        score: system output scores
        gt: ground-truth for given trials
        """
        from sklearn.metrics import roc_curve

        label = np.where(gt == 'Y', 1, 0)
        target_num = label[label == 1].size
        nontarget_num = label[label == 0].size

        fpr, tpr, thres = roc_curve(label, score)
        fnr = 1 - tpr
        return fpr, tpr, fnr, thres, target_num, nontarget_num

    @staticmethod
    def compute_auc(fpr, tpr, fpr_stop=1):
        """ Computes the under area curve (AUC) given FPR and TPR values
        fpr: false positive rates
        tpr: true positive rates
        fpr_stop: fpr value for calculating partial AUC"""
        width = [x - fpr[i] for i, x in enumerate(fpr[1:]) if fpr[i + 1] <= fpr_stop]
        height = [(x + tpr[i]) / 2 for i, x in enumerate(tpr[1:])]
        p_height = height[0:len(width)]
        auc = sum([width[i] * p_height[i] for i in range(0, len(width))])
        return auc

    @staticmethod
    def compute_eer(fpr, fnr):
        """ computes the equal error rate (EER) given FNR and FPR values
        fpr: false positive rates
        fnr: false negative rates"""
        errdif = [abs(fpr[j] - fnr[j]) for j in range(0, len(fpr))]
        idx = errdif.index(min(errdif))
        eer = np.mean([fpr[idx], fnr[idx]])
        return eer

    # TODO: need to validate this and make command-line inputs
    @staticmethod
    def compute_ci(score, gt, ci_level, fpr_stop):
        """ compute the confidence interval for AUC
        score: system output scores
        gt: ground-truth for given trials
        lower_bound: lower bound percentile
        upper_bound: upper bound percentile"""
        from sklearn.metrics import roc_auc_score

        lower_bound = round((1.0 - float(ci_level)) / 2.0, 3)
        upper_bound = round((1.0 - lower_bound), 3)
        n_bootstraps = 500
        rng_seed = 77  # control reproducibility
        bootstrapped_auc = []

        rng = np.random.RandomState(rng_seed)
        indices = np.copy(score.index.values)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            new_indices = rng.choice(indices, len(indices))
            fpr, tpr, fnr, thres, t_num, nt_num = Metrics.compute_points_sk(
                score[new_indices].values, gt[new_indices])
            auc = Metrics.compute_auc(fpr, tpr, fpr_stop)
            # print("Bootstrap #{} FPR_stop {}, AUC: {:0.3f}".format(i + 1, fpr_stop, auc))

            bootstrapped_auc.append(auc)

        sorted_aucs = sorted(bootstrapped_auc)

        # Computing the lower and upper bound of the 90% confidence interval (default)
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        ci_lower = sorted_aucs[int(lower_bound * len(sorted_aucs))]
        ci_upper = sorted_aucs[int(upper_bound * len(sorted_aucs))]

        ci_tpr = 0  # TODO: after calculating CI for each TPR
        #print("Confidence interval for AUC: [{:0.5f} - {:0.5f}]".format(ci_lower, ci_upper))
        return ci_lower, ci_upper, ci_tpr

    # Calculate d-prime and beta
    @staticmethod
    def compute_dprime(fpr, tpr, d_level=0.0):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        #fpr_a = [0, .0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772,.9938,.9987, 1]
        #tpr_a = [0, 0.0013,.0062,.0228,.0668,.1587,.3085,.5,.6915,.8413,.9332,.9772, 1]
        #d_level = 0.2

        def range_limit(n, minn, maxn):
            return max(min(maxn, n), minn)

        d = []
        d_max = None
        d_max_idx = None
        beta = []
        beta_max = None
        beta_max_idx = None
        Z = norm.ppf
        mask = []
        for idx, x in enumerate(fpr):
            d.append(Z(range_limit(tpr[idx], 0.00001, .99999)) -
                     Z(range_limit(fpr[idx], 0.00001, 0.99999)))
            beta.append(exp((Z(range_limit(fpr[idx], 0.00001, 0.99999))
                             ** 2 - Z(range_limit(tpr[idx], 0.00001, .99999))**2) / 2))
            if (tpr[idx] >= d_level and tpr[idx] <= 1 - d_level and fpr[idx] >= d_level and fpr[idx] <= 1 - d_level):
                if (d_max == None or d_max < d[idx]):
                    d_max = d[idx]
                    d_max_idx = idx
                if (beta_max == None or beta_max < d[idx]):
                    beta_max = d[idx]
                    beta_max_idx = idx
                mask.append(1)
            else:
                mask.append(0)

        if (d_max_idx == None):
            return None, (0, 0), None, (0, 0)
        return d_max, (fpr[d_max_idx], tpr[d_max_idx]), beta_max, (fpr[beta_max_idx], tpr[beta_max_idx])

    @staticmethod
    def compute_aprime(fpr_a, tpr_a):
        """ computes the d-prime given TPR and FPR values
        tpr: true positive rates
        fpr: false positive rates"""
        from scipy.stats import norm
        from math import exp

        fpr = list(fpr_a)
        tpr = list(tpr_a)
        Z = norm.ppf
        a = []
        for i in range(0, len(fpr)):

            # Starting a' calculation
            if(fpr[i] <= 0.5 and tpr[i] >= 0.5):
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - fpr[i] * (1.0 - tpr[i]))
            elif(fpr[i] <= tpr[i] and tpr[i] <= 0.5):
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - fpr[i] / (4.0 * tpr[i]))
            else:
                a.append(0.75 + (tpr[i] - fpr[i] / 4.0) - (1.0 - tpr[i]) / (4.0 * (1.0 - fpr[i])))

        a_idx = a.index(max(a))
        a_max_point = (fpr[a_idx], tpr[a_idx])

        return max(a), a_max_point

    @staticmethod
    def linear_interpolated_point(x, y, x0):
        # given a list or numpy array of x and y, compute the y for some x0.
        # currently only applicable to interpolating ROC curves

        xy = list(zip(x, y))
        xy.sort(key=lambda x: x[0])

        if (len(x) == 0) or (len(y) == 0):
            print("ERROR: no data in x or y to interpolate.")
            exit(1)
        elif len(x) != len(y):
            print("ERROR: x and y are not the same length.")
            exit(1)

        # find x0 in the set of x's
        tuples = [p for p in xy if p[0] == x0]
        if len(tuples) > 0:
            return tuples
        else:
            # find the largest x smaller than x0
            smaller = [p for p in xy if p[0] < x0]
            ix_x01 = len(smaller) - 1  # index for the relevant point

            # if nothing is in the list of smaller points
            if ix_x01 == -1:
                x01 = 0
                y01 = 0
                x02 = xy[0][0]
                y02 = xy[0][1]
                if x02 != x01:
                    y0 = y01 + (y02 - y01) / (x02 - x01) * (x0 - x01)
                else:
                    y0 = (y02 + y01) / 2
                return [(x0, y0)]

            x01 = xy[ix_x01][0]
            y01 = xy[ix_x01][1]
            ix_x02 = ix_x01 + 1

            # check to see if there is a next x. If not, let it be (1,1)
            try:
                x02 = xy[ix_x02][0]
                y02 = xy[ix_x02][1]
            except IndexError:
                x02 = 1
                y02 = 1

            # linear interpolate
            if x02 != x01:
                y0 = y01 + (y02 - y01) / (x02 - x01) * (x0 - x01)
            else:
                y0 = (y02 + y01) / 2

            # return a single set of tuples to maintain format
            return [(x0, y0)]

    @staticmethod
    def compute_points_donotuse(score, gt):  # do not match with R results
        """ computes false positive rate (FPR) and false negative rate (FNR)
        given trial scores and their ground-truth.
        score: system output scores
        gt: ground-truth for given trials
        output:
        """
        from collections import Counter
        score_sorted = np.sort(score)
        val = score_sorted[::-1]
        # Since the score was sorted, the ground-truth needs to be reallocated by the index sorted
        binary = gt[np.argsort(score)[::-1]]
        # use the unique scores as a threshold value
        t = np.unique(score_sorted)[::-1]
        total = len(t)
        fpr, tpr, fnr = np.zeros(total), np.zeros(total), np.zeros(total)
        fp, tp, fn, tn = np.zeros(total), np.zeros(total), np.zeros(total), np.zeros(total)
        yes = binary == 'Y'
        no = np.invert(yes)
        counts = Counter(binary)
        n_N, n_Y = counts['N'], counts['Y']
        for i in range(0, total):
            tn[i] = np.logical_and(val < t[i], no).sum()
            fn[i] = np.logical_and(val < t[i], yes).sum()
            tp[i] = n_Y - fn[i]
            fp[i] = n_N - tn[i]
        # Compute true positive rate for current threshold
        tpr = tp / (tp + fn)
        # Compute false positive rate for current threshold
        fpr = fp / (fp + tn)
        # Compute false negative rate for current threshold.
        fnr = 1 - tpr     # fnr = 1 - tpr
        return fpr, tpr, fnr, t
