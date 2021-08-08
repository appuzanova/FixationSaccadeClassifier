import math
import numpy as np

from enum import Enum, unique
from typing import Tuple, List


class IDTFixationSaccadeClassifier:

    def __init__(self, threshold: float = 100.0, win_len: int = 50):
        self.threshold = threshold
        self.win_len = win_len

    @staticmethod
    def _calc_min_max_dispersion(x: np.array, y: np.array, win_beg: int, win_end: int) -> int:
        dx = np.max(x[win_beg: win_end]) - np.min(x[win_beg: win_end])
        dy = np.max(y[win_beg: win_end]) - np.min(y[win_beg: win_end])
        return dx + dy

    def fit_predict(self, x: np.array, y: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations, saccades = [], []
        fixation_colors, saccades_colors = [], []

        win_beg, win_end = 0, min(len(x) - 1, self.win_len)

        fix_c, sacc_c = 0, 0

        while (win_beg < len(x)):
            dispersion = self._calc_min_max_dispersion(x, y, win_beg, win_end)
            if dispersion < self.threshold:
                while win_end < len(x) - 1 and dispersion < self.threshold:
                    win_end += 1
                    dispersion = self._calc_min_max_dispersion(x, y, win_beg, win_end)
                for i in range(win_beg, win_end):
                    fixations.append(i)
                    fixation_colors.append(fix_c)
                fix_c += 10
                win_beg, win_end = win_end + 1, min(len(x) - 1, win_end + 1 + self.win_len)
            else:
                saccades.append(win_beg)
                if len(saccades) > 0 and win_beg - 1 != saccades[-1]:
                    sacc_c += 10
                saccades_colors.append(sacc_c)
                win_beg += 1

        return fixations, saccades, fixation_colors, saccades_colors


class IVTFixationSaccadeClassifier:

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def fit_predict(self, x: np.array, y: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations, saccades = [], []
        fixation_colors, saccades_colors = [], []

        fix_c, sacc_c = 0, 0
        for i in range(1, len(x)):
            dist = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            if dist <= self.threshold:
                fixations.append(i)
                fixation_colors.append(fix_c)
                if len(saccades) > 0 and i - 1 == saccades[-1]:
                    sacc_c += 10
            else:
                saccades.append(i)
                saccades_colors.append(sacc_c)
                if len(fixations) > 0 and i - 1 == fixations[-1]:
                    fix_c += 10

        return fixations, saccades, fixation_colors, saccades_colors


class IHMMFixationSaccadeClassifier:

    @unique
    class States(Enum):
        SACC = 'sacc'
        FIX = 'fix'

    def __init__(
            self,
            fix_median: float = 1.0,
            fix_variance: float = 10.0,
            sacc_median: float = 80.0,
            sacc_variance: float = 60.0,
            prob_fix_fix: float = math.log(0.95),
            prob_sacc_sacc: float = math.log(0.95),
            prob_fix_sacc: float = math.log(0.95),
            prob_sacc_fix: float = math.log(0.95)
    ):
        self.fix_median = fix_median
        self.fix_variance = fix_variance
        self.sacc_median = sacc_median
        self.sacc_variance = sacc_variance
        self.log_probas = {
            self.States.FIX: {
                self.States.FIX: prob_fix_fix,
                self.States.SACC: prob_fix_sacc
            },
            self.States.SACC: {
                self.States.FIX: prob_sacc_fix,
                self.States.SACC: prob_sacc_sacc
            },
        }

    @staticmethod
    def _normal_dist(x: np.array, mu: float, sigma: float):
        m1 = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        m2 = math.exp(-math.pow(x - mu, 2) / (2 * math.pow(sigma, 2)))
        return m1 * m2

    def _calc_val_proba(self, value: np.array, state: States):
        if state == self.States.SACC:
            p = self._normal_dist(value, self.sacc_median, self.sacc_variance)
            if p == 0:
                p += 0.0001
            return math.log(p)
        else:
            p = self._normal_dist(value, self.fix_median, self.fix_variance)
            if p == 0:
                p += 0.0001
            return math.log(p)


    def _find_path(self, x: np.array, y: np.array):
        dst = []

        for i in range(1, len(x)):
            dist = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            dst.append(dist)

        values = [{}]
        path = {}
        v0 = {
            self.States.FIX: math.log(0.55),
            self.States.SACC: math.log(0.45)
        }

        for state in self.States:
            values[0][state] = v0[state] + self._calc_val_proba(dst[0], state)
            path[state] = [state]

        for elem in range(1, len(dst)):
            values.append({})
            path_new = {}
            for st in self.States:
                tmp = [(values[elem - 1][st0] + self.log_probas[st0][st] + self._calc_val_proba(dst[elem], st), st0)
                       for st0 in self.States]
                prob, state = max(tmp)
                values[elem][st] = prob
                path_new[st] = path[state] + [st]
            path = path_new

        prob, state = max([(values[len(dst) - 1][st], st) for st in self.States])

        return prob, path[state]

    def fit_predict(self, x: np.array, y: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations, saccades = [], []
        fixation_colors, saccades_colors = [], []

        pr, values = self._find_path(x, y)

        fix_c, sacc_c = 0, 0
        for i in range(0, len(values)):
            if values[i] == self.States.FIX and len(values) > 0 and values[i - 1] == self.States.SACC:
                sacc_c += 10
                fixations.append(i)
                fixation_colors.append(fix_c)
            elif values[i] == self.States.FIX:
                fixations.append(i)
                fixation_colors.append(fix_c)
            elif values[i] == self.States.SACC and len(values) > 0 and values[i - 1] == self.States.SACC:
                fix_c += 10
                saccades.append(i)
                saccades_colors.append(sacc_c)
            else:
                saccades.append(i)
                saccades_colors.append(sacc_c)

        return fixations, saccades, fixation_colors, saccades_colors


class IAOIFixationSaccadeClassifier:

    def __init__(self, threshold: float = 15.0, areas: List[List[float]] = [[-100.0, -100.0, 100.0, 100.0]]):
        self.threshold = threshold
        self.areas = areas

    def _find_area(self, x: float, y: float):
        area = None
        for a in self.areas:
            if a[0] <= x <= a[2] and a[1] <= y <= a[3]:
                area = a
                break
        return area


    def fit_predict(self, x: np.array, y: np.array) -> Tuple[List[int], List[int]]:

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations = []
        fixation_colors = []

        fix_c, sacc_c = 0, 0

        curr_a = None
        points = []
        for i in range(len(x)):
            a = self._find_area(x[i], y[i])
            if a is not None:
                if curr_a is None:
                    curr_a = a
                    points.append(i)
                elif curr_a == a:
                    points.append(i)
                else:
                    if len(points) < self.threshold:
                        points = []
                        curr_a = None
                    else:
                        for elem in points:
                            fixations.append(elem)
                            fixation_colors.append(fix_c)
                            fix_c += 10
                        points = [i]
                        curr_a = a

            else:
                curr_a = None
                if len(points) > self.threshold:
                    for elem in points:
                        fixations.append(elem)
                        fixation_colors.append(fix_c)
                        fix_c += 10
                    points = []

                else:
                    points = []

        return fixations, fixation_colors

class IWVTFixationSaccadeClassifier:

    def __init__(self, threshold: float = 15.0, win_len: int = 10):
        self.threshold = threshold
        self.win_len = win_len

    @staticmethod
    def _calc_win_dist(x: np.array, y: np.array, win_beg: int, win_end: int) -> int:
        dx = x[win_beg] - x[win_end]
        dy = y[win_beg] - y[win_end]
        return np.hypot(dx, dy)

    def fit_predict(self, x: np.array, y: np.array) -> Tuple[List[int], List[int], List[int], List[int]]:

        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y shape does not match')

        fixations, saccades = [], []
        fixation_colors, saccades_colors = [], []

        win_beg, win_end = 0, min(len(x) - 1, self.win_len)

        fix_c, sacc_c = 0, 0

        while (win_beg < len(x)):
            dispersion = self._calc_win_dist(x, y, win_beg, win_end)
            if dispersion < self.threshold:
                while win_end < len(x) - 1 and dispersion < self.threshold:
                    win_end += 1
                    dispersion = self._calc_win_dist(x, y, win_beg, win_end)
                for i in range(win_beg, win_end):
                    fixations.append(i)
                    fixation_colors.append(fix_c)
                fix_c += 10
                win_beg, win_end = win_end + 1, min(len(x) - 1, win_end + 1 + self.win_len)
            else:
                saccades.append(win_beg)
                if len(saccades) > 0 and win_beg - 1 != saccades[-1]:
                    sacc_c += 10
                saccades_colors.append(sacc_c)
                win_beg += 1

        return fixations, saccades, fixation_colors, saccades_colors