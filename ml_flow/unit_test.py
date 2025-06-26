"""
Unit test module for wafer defect feature engineering and data processing pipeline.
Covers region density, Radon transform features, geometric features, and full model pipeline testing.
"""

import unittest
import pandas as pd
import numpy as np
import itertools
from unittest.mock import patch, MagicMock
from scipy import ndimage, interpolate, stats
from skimage import measure
from skimage.transform import radon
from sklearn.model_selection import train_test_split
from collections import Counter


def find_dim(x):
    """Return the (rows, columns) dimensions of a 2D numpy array."""
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

def cal_den(x):
    """Calculate percentage of defect pixels (value == 2) in the wafer map."""
    if np.size(x) == 0:
        return 0.0
    return 100 * (np.sum(x == 2) / np.size(x))

def find_regions(x):
    """
    Divide wafer map into 13 subregions and calculate defect density for each.
    Used to capture spatial distribution of defects.
    """
    rows, cols = np.size(x, axis=0), np.size(x, axis=1)
    if rows < 5 or cols < 5:
        return [0.0] * 13
    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)
    ind1 = np.append(ind1, rows) if ind1[-1] < rows else ind1
    ind2 = np.append(ind2, cols) if ind2[-1] < cols else ind2
    reg1 = x[ind1[0]:ind1[1], :]
    reg2 = x[:, ind2[4]:]
    reg3 = x[ind1[4]:, :]
    reg4 = x[:, ind2[0]:ind2[1]]
    reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
    reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
    reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
    reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
    reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
    reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
    reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
    reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
    reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]
    fea_reg_den = [cal_den(reg) for reg in [reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13]]
    return fea_reg_den

def change_val(img):
    """
    Replace all pixel values of 1 with 0 in the wafer map.
    Useful for standardizing map values before analysis.
    """
    img[img == 1] = 0
    return img

def cubic_inter_mean(img):
    """
    Compute Radon transform of the wafer map, then apply cubic interpolation
    to summarize the projection into a 20-length feature vector.
    """
    if img.size == 0: return np.zeros(20)
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis=1)
    if xMean_Row.size < 4: return np.zeros(20)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100
    return ynew

def fea_geom(img):
    """
    Extract geometric features (area, perimeter, major/minor axis, eccentricity, solidity)
    of the largest connected defect region in the wafer map.
    """
    if img is None or not isinstance(img, np.ndarray): return None
    img_labels = measure.label(img, connectivity=1, background=0)
    if img_labels.max() == 0:
        return None
    props = measure.regionprops(img_labels)
    if not props:
        return None
    largest_prop = max(props, key=lambda p: p.area)
    return (largest_prop.area, largest_prop.perimeter, largest_prop.major_axis_length,
            largest_prop.minor_axis_length, largest_prop.eccentricity, largest_prop.solidity)

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=None):
    """Placeholder for confusion matrix plot (not implemented in test)."""
    pass


class TestWaferFeatureEngineering(unittest.TestCase):
    """Unit tests for wafer feature extraction functions."""

    def test_find_dim(self):
        """Test shape extraction from wafer map."""
        test_array = np.zeros((10, 25))
        self.assertEqual(find_dim(test_array), (10, 25))

    def test_cal_den(self):
        """Test defect density calculation."""
        test_array = np.array([[2, 0, 2], [1, 2, 0]])
        self.assertAlmostEqual(cal_den(test_array), 50.0)
        test_array_no_defects = np.ones((5, 5))
        self.assertAlmostEqual(cal_den(test_array_no_defects), 0.0)
        self.assertAlmostEqual(cal_den(np.array([])), 0.0)

    def test_find_regions(self):
        """Test region-based density calculation."""
        test_map = np.zeros((25, 25))
        test_map[0:5, :] = 2
        regions = find_regions(test_map)
        self.assertEqual(len(regions), 13)
        self.assertAlmostEqual(regions[0], 100.0)
        self.assertAlmostEqual(regions[4], 0.0)

    def test_change_val(self):
        """Test transformation of pixel values from 1 to 0."""
        test_array = np.array([[1, 2, 0], [1, 1, 2]])
        expected_array = np.array([[0, 2, 0], [0, 0, 2]])
        np.testing.assert_array_equal(change_val(test_array), expected_array)

    def test_fea_geom(self):
        """Test geometric feature extraction."""
        test_map = np.zeros((10, 10))
        test_map[2:5, 2:5] = 2
        features = fea_geom(test_map)
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 6)
        self.assertEqual(features[0], 9)
        test_map_no_defects = np.zeros((10, 10))
        features_none = fea_geom(test_map_no_defects)
        self.assertIsNone(features_none)

    def test_cubic_inter_mean(self):
        """Test cubic interpolation from Radon transform."""
        test_map = np.random.randint(0, 3, size=(25, 25))
        features = cubic_inter_mean(test_map)
        self.assertEqual(features.shape, (20,))


class TestWaferDataProcessing(unittest.TestCase):
    """Unit tests for full wafer data cleaning and ML pipeline."""

    def setUp(self):
        """Initialize synthetic dataframe with different defect types and shapes."""
        data = {
            'waferMap': [np.ones((30, 30)),
                         np.random.randint(0, 3, size=(27, 27)),
                         np.array([[0, 0], [0, 0]]),
                         np.random.randint(0, 3, size=(26, 26))],
            'trainTestLabel': [[['Test']], [['Training']], [['Test']], [['Training']]],
            'failureType': [[[]], [['Near-full']], [['Edge-Loc']], [['Center']]],
            'waferIndex': [4.0, 3.0, 2.0, 1.0],
        }
        self.df = pd.DataFrame(data)

    def test_initial_cleaning(self):
        """Test column rename and waferIndex processing."""
        df_clean = self.df.copy()
        df_clean.rename(columns={'trianTestLabel': 'trainTestLabel'}, inplace=True)
        self.assertIn('trainTestLabel', df_clean.columns)
        self.assertNotIn('trianTestLabel', df_clean.columns)

        df_clean.waferIndex = df_clean.waferIndex.astype(int)
        self.assertTrue(pd.api.types.is_integer_dtype(df_clean['waferIndex']))

        df_clean = df_clean.drop(['waferIndex'], axis=1)
        self.assertNotIn('waferIndex', df_clean.columns)

    def test_full_processing_pipeline(self):
        """Test full filtering logic to isolate a valid wafer entry."""
        df_processed = self.df.copy()
        df_processed.rename(columns={'trianTestLabel':'trainTestLabel'}, inplace=True)
        df_processed['waferMapDim'] = df_processed.waferMap.apply(find_dim)

        df_processed['failureType'] = df_processed.failureType.apply(
            lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else np.nan
        )

        mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
                        'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8}
        df_processed['failureNum'] = df_processed['failureType'].map(mapping_type)

        df_processed = df_processed.dropna(subset=['failureNum'])
        df_processed = df_processed[df_processed['failureType'] != 'Near-full']
        df_processed = df_processed[df_processed['waferMapDim'].apply(lambda x: all(np.greater(x, (5,5))))]

        self.assertEqual(len(df_processed), 1)
        self.assertEqual(df_processed.iloc[0]['failureType'], 'Center')

    @patch('matplotlib.pyplot.show')
    @patch('sklearn.ensemble.RandomForestClassifier')
    @patch('sklearn.svm.LinearSVC')
    def test_model_pipeline_runs(self, mock_svc, mock_rf, mock_show):
        """Test complete feature extraction and model training pipeline with mocks."""
        maps = [np.random.randint(0, 3, size=(26, 26)) for _ in range(8)]
        for i in range(len(maps)):
            maps[i][10+i, 10+i] = 2

        data = {
            'waferMap': maps,
            'failureType': ['Center', 'Center', 'Donut', 'Donut', 'Edge-Loc', 'Edge-Loc', 'Scratch', 'Scratch'],
            'failureNum': [0, 0, 1, 1, 2, 2, 6, 6]
        }
        df_final = pd.DataFrame(data)

        df_final['fea_reg'] = df_final.waferMap.apply(find_regions)
        df_final['fea_cub_mean'] = df_final.waferMap.apply(cubic_inter_mean)
        df_final['fea_cub_std'] = df_final.waferMap.apply(cubic_inter_mean)
        df_final['fea_geom'] = df_final.waferMap.apply(fea_geom)

        df_final.dropna(subset=['fea_geom'], inplace=True)
        self.assertEqual(len(df_final), 8)

        a = df_final.fea_reg.tolist()
        b = df_final.fea_cub_mean.tolist()
        c = df_final.fea_cub_std.tolist()
        d = df_final.fea_geom.tolist()

        fea_all = np.concatenate((np.array(a), np.array(b), np.array(c), np.array(d)), axis=1)
        label = np.array(df_final.failureNum)

        self.assertEqual(fea_all.shape[0], 8)
        self.assertEqual(fea_all.shape[1], 59)

        X_train, X_test, y_train, y_test = train_test_split(
            fea_all, label, stratify=label, random_state=42, test_size=0.5
        )

        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 4)

        mock_rf_instance = mock_rf.return_value
        mock_rf_instance.fit(X_train, y_train)
        mock_rf_instance.fit.assert_called_once_with(X_train, y_train)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
