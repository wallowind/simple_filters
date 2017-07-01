import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from LeapLog import get_log
import sys


class Filter(object):
    """
Class implements different filtering techniques for biosignal processing,
where 'data_source' must be filename for text file with strings, containing
samples of data, delimited by ','; if sample is multichannel one also must
point out length of a segment, which representing one channel; if strings
contain additional information one must set parameter 'additional_info' with
number of columns with this info (it must be in the end of line).
Parameter 'log_level' sets level for logging function.
    """

    def __init__(self, data_source, segmentation_length=None,
                 additional_info=None, log_level='WARNING'):
        self.log = get_log(__name__, log_level)
        try:
            temp = np.genfromtxt(data_source, delimiter=',')
        except Exception as msg:
            self.log.error(msg)
            sys.exit(1)
        if additional_info:
            temp = temp[:, :-additional_info]
        seg = segmentation_length
        if seg:
            if temp.shape[1] % seg:
                self.log.warning("Length of segment [{0}] not coincident with sample length [{1}]".format(
                    seg, temp.shape[1]))
            channels = temp.shape[1] // seg
            self.data = np.zeros(shape=(temp.shape[0] * seg, channels))
            for i in range(temp.shape[0]):
                for j in range(channels):
                    self.data[i * seg:(i + 1) * seg,
                              j] = temp[i, j * seg:(j + 1) * seg]
        else:
            self.data = np.zeros(shape=(temp.shape[0] * temp.shape[1], 1))
            for i in range(temp.shape[0]):
                self.data[i * temp.shape[1]:(i + 1) * temp.shape[1]] = temp[i]
        self.log.warning("Data was reshaped to {0}".format(self.data.shape))

    def _ica(self, data, **kwargs):
        model = FastICA(**kwargs)
        return model.fit_transform(np.copy(data))

    def _pca(self, data, **kwargs):
        model = PCA(**kwargs)
        return model.fit_transform(np.copy(data))

    def _params_test(self, samples, channels):
        if samples:
            try:
                temp = np.copy(self.data[:samples, :])
            except Exception as msg:
                self.log.error(msg)
                sys.exit(2)
        else:
            temp = np.copy(self.data)
        if channels:
            try:
                temp = temp[:, channels]
            except IndexError:
                self.log.error("Data contains only {0} channels and you set {1}".format(
                    self.data.shape[1], channels))
                sys.exit(1)
            except Exception as msg:
                self.log.error(msg)
                sys.exit(2)
        if len(temp.shape) == 1:
            temp = temp.reshape(-1, 1)
        return temp

    def ica(self, samples=None, channels=None, **kwargs):
        """
Function performs 'Independent Component Analysis' decomposition. It based on
FastICA algorithm of 'scikit-learn' library. Use command 'ica_doc' for documentation.
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list or tuple
with chosen channels or int with specific channel.
        """
        return self._ica(self._params_test(samples, channels), **kwargs)

    def pca(self, samples=None, channels=None, **kwargs):
        """
Function performs 'Principal Component Analysis' decomposition. It based on
PCA algorithm of 'scikit-learn' library. Use command 'pca_doc' for documentation.
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list or tuple
with chosen channels or int with specific channel.
        """
        return self._pca(self._params_test(samples, channels), **kwargs)

    @property
    def ica_doc(self):
        print(FastICA.__doc__)

    @property
    def pca_doc(self):
        print(PCA.__doc__)


if __name__ == '__main__':
    a = Filter('neupy_raw.csv', 150, 8)
