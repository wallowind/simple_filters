import numpy as np
import scipy.signal
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
'sampling_rate' is signal acquisition frequency.
Parameter 'log_level' sets level for logging function.
    """

    def __init__(self, data_source, sampling_rate, segmentation_length=None,
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
        self.rate = sampling_rate

    def _params_test(self, samples, channels):
        if samples:
            if hasattr(samples, '__len__') and len(samples) == 2:
                try:
                    temp = np.copy(self.data[samples[0]:samples[1], :])
                except Exception as msg:
                    self.log.error(msg)
                    sys.exit(2)
            else:
                try:
                    temp = np.copy(self.data[:samples, :])
                except Exception as msg:
                    self.log.error(msg)
                    sys.exit(2)
        else:
            temp = np.copy(self.data)
            self.log.info("All samples will be used in processing.")
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

    def _exec_func(self, name, args):
        try:
            _exec = getattr(self, name)
            return _exec(**args)
        except AttributeError:
            self.log.error(
                "Filter class does not have function: {0}".format(name))
            sys.exit(1)

    def ica(self, frame=10, data=None, samples=None, channels=None, **kwargs):
        """
Function performs 'Independent Component Analysis' decomposition. It based on
FastICA algorithm of 'scikit-learn' library. Use command 'ica_doc' for documentation.
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list
with chosen channels or int with specific channel.
        """
        model = FastICA(n_components=frame, **kwargs)
        data = self._params_test(samples, channels) if data is None else data
        if data.shape[0] % frame:
            new_shape = data.shape[0] // frame
            data = np.copy(data[:new_shape * frame, :])
            self.log.warning("""
Length of sample length [{0}] not coincident with overall samples [{1}] and
therefore data samples will be clipped to {2}.
                            """.format(frame, data.shape[0], new_shape * frame))
        to_res = []
        for j in range(data.shape[1]):
            temp = np.copy(data[:, j])
            temp = temp.reshape(-1, frame)
            a = model.fit_transform(temp)
            to_res.append(a.reshape(-1, 1))
        return np.concatenate(to_res, axis=1)

    def pca(self, frame=10, data=None, samples=None, channels=None, **kwargs):
        """
Function performs 'Principal Component Analysis' decomposition. It based on
PCA algorithm of 'scikit-learn' library. Use command 'pca_doc' for documentation.
'frame' - sample length.
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list
with chosen channels or int with specific channel.
        """
        model = PCA(n_components=frame, **kwargs)
        data = self._params_test(samples, channels) if data is None else data
        if data.shape[0] % frame:
            new_shape = data.shape[0] // frame
            data = np.copy(data[:new_shape * frame, :])
            self.log.warning("""
Length of sample length [{0}] not coincident with overall samples [{1}] and
therefore data samples will be clipped to {2}.
                            """.format(frame, data.shape[0], new_shape * frame))
        to_res = []
        for j in range(data.shape[1]):
            temp = np.copy(data[:, j])
            temp = temp.reshape(-1, frame)
            a = model.fit_transform(temp)
            to_res.append(a.reshape(-1, 1))
        return np.concatenate(to_res, axis=1)

    def fft(self, frame, eliminate_freq, data=None, samples=None, channels=None, **kwargs):
        """
Function performs filtering in frequency domain. The idea is that:
1) Fourie transform apply to signal (Fast Fourie Transform from numpy library)
2) It convolves with the filter core and selected frequencies is eliminated
3) Inverse Fourie transform of convolution and restoring filtered signal
'frame' - selected length of sliding window (longer gives better frequency
resolution, but poorer time resolution and vice verse);
'eliminate_freq' - list with frequencies to eliminate (must contains only
positive numbers from zero to max frequency = data sampling rate / 2)
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list
with chosen channels or int with specific channel.
For documentation on numpy functions use command 'fft_doc'.
        """
        data = self._params_test(samples, channels) if data is None else data
        if frame % 2:
            frame += 1
            self.log.warning("""
Length of sliding window increased by 1 and it is {0} now. For certain reason.
                            """.format(frame))
        parts = data.shape[0] // frame
        if data.shape[0] % frame:
            self.log.warning("""
Length of sliding window {0} not coincident with amount of samples {1}.
The amount of data samples to be filtered will be cropped to {2}]
                            """.format(frame, data.shape[0], parts * frame))
        freq = self.rate / (frame / 2 + 1)  # Frequency bin
        self.log.warning(
            "For given length of sliding window the frequency resolution is {0:.1f} Hz.".format(freq))
        possible_freq = [freq * i for i in range(frame // 2 + 1)]
        core = [1 for _ in range(frame // 2 + 1)]
        for f in eliminate_freq:
            eliminate = min(possible_freq, key=lambda x: abs(x - f))
            idx = possible_freq.index(eliminate)
            core[idx] = 0
            self.log.info(
                "For frequency {0} Hz will be deleted closest frequency: {1:.1f} Hz".format(f, (idx) * freq))
        core = np.array(core).reshape(-1, 1)
        temp = []
        for i in range(parts):
            image = np.fft.rfft(
                data[i * frame:(i + 1) * frame, :], n=frame, axis=0)
            convolution = np.multiply(image, core)
            filtered = np.fft.irfft(convolution, n=frame, axis=0)
            temp.append(filtered)
        return np.concatenate(temp)

    def wiener(self, data=None, samples=None, channels=None, **kwargs):
        """
Function performs Wiener filter algorithm to signal (realization from scipy.signal library).
THIS IS NOT WIENER, but very coarse mock of real algorithm.
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list
with chosen channels or int with specific channel.
For documentation on scipy wiener function use command 'wiener_doc'.
        """
        data = self._params_test(samples, channels) if data is None else data
        return scipy.signal.wiener(data)

    def mav(self, kernel=3, data=None, samples=None, channels=None, **kwargs):
        """
Function performs Moving Average algorithm to signal (realization from scipy.signal.medfilt library).
'kernel' - size of moving window; must be odd; default=3
        """
        data = self._params_test(samples, channels) if data is None else data
        core = [kernel, 1]
        return scipy.signal.medfilt(data, kernel_size=core)

    def wavelet(self, data=None, level=1000, samples=None, channels=None, **kwargs):
        """
Function applies wavelet decomposition to signal (based on PyWavelet library).
'level' - level of decomposition which will be used for inverse transform.
For comprehensive documentation (there are plenty significant parameters) one
should use command 'wavelet_doc' or check "https://pywavelets.readthedocs.io"
        """
        if 'wavelet' in kwargs.keys():
            wav_name = kwargs['wavelet']
        else:
            wav_name = 'db4'
        data = self._params_test(samples, channels) if data is None else data
        try:
            wav = pywt.Wavelet(wav_name)
        except Exception as msg:
            self.log.error(msg)
            sys.exit(2)
        max_level = pywt.dwt_max_level(data.shape[0], wav)
        if level > max_level:
            self.log.warning("""
The maximum level of decomposition for chosen wavelet is {0} and it
will be used for inverse transform.
                            """.format(max_level))
            level = max_level
        dec_res = pywt.wavedec(data, wav, level=None, axis=0, **kwargs)
        return pywt.waverec(dec_res[:level + 1], wav, axis=0, **kwargs)

    def pipeline(self, params, samples=None, channels=None):
        """
Function creates pipeline from different filters of Filter class.
'params' is dictionary in form: {'filter': **kwargs}, where **kwargs is
parameters for 'filter'. One should use OrderedDict type of dictionary because
all filters apply to signal in cascade, and order is not guaranteed for
standard python's dictionary (for Python<=3.5 at least)
'samples' - number of samples for processing;
'channels' - which channels you want to use in processing; it can be list
with chosen channels or int with specific channel.
        """
        data = self._params_test(samples, channels)
        for key, value in params.items():
            value['data'] = data
            data = self._exec_func(key, value)
        return data

    def plot(self, signals):
        """
Function for simplest plotting (based on matplotlib.pyplot library).
'signals' - list with signals to plot. Every signal will be plotted on one
figure (column align).
        """
        fig = plt.figure()
        for i in range(len(signals)):
            ax = fig.add_subplot(len(signals), 1, i + 1)
            ax.plot(signals[i])
        plt.show()

    def signal(self, samples=None, channels=None):
        """
Function return data slices, stored into class object.
        """
        return self._params_test(samples, channels)

    @property
    def ica_doc(self):
        print(FastICA.__doc__)

    @property
    def pca_doc(self):
        print(PCA.__doc__)

    @property
    def fft_doc(self):
        print("Forward Fourie transform:")
        print(np.fft.rfft.__doc__)
        print("-------------------------")
        print("Inverse Fourie transform:")
        print(np.fft.irfft.__doc__)

    @property
    def wiener_doc(self):
        print(scipy.signal.wiener.__doc__)

    @property
    def wavelet_doc(self):
        print("Wavelet decomposition:")
        print(pywt.wavedec.__doc__)
        print("----------------------")
        print("Inverse transform:")
        print(pywt.waverec.__doc__)
