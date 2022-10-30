from pathlib import Path
import os
import tensorflow as tf
import librosa
from librosa import filters, core
from scipy import signal
import numpy as np
import json
import glob
import random


class PreprocessWrapper:
    """ Wrapper object for creating, reading and preprocessing datasets.
    """
    def __init__(self, dlnet_config: dict, ds_config: str):
        """
        Init wrapper object. Reads DL Network config and dataset config.

        Parameters
        ----------
        dlnet_config : dict
            Config for DL Network and preprocessing
        ds_config : str
            Path to Dataset config to extract classes
        """
        # Set config:
        self.config = dlnet_config

        # Set random seed:
        random.seed = self.config['random_seed']

        # Classes
        if self.config['binary']:
            self.config['classes'] = ['compressed_wav', 'uncompr_wav']
        else:
            self.config['classes'] = self.get_classes_from_dataset(ds_config)

        # Input shape and filter settings:
        if self.config['calculate_mel']:
            # Mel filter init:
            self._mel_filter = filters.mel(self.config['sr'],
                                           self.config['n_fft'],
                                           n_mels=dlnet_config['n_mels'],
                                           norm='slaney')
            self.config['input_shape'] = (self.config['n_mels'],
                                          self.config['n_frames'],
                                          1)
        elif self.config['filter_signal']:
            # Crop spectrogram
            # frequency array
            self._freqs = np.fft.rfftfreq(self.config['n_fft'],
                                          d=1/self.config['sr'])
            # cutoff frequency bin at cutoff frequency
            self._cutoff_bin = int(np.argmin(np.abs(
                self._freqs-self.config['filter_config'][1])))
            self.config['input_shape'] = (int(
                                        len(self._freqs) - self._cutoff_bin),
                                          self.config['n_frames'],
                                          1)
        else:
            self.config['input_shape'] = (int(self.config['n_fft']/2 + 1),
                                          self.config['n_frames'],
                                          1)

    # Groundtruth extraction from folder name
    def folder_name_to_one_hot(self, file_path: str):
        """
        Extracts label from path information by searching matches between
        classes and path parts.

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        tensorflow.one_hot
            Tensor with label information.

        Raises
        ------
        ValueError
            Data cannot be labeled.
        """
        for label_idx, label in enumerate(self.config['classes']):
            if label in Path(file_path).parts:
                # get one hot encoded array
                return tf.one_hot(label_idx, len(self.config['classes']),
                                  dtype=tf.uint8)
        raise ValueError("Data cannot be labeled.")

    def load_and_preprocess_data(self, file_path: str):
        """
        Loads wav data and computes (mel) spectrum.

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        Tuple(numpy.ndarray, tensorflow.one_hot)
            Spectrum and Tensorflow one_hot object.
        """
        # path string is saved as byte array in tf.data.dataset
        # -> convert back to str
        if type(file_path) is not str:
            file_path = file_path.numpy()
            file_path = file_path.decode('utf-8')

        # load audio data
        y, _ = core.load(file_path, sr=self.config['sr'],
                         mono=self.config['mono'], dtype=np.float32,
                         res_type='kaiser_best')

        if self.config['filter_signal']:
            sos = signal.butter(10,
                                self.config['filter_config'][1],
                                self.config['filter_config'][0],
                                fs=self.config['sr'],
                                output='sos')
            y = signal.sosfilt(sos, y)

        # calculate stft from audio data
        spectrogram = core.stft(y, n_fft=self.config['n_fft'],
                                hop_length=self.config['hop_length'],
                                win_length=self.config['win_length'],
                                window=self.config['window'],
                                center=self.config['center'],
                                dtype=np.complex64,
                                pad_mode=self.config['pad_mode'])

        # get ground truth from file_path string
        one_hot = self.folder_name_to_one_hot(file_path)

        if self.config['calculate_mel']:
            # filter stft with mel-filter
            spectrogram = self._mel_filter.dot(np.abs(spectrogram)
                                               .astype(np.float32) **
                                               self.config['power'])
        else:
            # Crop spectrum at filter frequency:
            # Crop upper part if 'low'
            if self.config['filter_config'][0] == 'low':
                spectrogram = spectrogram[:self._cutoff_bin]
            # crop lower part else
            else:
                spectrogram = spectrogram[self._cutoff_bin:]

            spectrogram = librosa.amplitude_to_db(np.abs(spectrogram),
                                                  ref=np.max)
            spectrogram -= np.mean(spectrogram)
            spectrogram /= np.std(spectrogram)

        # add channel dimension for conv layer compatibility
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        return spectrogram, one_hot

    def preprocessing_wrapper(self, file_path: str):
        """
        Wrapper for preprocessing function to use in tensorflow API.

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        Tuple(numpy.ndarray, tensorflow.one_hot)
            Spectrum and Tensorflow one_hot object.
        """
        spec, one_hot = tf.py_function(func=self.load_and_preprocess_data,
                                       inp=[file_path],
                                       Tout=[tf.float32, tf.uint8])

        spec = tf.ensure_shape(spec, self.config['input_shape'])
        one_hot = tf.ensure_shape(one_hot, len(self.config['classes']))
        return spec, one_hot

    def tf_dataset_from_codec(self, codec_dir, train_test_ratio=.8,
                              save=False):
        """
        Generate a tensorflow dataset from the codec directory.

        Parameters
        ----------
        codec_dir : str
            Path to codec directory in database directory
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8
        save : bool, optional
            Save database to disk, by default False

        Returns
        -------
        tuple, tensorflow.data.Dataset
            Two tensowrflow datasets: Train, Test
        """
        # autotune computation
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_list, test_list = self.get_train_test_lists(codec_dir,
                                                          train_test_ratio)
        # Train set
        train_set = tf.data.Dataset.list_files(train_list[0])
        for train in train_list[1:]:
            # define a dataset of file paths
            train_set = train_set.concatenate(tf.data.Dataset.list_files(
                                            train))
        # Test set
        test_set = tf.data.Dataset.list_files(test_list[0])
        for test in test_list[1:]:
            # define a dataset of file paths
            test_set = test_set.concatenate(tf.data.Dataset.list_files(test))
        # Preprocessing via map
        train_set = train_set.map(self.preprocessing_wrapper,
                                  num_parallel_calls=AUTOTUNE)
        test_set = test_set.map(self.preprocessing_wrapper,
                                num_parallel_calls=AUTOTUNE)
        if save:
            folder_name = 'tf_dataset_' + os.path.split(codec_dir)[1]
            folder_path = os.path.join('_data', folder_name)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
            # save dataset to disk
            train_name = os.path.split(codec_dir)[1]+'_train_set'
            train_path = os.path.join(folder_path, train_name)
            test_name = os.path.split(codec_dir)[1]+'_test_set'
            test_path = os.path.join(folder_path, test_name)
            tf.data.experimental.save(dataset=train_set,
                                      path=train_path,
                                      compression='GZIP')
            tf.data.experimental.save(dataset=test_set,
                                      path=test_path,
                                      compression='GZIP')
            # Save the config to the folder:
            with open(os.path.join(folder_path, 'DLNet_config.json'),
                      'w') as fp:
                json.dump(self.config, fp, sort_keys=True, indent=4)
        return train_set, test_set

    def tf_dataset_from_database(self, db_path: str, train_test_ratio=.8,
                                 save=False):
        """
        Generate a tensorflow dataset from the database directory with all
        subfolders(codecs) included.

        Parameters
        ----------
        db_path : str
            Path to database directory.
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8
        save : bool, optional
            Save database to disk, by default False

        Returns
        -------
        tuple, tensorflow.data.Dataset
            Two tensowrflow datasets: Train, Test
        """
        # Get codec dirs:
        codecs = glob.glob(os.path.join(db_path, 'compressed_wav', '**'))
        codecs.append(os.path.join(db_path, 'uncompr_wav'))
        # Loop over codecs:
        train_set, test_set = self.tf_dataset_from_codec(codecs[0],
                                                         train_test_ratio)
        for codec in codecs[1:]:
            train, test = self.tf_dataset_from_codec(codec, train_test_ratio)
            train_set = train_set.concatenate(train)
            test_set = test_set.concatenate(test)
        if save:
            folder_name = 'tf_dataset_' + os.path.split(db_path)[1]
            folder_path = os.path.join('_data', folder_name)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
            # save dataset to disk
            train_name = os.path.split(db_path)[1]+'_train_set'
            train_path = os.path.join(folder_path, train_name)
            test_name = os.path.split(db_path)[1]+'_test_set'
            test_path = os.path.join(folder_path, test_name)
            tf.data.experimental.save(dataset=train_set,
                                      path=train_path,
                                      compression='GZIP')
            tf.data.experimental.save(dataset=test_set,
                                      path=test_path,
                                      compression='GZIP')
            # Save the config to the folder:
            with open(os.path.join(folder_path, 'DLNet_config.json'),
                      'w') as fp:
                json.dump(self.config, fp, sort_keys=True, indent=4)
        return train_set, test_set

    def get_train_test_lists(self, codec_dir: str, train_test_ratio=.8):
        """
        Returns two lists of subfolders in codec_dir for train and test
        directories. To keep the dataset small, only the first 35 folders are
        taken.

        Parameters
        ----------
        codec_dir : str
            Path to codec directory in database directory
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8

        Returns
        -------
        tuple, list
            Two list for train and test file directories.
        """
        # Number of subfolders:
        n_folders = len(glob.glob(os.path.join(codec_dir, '**')))
        # Take only 35 tracks
        n_folders = 35 if n_folders > 35 else n_folders
        # Seed list
        seeds = list(range(1, n_folders+1))
        # Train and test indices:
        train_idx = random.sample(seeds, int(train_test_ratio*n_folders))
        test_idx = list(set(seeds)-set(train_idx))
        # Train list
        train = [os.path.join(codec_dir, str(idx), '*.wav')
                 for idx in train_idx]
        # Test list
        test = [os.path.join(codec_dir, str(idx), '*.wav') for idx in test_idx]
        return train, test

    def load_tf_dataset(self, directory: str):
        """
        Wrapper to load dataset from directory.

        Parameters
        ----------
        directory : str
            Path to directory.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset
        """
        return tf.data.experimental.load(directory,
                        (tf.TensorSpec(self.config['input_shape'],
                                       dtype=tf.float32, name=None),
                         tf.TensorSpec(len(self.config['classes']),
                                       dtype=tf.uint8, name=None)),
                        compression='GZIP')

    def get_classes_from_dataset(self, json_file: str):
        """
        Get all available classes from the dataset config.

        Parameters
        ----------
        json_file : str
            Path to dataset config file

        Returns
        -------
        list
            List of all possible classes.
        """
        # Read json Dataset_config
        with open(json_file, "r") as read_file:
            config = json.load(read_file)
            # Get list of codec settings:
            codec_list = list(config.keys())[5:]
            # Replace 'db_format' by 'uncompr_wav' and return
            return ['uncompr_wav' if i == 'db_format' else i for i in
                    codec_list]

    @property
    def classes(self):
        return self.config['classes']

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config
