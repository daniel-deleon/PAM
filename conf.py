# Spectrogram parameters

BLUE_B = dict(
    low_cut_off_freq = 39,
    high_cut_off_freq = 51,
    blur_axis = 'time',
    factor = 3.25,
    num_fft = 512,
    # padding in seconds to add to beginning/ending of wav files
    padding_secs = 5
)
BLUE_D = dict(
    low_cut_off_freq = 30,
    high_cut_off_freq = 90,
    blur_axis = 'frequency',
    factor = 1.125,
    num_fft = 256,
    # padding in seconds to add to beginning/ending of wav files
    padding_secs = 3
)
FIN = dict(
    low_cut_off_freq = 10,
    high_cut_off_freq = 36,
    blur_axis = 'frequency',
    factor = 2.0,
    num_fft = 128,
    # padding in seconds to add to beginning/ending of wav files
    fin_padding = 3
)

# Classification model parameters
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MODEL_GRAPH_NAME = 'classify_image_graph_def.pb'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'
#DATA_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
#DATA_URL = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
#DATA_URL = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'