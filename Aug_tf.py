import librosa
import librosa.display
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import tensorflow_io as tfio

def sparse_warp(mel_spectrogram, time_warping_para=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """

    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[0], fbank_size[1]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    tf.random.set_seed(10)
    pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32, seed=10) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_freq,src_ctr_pt_time ), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_freq, dest_ctr_pt_time), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image

# frequency masking
freq_mask = tfio.audio.time_mask(warpd_spec, param=2)

# time masking
time_mask = tfio.audio.freq_mask(freq_mask, param=12)

def show_spec(spec):
  D = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
  img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=22050)
  plt.colorbar(img,  format="%+2.f dB")
  plt.show()

