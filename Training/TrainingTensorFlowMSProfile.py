import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import ViTConfig, TFViTForImageClassification
import time
from datetime import datetime
from packaging import version

def get_file_paths(directory):
    return [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.npy')]

def load_and_preprocess_npy(path):
    def _load_numpy(path_str):
        array = np.load(path_str.decode())
        array = np.transpose(array, (2, 0, 1))
        label = 1 if path_str.decode().endswith("1.npy") else 0
        return array.astype(np.float32), np.int32(label)
    
    data, label = tf.numpy_function(_load_numpy, [path], [tf.float32, tf.int32])
    data.set_shape((12, 128, 128))
    label.set_shape(())
    return data, label

def build_dataset(file_path, batch_size):
    file_paths = get_file_paths(file_path)
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_and_preprocess_npy)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset

class ManualEpochProfiler(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_begin(self, epoch, logs=None):
        epoch_log_dir = os.path.join(self.log_dir, f"epoch_{epoch}")
        options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 2,
                                                           python_tracer_level = 0,
                                                           device_tracer_level = 1)
        tf.profiler.experimental.start(epoch_log_dir, options = options)

    def on_epoch_end(self, epoch, logs=None):
        tf.profiler.experimental.stop()

def train_model(epochs=15, lr=1e-4):

    train_dir = 'SmallerSampleData/train_sample'
    val_dir = 'SmallerSampleData/validation_sample'

    strategy = tf.distribute.MirroredStrategy()

    per_worker_batch_size = 64
    num_workers = strategy.num_replicas_in_sync

    print(f"Number of workers: {num_workers}")

    global_batch_size = per_worker_batch_size * num_workers

    print(f"Global batch size: {global_batch_size}")

    train_dataset = build_dataset(train_dir, global_batch_size)
    validation_dataset = build_dataset(val_dir, global_batch_size)
    
    steps_per_epoch = len(get_file_paths(train_dir)) // global_batch_size
    validation_steps = len(get_file_paths(val_dir)) // global_batch_size

    log_dir = f'tf_epoch_logs_{num_workers}_gpus'

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, write_steps_per_second=True, update_freq='epoch') 

    with strategy.scope():
        config = ViTConfig(
                    image_size=128,
                    patch_size=16,
                    num_channels=12,
                    hidden_size=768,
                    num_hidden_layers=3,
                    num_attention_heads=3,
                    intermediate_size=3072,
                    num_labels=2,
                )
        model = TFViTForImageClassification(config)
        model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'])
        model.fit(train_dataset, 
                  epochs=epochs, 
                  steps_per_epoch=steps_per_epoch,
                  validation_data=validation_dataset,
                  validation_steps=validation_steps,
                  callbacks=[tboard_callback, ManualEpochProfiler(log_dir=log_dir)])
    
    return model, num_workers

def main():
    model, num_workers = train_model(epochs=15, lr=1e-4)
    model.save(f"model_{num_workers}_gpus.keras")

if __name__ == "__main__":
    main()
    