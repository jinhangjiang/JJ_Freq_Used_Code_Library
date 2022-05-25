# Set Seeds
import torch
import random
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available

def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  if is_torch_available():
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

  if is_tf_available():
      import tensorflow as tf
 
      tf.random.set_seed(seed)
 
set_seed(42)
