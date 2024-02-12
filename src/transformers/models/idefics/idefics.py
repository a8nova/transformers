import torch
import sys

from transformers import (
    AutoProcessor,
    TFIdeficsForVisionText2Text,
    is_vision_available,
    IdeficsForVisionText2Text,
    BitsAndBytesConfig,
    is_tf_available
    
)
from transformers.testing_utils import torch_device
if is_tf_available():
    import tensorflow as tf

    from transformers import TFIdeficsForVisionText2Text, TFIdeficsModel, IdeficsProcessor
    from transformers.models.idefics.configuration_idefics import IdeficsPerceiverConfig, IdeficsVisionConfig
    from transformers.models.idefics.modeling_idefics import IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image

import requests

checkpoint = "HuggingFaceM4/tiny-random-idefics"
def run_idefics(py_or_tf="pt"):

  device = "cuda" if torch.cuda.is_available() else "cpu"
  quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
  )

  if py_or_tf == "pt":
     model = IdeficsForVisionText2Text.from_pretrained(checkpoint,
                                                       use_safetensors=False,
                                                       offload_folder=checkpoint)
                                                       #quantization_config=quantization_config)
  else:
     model = TFIdeficsForVisionText2Text.from_pretrained(checkpoint,
                                                         from_pt=True)

  processor = AutoProcessor.from_pretrained(checkpoint)
  dogs_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"
  #cats_image_obj = Image.open(cat_image_path)  # 2 cats
  prompts = [
    [
           "User:",
           dogs_image_url,
           "Describe this image.\nAssistant:",
    ],
   ]
   
  if py_or_tf == "pt":
     inputs = processor(prompts, return_tensors=py_or_tf).to(device)
  else:
     inputs = processor(prompts, return_tensors=py_or_tf)

 
  # Generation args
  bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
  #print("\ninputs ", inputs)
  generated_ids = model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=100)
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
  for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")

def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im
    
run_idefics("tf")
