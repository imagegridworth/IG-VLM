from .base_model_inference import *
import math
import re
from io import BytesIO
import torch
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


class Llava2Processor(BaseModelInference):
    def __init__(self, model_name, local_save_path=""):
        super().__init__(model_name, local_save_path)

    def load_model(self):
        model_name = get_model_name_from_path(self.model_name)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            self.model_name,
            None,
            model_name,
            device=torch.cuda.current_device(),
            device_map="cuda",
        )

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # Prepare images
        image_sizes = [self.raw_image.size]
        images_tensor = process_images(
            [self.raw_image], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        # Prepare input_ids
        input_ids = (
            tokenizer_image_token(
                self.user_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        # Decode and return outputs
        self.result_rext = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

    def extract_answers(self):
        return self.result_rext.split("ASSISTANT:")[-1]

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)
