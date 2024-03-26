from .base_model_inference import *
import math
import re
from io import BytesIO
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class LlavaNext16Processor(BaseModelInference):
    def __init__(self, model_name, local_save_path=""):
        super().__init__(model_name, local_save_path)

    def load_model(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.model.to("cuda:0")

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        # Prepare images

        inputs = self.processor(
            self.user_prompt, self.raw_image, return_tensors="pt"
        ).to("cuda:0")
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        self.result_text = self.processor.decode(output[0], skip_special_tokens=True)

        """
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
        self.result_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        """

    def extract_answers(self):
        return self.result_text.split("ASSISTANT:")[-1]

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)
