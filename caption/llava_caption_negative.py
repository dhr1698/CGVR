import argparse
import json
import os
import sys

import shortuuid
import torch

sys.path.append("./")
sys.path.append("./model")

from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from model.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from model.llava.conversation import conv_templates
from model.llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from model.llava.model.builder import load_pretrained_model
from model.llava.utils import disable_torch_init


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)  # llava-v1.5-7b
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    with open(args.question_file, "r") as f:
        questions = [json.loads(q) for q in f]

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    with open(args.answer_file, "w") as f:
        for question in tqdm(questions):
            idx = question["question_id"]
            image_file = question["image"]

            image_token_se = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n"
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n"

            if args.perspective == "negative":
                caption_prompt = "Describe the image in detail, but give an opposite description by adding 'no' in front of each key word."
            else:
                print("Error caption mode! Only negative.")
                sys.exit(1)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs + caption_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(os.path.join(args.image_folder, image_file)).convert(
                "RGB"
            )
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_tensor = image_tensor.unsqueeze(0).half().cuda()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()

            answer_id = shortuuid.uuid()
            answer_json = json.dumps(
                {
                    "question_id": idx,
                    "question": question["text"],
                    "caption": outputs,
                    "answer_id": answer_id,
                    "model_name": model_name,
                }
            )
            f.write(answer_json + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/coco")
    parser.add_argument(
        "--question-file",
        type=str,
        default="benchmark/POPE/coco/coco_pope_adversarial.jsonl",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default="output/coco_pope/caption/coco_pope_adversarial_caption_llava_negative.jsonl",
    )
    parser.add_argument("--perspective", type=str, default="negative")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_model(args)
