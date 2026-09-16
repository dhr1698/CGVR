import argparse
import torch

import json
import os
import shortuuid

import sys
sys.path.append("./")

from model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.conversation import conv_templates
from model.llava.model.builder import load_pretrained_model
from model.llava.utils import disable_torch_init
from model.llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
from tqdm import tqdm
from transformers import set_seed


def check_yes_or_no(string):
    check_list = ["yes", "no", "Yes", "No"]
    word_list = string.split(" ")
    for word in word_list:
        for check in check_list:
            if word == check:
                return word
    return None


def get_token_id(tokenizer, token, output_ids):
    prefixed_token = "▁" + token
    try:
        return output_ids.index(tokenizer.convert_tokens_to_ids(prefixed_token))
    except ValueError:
        return output_ids.index(tokenizer.convert_tokens_to_ids(token))


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)  # llava-v1.5-7b
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    with open(args.question_file, "r") as f:
        questions = [json.loads(q) for q in f]

    with open(args.caption_file, "r") as f:
        captions = [json.loads(c) for c in f]

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    with open(args.answer_file, "w") as f:
        for i, question in enumerate(tqdm(questions)):
            idx = question["question_id"]
            image_file = question["image"]
            qs = question["text"]
            label = question["label"]

            image_caption = captions[i]["caption"]
            image_caption = "Caption: {}\n".format(image_caption)

            task_prompt = "Task: Review the image and answer the following question. Think step by step.\nQuestion: "

            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + image_caption + task_prompt + qs.lower()
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + image_caption + task_prompt + qs.lower()

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor = image_tensor.unsqueeze(0).half().cuda()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            with torch.inference_mode():
                output_model = model(input_ids, images=image_tensor, return_dict=True)
                predictions = output_model.logits
                probabilities = torch.softmax(predictions[0, -1, :], dim=-1)
                top_k_values, top_k_indices = probabilities.topk(args.topk)

                predicted_tokens = tokenizer.decode(top_k_indices.tolist(), skip_special_tokens=True).strip().split()
                aggregation_dict = {"Yes": 0, "No": 0}

                for token, value in zip(predicted_tokens, top_k_values):
                    if token in aggregation_dict:
                        aggregation_dict[token] += value.item()

                final_answer = "Yes" if aggregation_dict["Yes"] > aggregation_dict["No"] else "No"

            answer_id = shortuuid.uuid()
            answer_json = json.dumps({
                "question_id": idx,
                "question": question["text"],
                "answer": final_answer,
                "label": label,
                "aggregation_logits": aggregation_dict,
                "answer_id": answer_id,
                "model_name": model_name
            })
            f.write(answer_json + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/coco")
    parser.add_argument("--question-file", type=str, default="benchmark/POPE/coco/coco_pope_adversarial.jsonl")
    parser.add_argument("--answer-file", type=str, default="output/coco_pope/NF/NF_llava_normal_coco_adversarial_pope.jsonl")
    parser.add_argument("--caption-file", type=str, default="output/coco_pope/caption/coco_pope_adversarial_caption_llava_normal.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_model(args)
