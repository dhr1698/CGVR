import argparse

import json
import os
import shortuuid

import re
import sys

from tqdm import tqdm


def main(args):

    with open(args.question_file, "r") as f:
        questions = [json.loads(q) for q in f]
    # questions = questions[:3]

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    with open(args.answer_file, "w") as f:
        for question in tqdm(questions):
            idx = question["question_id"]
            qs = question["text"]

            if args.perspective == "inverse":
                pat = re.compile(r"Is there a(n)? (.*) in the (image|imange)\?")
            else:
                print("Error caption mode! Only inverse.")
                sys.exit(1)

            if mat := pat.match(qs):
                target = mat.group(2).strip()
                outputs = f"No no {target} in the image?"
            else:
                print("[Warning] none-match line:", idx)

            answer_id = shortuuid.uuid()
            answer_json = json.dumps({
                "question_id": idx,
                "question": question["text"],
                "caption": outputs,
                "answer_id": answer_id,
                "model_name": "wo"
            })
            f.write(answer_json + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="data/coco")
    parser.add_argument("--question-file", type=str, default="benchmark/POPE/coco/coco_pope_adversarial.jsonl")
    parser.add_argument("--answer-file", type=str, default="output/coco_pope/caption/coco_pope_adversarial_caption_llava_inverse.jsonl")
    parser.add_argument("--perspective", type=str, default="inverse")
    args = parser.parse_args()

    main(args)
