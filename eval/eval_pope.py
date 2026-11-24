import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file", type=str, default="benchmark/POPE/coco/coco_pope_adversarial.jsonl"
    )
    parser.add_argument(
        "--gen_file_contradiction",
        type=str,
        default="output/coco_pope/NF/NF_llava_contradiction_coco_adversarial_pope.jsonl",
    )
    parser.add_argument(
        "--gen_file_guide",
        type=str,
        default="output/coco_pope/NF/NF_llava_guide_coco_adversarial_pope.jsonl",
    )
    parser.add_argument("--a", type=float, default=0.6)
    parser.add_argument("--b", type=float, default=0.4)
    return parser.parse_args()


def load_jsonl(file_path):
    try:
        with open(file_path, "r") as f:
            return [json.loads(l) for l in f]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"Failed to load or parse {file_path}: {e}")
        return []


def compute_probabilities(weights, *args):
    probabilities = {"Yes": 0, "No": 0}

    for weight, data in zip(weights, args):
        for key in ["Yes", "No"]:
            probabilities[key] += weight * data["aggregation_logits"][key]

    return probabilities["Yes"], probabilities["No"]


def eval_predictions(args, gts, predictions):
    true_pos, true_neg, false_pos, false_neg, unknown, yes_answers = 0, 0, 0, 0, 0, 0
    weights = (args.a, args.b)

    for gt, contradiction, guide in zip(gts, *predictions):
        gt_answer = gt["label"].lower().strip()
        yes_prob, no_prob = compute_probabilities(weights, contradiction, guide)
        gen_answer = "yes" if yes_prob > no_prob else "no"

        if gt_answer in {"yes", "no"}:
            if gen_answer == "yes":
                if gt_answer == "yes":
                    true_pos += 1
                else:
                    false_pos += 1
                yes_answers += 1
            elif gen_answer == "no":
                if gt_answer == "no":
                    true_neg += 1
                else:
                    false_neg += 1
            else:
                unknown += 1
        else:
            unknown += 1

    total_questions = len(gts)
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0
    yes_proportion = yes_answers / total_questions if total_questions > 0 else 0
    unknown_prop = unknown / total_questions if total_questions > 0 else 0

    return accuracy, precision, recall, f1, yes_proportion, unknown_prop


def main():
    args = parse_args()
    gts = load_jsonl(args.gt_file)
    contradiction_predictions = load_jsonl(args.gen_file_contradiction)
    guide_predictions = load_jsonl(args.gen_file_guide)

    results = eval_predictions(
        args, gts, (contradiction_predictions, guide_predictions)
    )

    print(f"Accuracy: {results[0] * 100:.2f}%")
    print(f"Precision: {results[1] * 100:.2f}%")
    print(f"Recall: {results[2] * 100:.2f}%")
    print(f"F1 Score: {results[3] * 100:.2f}%")
    print(f"Yes Proportion: {results[4] * 100:.2f}%")
    print(f"Unknown Proportion: {results[5] * 100:.2f}%")


if __name__ == "__main__":
    main()
