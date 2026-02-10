import json
from transformers import AutoTokenizer

MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 你说的 hf1 我先用这个
DATA_PATH = r"data/train.json"          # 按你的路径改

def span_to_bio(text, label_dict):
    tags = ["O"] * len(text)
    for etype, entities in label_dict.items():
        for ent, spans in entities.items():
            for start, end in spans:
                tags[start] = f"B-{etype}"
                for i in range(start + 1, end + 1):
                    tags[i] = f"I-{etype}"
    return tags

def char_tag_to_token_label(offset, char_tags):
    """offset: (start_char, end_char) with end exclusive in HF"""
    s, e = offset
    if s == e:  # special tokens like [CLS]/[SEP] often have (0,0)
        return -100, None

    chunk = char_tags[s:e]  # HF offset end is exclusive
    if all(t == "O" for t in chunk):
        return "O", "O"

    # first non-O decides type + B/I
    for idx, t in enumerate(chunk):
        if t != "O":
            ent_type = t.split("-", 1)[1]
            # if the first non-O char is exactly at token start and is B- -> B
            if idx == 0 and t.startswith("B-"):
                return f"B-{ent_type}", t
            else:
                return f"I-{ent_type}", t

    return "O", "O"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 读一条样本
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline().strip())

    text = sample["text"]
    label = sample.get("label", {})
    char_tags = span_to_bio(text, label)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=128
    )

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]

    print("\nTEXT:", text)
    print("\n--- Alignment Table (token | offset | slice | token_label) ---")
    for tok, off in zip(tokens, offsets):
        s, e = off
        slice_text = text[s:e] if s != e else ""
        token_label, _ = char_tag_to_token_label(off, char_tags)
        print(f"{tok:<12}  {str(off):<12}  {slice_text:<12}  {token_label}")

if __name__ == "__main__":
    main()
