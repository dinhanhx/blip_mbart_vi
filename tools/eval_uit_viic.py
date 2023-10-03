import json
import warnings
from pathlib import Path
from typing import Dict, List

import click
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    MBartForConditionalGeneration,
    MBartTokenizer,
)


def inference(dataset: List[Dict]):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    generated_captions: List[Dict] = []
    for datapoint in tqdm(dataset):
        raw_image = Image.open(datapoint["image_file"]).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda:0", torch.bfloat16)

        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        generated_captions.append({"id": datapoint["id"], "caption": caption})

    with open("results/uit_viic/generated_captions.json", "w", encoding="utf-8") as fp:
        json.dump(generated_captions, fp, ensure_ascii=False, indent=4)


def translate():
    tokenizer_en2vi = MBartTokenizer.from_pretrained(
        "vinai/vinai-translate-en2vi", src_lang="en_XX"
    )
    model_en2vi = MBartForConditionalGeneration.from_pretrained(
        "vinai/vinai-translate-en2vi"
    )
    if torch.cuda.is_available():
        model_en2vi = model_en2vi.to("cuda:0")

    generated_captions = json.load(open("results/uit_viic/generated_captions.json"))
    translated_captions: List[Dict] = []
    for datapoint in tqdm(generated_captions):
        inputs = tokenizer_en2vi(datapoint["caption"], return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda:0")

        outputs = model_en2vi.generate(
            **inputs,
            decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True,
        )
        caption = tokenizer_en2vi.decode(outputs[0], skip_special_tokens=True)
        translated_captions.append({"id": datapoint["id"], "caption": caption})

        with open(
            "results/uit_viic/translated_captions.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(translated_captions, fp, ensure_ascii=False, indent=4)


@click.command()
@click.argument("task")
def main(task: str):
    img_root_dir = Path("/storage/anhvd/data/coco-2017-images/train2017/")
    assert img_root_dir.is_dir()
    json_file = Path(
        "/storage/anhvd/data/coco-2017-vi/vi/uitviic_captions_test2017.json"
    )
    assert json_file.is_file()

    dataset: List[Dict] = json.load(open(json_file))["annotations"]
    for datapoint in dataset:
        fmt_file = str(datapoint["image_id"]).zfill(12) + ".jpg"
        img_file = img_root_dir.joinpath(fmt_file)
        assert img_file.is_file()
        datapoint["image_file"] = img_file

    if task == "blip inference":
        inference(dataset=dataset)
    elif task == "mbart translate":
        translate()
    else:
        warnings.warn(f"Task {task} is unknown")


if __name__ == "__main__":
    main()
