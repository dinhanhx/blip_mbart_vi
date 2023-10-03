from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    MBartForConditionalGeneration,
    MBartTokenizer,
)


def translate(
    tokenizer: MBartTokenizer,
    model: MBartForConditionalGeneration,
    question: str,
    language_code: str,
):
    inputs = tokenizer(question, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda:0")

    outputs = model.generate(
        **inputs,
        decoder_start_token_id=tokenizer.lang_code_to_id[language_code],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def inference(
    processor: BlipProcessor,
    model: BlipForQuestionAnswering,
    question: str,
    image_file: Path,
):
    raw_image = Image.open(image_file).convert("RGB")
    inputs = processor(raw_image, question, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda:0", torch.float16)

    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    image_root_dir = Path("/storage/anhvd/data/coco-2017-images/")
    image_train_dir = image_root_dir.joinpath("train2017")
    image_val_dir = image_root_dir.joinpath("val2017")

    assert image_train_dir.is_dir()
    assert image_val_dir.is_dir()

    csv_file = Path("/storage/anhvd/data/ViVQA/test.csv")
    assert csv_file.is_file()
    dataset = pd.read_csv(csv_file, index_col=0)
    mid_dataset = dataset.copy(deep=True)
    out_dataset = dataset.copy(deep=True)
    len_dataset = len(dataset)

    # Setup vi2en
    tokenizer_vi2en = MBartTokenizer.from_pretrained(
        "vinai/vinai-translate-vi2en", src_lang="vi_VN"
    )
    model_vi2en = MBartForConditionalGeneration.from_pretrained(
        "vinai/vinai-translate-vi2en"
    )

    # Setup vqa
    processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model_vqa = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", torch_dtype=torch.float16
    )

    # Setup en2vi
    tokenizer_en2vi = MBartTokenizer.from_pretrained(
        "vinai/vinai-translate-en2vi", src_lang="en_XX"
    )
    model_en2vi = MBartForConditionalGeneration.from_pretrained(
        "vinai/vinai-translate-en2vi"
    )

    if torch.cuda.is_available():
        model_vi2en = model_vi2en.to("cuda:0")
        model_vqa = model_vqa.to("cuda:0")
        model_en2vi = model_en2vi.to("cuda:0")

    for i in tqdm(range(len_dataset)):
        question = dataset.iloc[i]["question"]

        image_file = Path()
        for image_dir in [image_train_dir, image_val_dir]:
            image_file = image_dir.joinpath(
                str(dataset.iloc[i]["img_id"]).zfill(12) + ".jpg"
            )
            if image_file.is_file():
                break

        translated_question = translate(tokenizer_vi2en, model_vi2en, question, "en_XX")
        generated_answer = inference(
            processor_vqa, model_vqa, translated_question, image_file
        )
        mid_dataset.at[i, "question"] = translated_question
        mid_dataset.at[i, "answer"] = generated_answer

        translated_answer = translate(
            tokenizer_en2vi, model_en2vi, generated_answer, "vi_VN"
        )
        out_dataset.at[i, "answer"] = translated_answer

    mid_dataset.to_csv("results/vivqa/mid.csv", index=True)
    out_dataset.to_csv("results/vivqa/generated_answers.csv", index=True)
