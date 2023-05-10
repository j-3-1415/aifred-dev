import os
import json
import textwrap
import torch
import argparse
import faiss
import pickle

from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline


def get_prompt(human_prompt, context):
    prompt_template=f"""### Human: Construct an answer to the following request:
        {human_prompt}

        Use the following information to construct an answer:
        {context}

        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Answer in German. \n### Assistant:"""
    return prompt_template


def remove_human_text(text):
    return text.split("### Human:", 1)[0]


def parse_text(data):
    for item in data:
        text = item["generated_text"]
        assistant_text_index = text.find("### Assistant:")
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len("### Assistant:"):].strip()
            assistant_text = remove_human_text(assistant_text)
            wrapped_text = textwrap.fill(assistant_text, width=100)
            return wrapped_text


def run_inference(
        input_txt,
        base_model,
        tokenizer
):
    index = faiss.read_index("docs.index")

    with open("aifred_docs.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index

    docs = store.similarity_search(input_txt)
    context = ' '.join([doc.page_content for doc in docs])

    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    raw_output = pipe(get_prompt(input_txt, context))
    output_txt = parse_text(raw_output)

    print(output_txt)



def main():
    parser = argparse.ArgumentParser(
        description="Script to wrap VICUNA inference on arbitrary .txt input."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input text file."
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to ouput text file."
    )

    args = parser.parse_args()

    # set openai key
    os.environ["OPENAI_API_KEY"] = "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"

    tokenizer = LlamaTokenizer.from_pretrained("TheBloke/wizardLM-7B-HF")

    base_model = LlamaForCausalLM.from_pretrained(
        "TheBloke/wizardLM-7B-HF",
        load_in_8bit=True,
        device_map='auto',
    )

    run_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        base_model=base_model,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()




