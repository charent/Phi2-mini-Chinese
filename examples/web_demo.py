import gradio as gr
from transformers import AutoTokenizer
from transformers import  GenerationConfig, TextIteratorStreamer
from threading import Thread
import torch

from model.modeling_phi import PhiPreTrainedModel

tokenizer_dir = './model_save/tokenizer/'
model_save_dir = './model_save/dpo/'
max_new_tokens = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = PhiForCausalLM.from_pretrained(model_save_dir).to(device)

eos_token = tokenizer.eos_token


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    
    # Formatting the input for the model.
    prompt = f"##提问:\n{message}\n##回答:\n"

    model_inputs = tokenizer([prompt], return_tensors="pt", return_token_type_ids=False).to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    # greedy search
    gen_conf = GenerationConfig(
        num_beams=1,
        do_sample=False,
        max_length=320,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        generation_config=gen_conf
    )

    # Starting the generation in a separate thread.
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  

    partial_message = ""

    for new_token in streamer:
        partial_message += new_token

        # Breaking the loop if the stop token is generated.
        if eos_token in partial_message:  
            break
        yield partial_message



if __name__ == "__main__":
    # Launching the web interface.
    demo = gr.ChatInterface(
                fn=predict,
                title="phi_mini_ChatBot",
                description="phi中文小模型对话测试",
                examples=['你好', '感冒了要怎么办']
            )
    demo.launch() 