import base64
from io import BytesIO

import gradio as gr
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from modules import chat, shared
from modules.chat import generate_chat_prompt

tools = [('terminal', 'Can be used to run terminal commands.', lambda x: x )]

def ui():
    pass


def custom_generate_chat_prompt(text, max_new_tokens, name1, name2, context, chat_prompt_size, **kwargs):

    context += "\n\nYou can use tools when necessary to help the user:\n\n"
    for name, description, _ in tools:
        context += f"> {name} - {description}\n"

    context += "\nTools should be used in the format <command, input>, such as <terminal, ls>."

    context += f"\n\nEXAMPLE:\n{name1}: hello\n{name2}: Hello, how can I help you today?\n{name1}: How much is 50!?\n{name2}: Let me calculate that for you. <calculator, 50!>"
    
    return generate_chat_prompt(text, max_new_tokens, name1, name2, context, chat_prompt_size, **kwargs)