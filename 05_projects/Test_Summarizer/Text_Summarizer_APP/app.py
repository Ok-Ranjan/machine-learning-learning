from fastapi import FastAPI, Request
from pydantic import BaseModel # its decide - which type of input must be take 
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)


# templating
templates = Jinja2Templates(directory=".")

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

# client --- (data-json) ----> server

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    test = text.strip().lower()
    return text


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt" # it returns - Pytorch tensors as inputs
        # by default Hugging face model is Pytorch model
    ).to(device)

    # generate the summary => token ids (generate token ids, not actual summary text)
    model.to(device) # ensure that our data and model on same device
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4, # transformer generate 4 seq of out(summary) -> and choose one best
        early_stopping=True # after get EOS -> stop: and choose one best
    )
    
    # token ids convert to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # 
    return summary

# API Endpoints
@app.post("/summarize/") # client send -> server | endpoint - '/summarize/'
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary} # return to client

@app.get("/", response_class=HTMLResponse)  # client get <- server | endpoint - '/'
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})