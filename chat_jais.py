from typing import List, Optional
from llama import Llama, Dialog
import fairscale
from deep_translator import GoogleTranslator
from langdetect import detect
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
jais_model_path = "inception-mbzuai/jais-13b-chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(jais_model_path)
jais_model = AutoModelForCausalLM.from_pretrained(jais_model_path, device_map="auto",local_files_only=False, offload_folder="./jais_chat_offloaded_folder",trust_remote_code=True)

def translate_text(text, target_language):
    # The API URL
    url = "https://a0a2-65-49-81-191.ngrok-free.app/translate"

    # The data to be sent in the request
    data = {
        "text":text,"language": target_language
    }

    # Sending the POST request
    response = requests.post(url, json=data)

    # Checking if the request was successful
    if response.status_code == 200:
        # Return the translated text
        return response.json().get(target_language, "")
    else:
        # Return an error message if the request failed
        return f"Failed to get response: {response.status_code}"


def create_jais_dialog(crisis, sector,is_injuries=0):
    #translation ...
    crisis=translate_text(crisis, "eng")
    sector=translate_text(sector, "eng")
    # create prompt
    if is_injuries:
        prompt = f"Please list suggested actions from a {sector} perspective that a nation should take in response to a {crisis} , Taking into account the presence of dead and injured  people, formatted as one action per line ending with a '+'."
    else:
        prompt = f"Please list suggested actions from a {sector} perspective that a nation should take in response to a {crisis} , formatted as one action per line ending with a '+'."
    

    # create system guidance
    system_guidance= f" without emojis."


# You are a political man in a place of taking orders for the best interest of the country in the field of economy, considering to show a good image for th>
#  about, i want the steps only with appreciating the egyptian culture and egyptians mindset, Don't say in the answer that you are a political man
    
    if sector == "health":
        system_guidance =  f" It is imperative to consistently make decisions that safeguard the health of citizens, mitigate the risk of potential natural disasters, and offer recommendations aimed at preventing such occurrences. So, give me a short answer about {crisis} without emojis."
    elif sector == "national security":
        system_guidance =  f" Your decisions should be geared towards ensuring Egyptian national security and the safety of the people, all while respecting international and diplomatic boundaries. So, give me a short answer about {crisis} without emojis."
    elif sector == "economic":
        system_guidance =  f" You are required to make decisions that serve the best interests of the nation's economy, considering prevailing economic conditions, the stock market, and the guidance of the Central Bank of Egypt. Give a short answer about {crisis} without emojis."
    elif sector == "education":
        system_guidance =f" You are tasked with making decisions that prioritize the best interests of the students, fostering their learning experiences, and offering recommendations that contribute to the advancement of education. So, give me a short answer about {crisis} without emojis."
    elif sector == "foreign policy":
        system_guidance = f" The decision must align accurately with the policies and laws of the Arab Republic of Egypt, ensuring it is in the best interest of the country, all the while upholding strong international diplomatic relations. So, give me a short answer about {crisis} without emojis."
    elif sector == "media":
        system_guidance = f" Compliance with the regulations set forth by the unions associated with this sector is imperative, encompassing both audio-visual and written domains. So, give me a short answer about {crisis} without emojis."


    

    dialogs=f"### Instruction: {system_guidance} \n\nComplete the conversation below between [|Human|] and [|AI|]::\n### Input: {prompt}\n### Response: [|AI|]"
   
    return dialogs

def get_jais_response(text,tokenizer=tokenizer,model=jais_model):
    print(f"The used device :{device}")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048-input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response: [|AI|]")
    #res=translate_text(response[-1], "arb")
    return response[-1]



jais_text=create_jais_dialog("The rise in the price of the dollar", "economic",is_injuries=0 )
print(get_jais_response(text=jais_text,tokenizer=tokenizer,model=jais_model))
