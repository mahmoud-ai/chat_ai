from typing import List, Optional
from llama import Llama, Dialog
import fairscale
import requests

def translate_text(text, target_language):
    # The API URL
    url = "https://a0a2-65-49-81-191.ngrok-free.app/translate"

    # The data to be sent in the request
    data = {
       "text": text,"language": target_language
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


llama_model = Llama.build(
        ckpt_dir="llama-2-7b-chat", # The directory containing checkpoint files for the pretrained model.
        tokenizer_path="tokenizer.model", # The path to the tokenizer model used for text encoding/decoding.
        max_seq_len=1024,    # The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size=8,    # The maximum batch size for generating sequences. Defaults to 8.
    )
def get_llama_response( dialogs: str):   
    
    results =llama_model.chat_completion(
        dialogs,
        max_gen_len=None, # The maximum length of generated sequences. If None, it will be set to the model's max sequence length. Defaults to None.
        temperature=0.6,    # The temperature value for controlling randomness in generation.
        top_p=0.9,  # The top-p sampling parameter for controlling diversity in generation. Defaults to 0.9.
    )
    #res=translate_text(f"{results[0]['generation']['content']}", "arb")
    return f"{results[0]['generation']['content']}"


def create_llama_dialog(crisis, sector,is_injuries=0):
    #translate function to endlish
    crisis=translate_text(crisis, "eng")
    sector=translate_text(sector, "eng")
    # create prompt
    # Construct the query
    if is_injuries:
        prompt = f"As an Egyptian political man Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis} Taking into account the presence of dead and injured  people, formatted as one action per line ending with a '+'."
    else:
        prompt = f"As an Egyptian political man Please list the suggested actions from a {sector} perspective that a nation should take in response to a {crisis}, formatted as one action per line ending with a '+'."
    
    # create system guidance
    system_guidance= f"act as an Egyptian political man without emojis."

    if sector == "health":
        system_guidance =  f"It is imperative to consistently make decisions that safeguard the health of citizens, mitigate the risk of potential natural disasters, and offer recommendations aimed at preventing such occurrences. So, give me a short answer about {crisis} without emojis."
    elif sector == "national security":
        system_guidance =  f"Your decisions should be geared towards ensuring Egyptian national security and the safety of the people, all while respecting international and diplomatic boundaries. So, give me a short answer about {crisis} without emojis."
    elif sector == "economic":
        system_guidance =  f"You are required to make decisions that serve the best interests of the nation's economy, considering prevailing economic conditions, the stock market, and the guidance of the Central Bank of Egypt. Give a short answer about {crisis} without emojis."
    elif sector == "education":
        system_guidance =f" You are tasked with making decisions that prioritize the best interests of the students, fostering their learning experiences, and offering recommendations that contribute to the advancement of education. So, give me a short answer about {crisis} without emojis."
    elif sector == "foreign policy":
        system_guidance = f" The decision must align accurately with the policies and laws of the Arab Republic of Egypt, ensuring it is in the best interest of the country, all the while upholding strong international diplomatic relations. So, give me a short answer about {crisis} without emojis."
    elif sector == "media":
        system_guidance = f" Compliance with the regulations set forth by the unions associated with this sector is imperative, encompassing both audio-visual and written domains. So, give me a short answer about {crisis} without emojis."


    # create dialog
    
    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": system_guidance },  
            {"role": "user", "content": prompt}
        ]
    ]

    return dialogs


print(get_llama_response(create_llama_dialog("The rise in the price of the dollar", "economic",is_injuries=0)))

