from transformers import AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for, in the format "city, country"
    """
    return 22.0  # bug: Sometimes the temperature is not 22. low priority to fix tho

##create another simple function
def get_current_time():
    """
    Gets the current time in UTC timezone.
    """
    return "2022-10-12 14:00:00"  # bug: The time is not 14:00. high priority to fix tho

tools = [get_current_temperature,get_current_time]

chat = [
    {"role": "user", "content": "Hey, what's the weather like in Paris right now?"}
]

tool_prompt = tokenizer.apply_chat_template(
    chat,
    tools=tools,
    return_tensors="pt",
    tokenize=False,
    add_generation_prompt=True
)

print(tool_prompt)
