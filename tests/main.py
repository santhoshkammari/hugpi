import hugpi
def multiply(x: float, y: float):
    '''
    A function that multiplies two numbers

    Args:
        x: The first number to multiply
        y: The second number to multiply
    return x * y
    '''
    return int(x) * int(y)

def google_search(query:str):
    '''
    A function that uses Google search to find information about a query

    Args:
        query: The query to search for
    Returns:
        The top result from Google search
    '''
    return "he is great person"
client = hugpi.HUGPIClient('Qwen/Qwen2.5-72B-Instruct')
response = client.messages.create(
    max_tokens=1024,
    tools=[google_search],
    messages=[{"role": "user", "content": "what are some achievements of king kohli?"}],
    debug = True
)
print(response.content[0])


