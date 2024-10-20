# HUGPI: Unleash the Power of Large Language Models 🚀

[![PyPI version](https://badge.fury.io/py/hugpi.svg)](https://badge.fury.io/py/hugpi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/hugpi.svg)](https://pypi.org/project/hugpi/)
[![Downloads](https://pepy.tech/badge/hugpi)](https://pepy.tech/project/hugpi)

HUGPI is a Python library that democratizes access to state-of-the-art language models. By leveraging Hugging Face's freely available large language models, HUGPI empowers developers to build sophisticated AI applications without the need for expensive API subscriptions or complex infrastructure.

## 🌟 Why HUGPI?

- 🆓 Access cutting-edge AI models at no cost
- 🔧 Unified API inspired by industry standards like OpenAI and Anthropic
- 🛠 Extend model capabilities with custom tools and function calling
- 🌊 Real-time interactions with streaming responses
- 🧠 Effortless conversation management for context-aware applications

HUGPI is your gateway to creating next-generation AI solutions, from chatbots and content generators to advanced reasoning systems and beyond. Harness the full potential of large language models and bring your ideas to life!

## 📦 Installation

Install HUGPI using pip:

```bash
pip install hugpi
```

## 🚀 Quick Start

Here's a simple example to get you started with HUGPI:

```python
from hugpi import HUGPIClient

# Initialize the client
email = 'your_huggingface_email@example.com'
password = 'your_huggingface_password'
api_key = f'{email}_{password}'

client = HUGPIClient(model='Qwen/Qwen2.5-72B-Instruct', api_key=api_key)

# Create a simple message
response = client.messages.create(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=100
)

print(response.content[0]['text'])
```

## 🤖 Available Models

HUGPI supports a wide range of powerful language models:

- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `CohereForAI/c4ai-command-r-plus-08-2024`
- `Qwen/Qwen2.5-72B-Instruct`
- `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `NousResearch/Hermes-3-Llama-3.1-8B`
- `mistralai/Mistral-Nemo-Instruct-2407`
- `microsoft/Phi-3.5-mini-instruct`

## 🛠 Features and Examples

### 1. Basic Message Creation

Create a simple message and get a response:

```python
response = client.messages.create(
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
    max_tokens=150
)
print(response.content[0]['text'])
```

### 2. Conversation Management

Maintain context across multiple messages:

```python
conversation = client.messages.create(
    messages=[{"role": "user", "content": "Let's talk about space exploration."}],
    conversation=True
)
print(conversation.content[0]['text'])

follow_up = client.messages.create(
    messages=[{"role": "user", "content": "What are the biggest challenges?"}],
    conversation=True
)
print(follow_up.content[0]['text'])
```

### 3. Streaming Responses

Get real-time responses for a more interactive experience:

```python
for chunk in client.messages.create(
    messages=[{"role": "user", "content": "Write a short story about a time traveler."}],
    max_tokens=200,
    stream=True
):
    print(chunk.content[0]['text'], end='', flush=True)
```

### 4. Tool Calling

Extend the model's capabilities with custom functions:

```python
def calculate_area(length: float, width: float):
    """Calculate the area of a rectangle."""
    return length * width

def get_current_time():
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

response = client.messages.create(
    max_tokens=1024,
    tools=[calculate_area, get_current_time],
    messages=[{"role": "user", "content": "What's the area of a 5x3 rectangle, and what time is it now?"}]
)

print(response.content[0])
```

### 5. Model Switching

Easily switch between different models:

```python
client_llama = HUGPIClient('meta-llama/Llama-3.2-11B-Vision-Instruct', api_key=api_key)
client_qwen = HUGPIClient('Qwen/Qwen2.5-72B-Instruct', api_key=api_key)

response_llama = client_llama.messages.create(
    messages=[{"role": "user", "content": "Describe the process of photosynthesis."}]
)
print("Llama response:", response_llama.content[0]['text'])

response_qwen = client_qwen.messages.create(
    messages=[{"role": "user", "content": "Describe the process of photosynthesis."}]
)
print("Qwen response:", response_qwen.content[0]['text'])
```

### 6. Advanced Prompting

Use system messages to set the tone or context for the conversation:

```python
response = client.messages.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant with expertise in environmental science."},
        {"role": "user", "content": "What are some effective ways to reduce carbon emissions?"}
    ],
    max_tokens=200
)
print(response.content[0]['text'])
```

### 7. Error Handling

Implement error handling to manage potential issues:

```python
try:
    response = client.messages.create(
        messages=[{"role": "user", "content": "Translate this to French: Hello, world!"}],
        max_tokens=50
    )
    print(response.content[0]['text'])
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

## 📊 Performance and Scalability

HUGPI is designed for high-performance scenarios:

- Optimized API calls
- Support for concurrent requests

## 🙏 Acknowledgements

HUGPI stands on the shoulders of giants:

- [Hugging Face](https://huggingface.co/) for their commitment to open-source AI and providing access to state-of-the-art language models.
- [Transformers](https://github.com/huggingface/transformers) library, which forms the backbone of our model interactions.
- [hugchat](https://github.com/Soulter/hugging-chat-api) package, whose groundwork in making Hugging Face models more accessible inspired and informed our development.

We extend our heartfelt gratitude to these projects and the entire open-source AI community for making advanced AI accessible to all.

## 🤝 Contributing

We welcome contributions! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for more information on how to get started.

## 📜 License

HUGPI is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hugpi/hugpi&type=Date)](https://star-history.com/#hugpi/hugpi&Date)

## 📚 Documentation

For full documentation, visit our [official documentation site](https://hugpi.readthedocs.io/).

## 💬 Community and Support

Join our [Discord community](https://discord.gg/hugpi) for discussions, support, and to connect with other HUGPI users.

---

HUGPI - Empowering developers with cutting-edge language model capabilities. Start building amazing AI-powered applications today! 🚀🤖