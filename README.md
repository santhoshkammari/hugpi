# HUGPI: High-Performance Language Model Inference 🚀

[![PyPI version](https://badge.fury.io/py/hugpi.svg)](https://badge.fury.io/py/hugpi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/hugpi.svg)](https://pypi.org/project/hugpi/)
[![Downloads](https://pepy.tech/badge/hugpi)](https://pepy.tech/project/hugpi)

HUGPI is a powerful and flexible Python library for seamless interaction with state-of-the-art language models. It provides a unified interface for working with multiple models, supporting features similar to OpenAI's GPT models, Anthropic's Claude, and seamless integration with LangChain.

## 🌟 Features

- 🤖 Support for multiple cutting-edge language models
- 🔧 Easy-to-use API similar to OpenAI and Anthropic Claude
- 🛠 Built-in tool calling functionality
- 📊 Usage tracking and token counting
- 🌊 Streaming responses for real-time output
- 🧠 Conversation management for context-aware interactions
- 🔌 LangChain compatibility

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
client = HUGPIClient(model='Qwen/Qwen2.5-72B-Instruct', api_key='your_email@gmail.com_your_password')

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

## 🛠 Advanced Usage

### Tool Calling

HUGPI supports tool calling, allowing you to extend the model's capabilities with custom functions:

```python
from hugpi import HUGPIClient

def multiply(x: float, y: float):
    """Multiply two numbers."""
    return int(x) * int(y)

def google_search(query: str):
    """Simulate a Google search."""
    return f"Search results for: {query}"

client = HUGPIClient('Qwen/Qwen2.5-72B-Instruct')
response = client.messages.create(
    max_tokens=1024,
    tools=[multiply, google_search],
    messages=[{"role": "user", "content": "What's 5 times 3, and can you search for 'AI advancements'?"}]
)

print(response.content[0])
```

### Streaming Responses

For real-time output, use the streaming feature:

```python
from hugpi import HUGPIClient

client = HUGPIClient('nvidia/Llama-3.1-Nemotron-70B-Instruct-HF')
for chunk in client.messages.create(
    messages=[{"role": "user", "content": "Write a short story about a robot learning to love."}],
    max_tokens=200,
    stream=True
):
    print(chunk.content[0]['text'], end='', flush=True)
```

## 🔗 LangChain Integration

HUGPI is designed to work seamlessly with LangChain:

```python
from langchain.llms import HuggingFaceTextGenInference
from hugpi import HUGPIClient

hugpi_client = HUGPIClient('meta-llama/Llama-3.2-11B-Vision-Instruct')
llm = HuggingFaceTextGenInference(inference_server_url="http://localhost:8080", max_new_tokens=512)

# Use HUGPI with LangChain components
from langchain import PromptTemplate, LLMChain

template = "Write a {adjective} poem about {subject}."
prompt = PromptTemplate(template=template, input_variables=["adjective", "subject"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

result = llm_chain.run(adjective="whimsical", subject="artificial intelligence")
print(result)
```

## 📊 Performance and Scalability

HUGPI is designed for high-performance scenarios:

- Efficient token management
- Optimized API calls
- Support for concurrent requests

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