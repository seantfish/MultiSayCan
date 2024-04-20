#!/usr/bin/env python3
from openai import OpenAI
from openlogprobs import extract_logprobs

client = OpenAI(
    base_url="http://localhost:8000/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)
# completion = client.chat.completions.create(
#    model="LLaMA_CPP",
#    messages=[
#        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority #is achieving user fulfillment via helping them with their requests."},
#        {"role": "user", "content": "Write a limerick about python exceptions"}
#    ]
# )
CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline"""

# models = openai.Model.list()
# print("Models:", models)

# model = models["data"][0]["id"]

# Completion API
# stream = False
# completion = client.chat.completions.create(
#     model="LLaMA_CPP",
#     messages=[
#         {"role": "system", "content": "A robot may not injure a human being"}
#         ],
#     logprobs=True)

# print("Completion results:")
# if stream:
#     for c in completion:
#         print(c)
# else:
#     print(completion)

# completion = client.chat.completions.create(
#     model="LLaMA_CPP",
#     # temperature=0,
#     # stop=None,
#     # seed=123,
#     # tools=None,
#     # logprobs=True,
#     # top_logprobs=2,
#     messages=[
#         {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority #is achieving user fulfillment via helping them with their requests."},
#         {"role": "user", "content": "Write a limerick about python exceptions"}
#     ]
# )
# print(completion.choices[0].message.content)

stream = client.completions.create(
    model="model",
    prompt="hello there",
    # stream=True,
    echo=True,
    logprobs=5
)
# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")

print(stream)
# def get_completion(
#     messages: list[dict[str, str]],
#     model: str = "LLaMA_CPP",
#     max_tokens=00,
#     temperature=0,
#     stop=None,
#     seed=123,
#     tools=None,
#     logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
#     top_logprobs=None,
# ) -> str:
#     params = {
#         "model": model,
#         "messages": messages,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "stop": stop,
#         "seed": seed,
#         "logprobs": logprobs,
#         "top_logprobs": top_logprobs,
#     }
#     if tools:
#         params["tools"] = tools

#     completion = client.chat.completions.create(**params)
#     return completion

#extract_logprobs("LLaMA_CPP", "i like pie", method="topk")
