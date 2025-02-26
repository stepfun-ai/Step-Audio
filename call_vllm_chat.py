import argparse
from openai import OpenAI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="VLLM server url.",
    )
    parser.add_argument(
        "--model-name", type=str, default="step-audio-chat", help="Model name."
    )
    args = parser.parse_args()

    server_url = args.server_url + "/v1"  # for chat route
    client = OpenAI(base_url=server_url, api_key="whatever")

    messages = [
        {
            "role": "system",
            "content": "You are an AI designed for conversation, currently unable to connect to the internet.",
        },
        {"role": "user", "content": "Introduce yourself."},
    ]
    completion = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
    )
    res = completion.choices[0].message.content
    print(res)
