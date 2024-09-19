# client.py
import os
import requests
import json
import argparse

# replace this URL with your exposed URL from the API builder. The URL looks like this
# SERVER_URL = 'https://8000-01hxj54gh5yry3bpaw5k8s5t5j.cloudspaces.litng.ai'
# get environment variable
SERVER_URL = os.getenv("LDS_RAG_URL")


def main():
    print(
        "Hello! Welcome to the LDS Scripture Search! Enter a question to get started:"
    )

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        payload = {"query": query}

        try:
            response = requests.post(f"{SERVER_URL}/predict", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            result = response.json()["output"]
            print(json.dumps(result, indent=2))
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")


if __name__ == "__main__":
    main()
