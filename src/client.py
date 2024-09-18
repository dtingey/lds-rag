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
    parser = argparse.ArgumentParser(
        description="Send a query to the Scripture Insight server."
    )
    parser.add_argument(
        "--query", type=str, required=True, help="The query text to send to the server."
    )

    args = parser.parse_args()

    payload = {"query": args.query}

    breakpoint()

    try:
        response = requests.post(f"{SERVER_URL}/predict", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        result = response.json()["output"]["response"]
        print(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")


if __name__ == "__main__":
    main()
