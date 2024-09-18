import os
import json
import uuid

from llama_index.readers.json import JSONReader
from llama_index.core import Document


class CustomJSONReader(JSONReader):
    def load_data(self, input_file):
        with open(input_file, "r") as file:
            data = json.load(file)

        documents = []
        for item in data.get("headings", []):
            documents.append(
                Document(
                    id_=str(uuid.uuid4()),
                    text=item["text"],
                    metadata={"reference": item["reference"]},
                )
            )

        for item in data.get("verses", []):
            documents.append(
                Document(
                    id_=str(uuid.uuid4()),
                    text=item["text"],
                    metadata={"reference": item["reference"]},
                )
            )

        return documents

    def load_dir(self, input_dir: dir):
        documents = []
        for file in os.listdir(input_dir):
            if file.endswith(".json"):
                documents.extend(self.load_data(os.path.join(input_dir, file)))
        return documents
