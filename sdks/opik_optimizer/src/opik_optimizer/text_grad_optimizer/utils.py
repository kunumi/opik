import re
from typing import List, Dict, Any

def extract_text_between_tags(text: str, tag: str) -> str:
    """
    Extracts text between specified start and end tags.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return matches[0] if matches else ""

class BatchLoader:
    def __init__(self, dataset: List, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> List[Dict[str, Any]]:
        if self.current_index >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch