from pathlib import Path
import requests

def make_api_call(api_request: str, save_path: Path) -> bool:
    response: requests.Response = requests.get(api_request, timeout=10)
    response.raise_for_status()
    print("API call successful., Response: ", response)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if 'API rate limit' in response.text:
        print("API rate limit reached. Data not saved.")
        return False
    save_path.write_bytes(response.content)
    return True