import requests
from typing import Optional
from dataclasses import dataclass

@dataclass
class FontMetadata:
    name: str
    url: str
    kind: str
    category: str

class GoogleFontsSource:
    def __init__(self, api_url: str, api_key: str):
        if not api_key or not api_url:
            raise Exception("Valid API information is required")
        self.api_url = api_url
        self.api_key = api_key

    def fetch_font_list(self) -> list[FontMetadata]:
        """Fetch list of fonts from Google Fonts API."""
        params = {'key': self.api_key}

        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch font list: {e}")
            return []

        fonts_data = response.json().get("items", [])
        font_list = []
        for font in fonts_data:
            family = font.get("family")
            files = font.get("files", {})
            # Prefer regular or 400 weight as a baseline. Could collect both?
            url = files.get("regular") or files.get("400") or next(iter(files.values()), None)

            if family and url:
                data = FontMetadata(
                    name=family.replace(" ", "_"), 
                    url=url,
                    category=font.get("category"),
                    kind=font.get("kind"))
                font_list.append(data)

        return font_list


if __name__ == '__main__':
    api_url = "https://www.googleapis.com/webfonts/v1/webfonts"
    api_key = "AIzaSyCSvL4yB61uFOu_V6pjItmkCs0d6zSfuKg"

    font_list = GoogleFontsSource(api_url, api_key).fetch_font_list()
    for font in font_list[:10]:
        print(font)
    