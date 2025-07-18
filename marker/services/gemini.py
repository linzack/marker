import json
import time
import traceback
from io import BytesIO
from typing import List, Annotated

import PIL
from google import genai
from google.genai import types
from google.genai.errors import APIError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService
import re

logger = get_logger()


class BaseGeminiService(BaseService):
    gemini_model_name: Annotated[
        str, "The name of the Google model to use for the service."
    ] = "gemma-3-27b-it"

    def img_to_bytes(self, img: PIL.Image.Image):
        image_bytes = BytesIO()
        img.save(image_bytes, format="WEBP")
        return image_bytes.getvalue()

    def get_google_client(self, timeout: int):
        raise NotImplementedError

    def process_images(self, images):
        image_parts = [
            types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/webp")
            for img in images
        ]
        return image_parts

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_google_client(timeout=timeout)
        image_parts = self.format_image_for_llm(image)

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts
                    + [
                        prompt
                    ],  # According to gemini docs, it performs better if the image is the first element
                    config={
                        "temperature": 0,
                        #"response_schema": response_schema,
                        #"response_mime_type": "application/json", # json mode disabled for gemma
                    },
                )
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                print(f"prompt: {prompt}\n\n")
                print(f"output: {output}\n\n")

                ####### retry parse json
                MAX_JSON_PARSE_ATTEMPTS = 5
                json_string = output.strip()
                parsed_successfully = False
                for attempt in range(MAX_JSON_PARSE_ATTEMPTS):
                    try:
                        # Attempt to extract JSON from a markdown code block first
                        json_match = re.search(r"```json\n([\s\S]*?)\n```", json_string)
                        if json_match:
                            current_json_to_parse = json_match.group(1).strip()
                        else:
                            # If not in a markdown block, assume it's raw JSON
                            current_json_to_parse = json_string
        
                        response_json = json.loads(current_json_to_parse)
                        parsed_successfully = True
                        break # Exit loop if parsing is successful
                    except json.JSONDecodeError as e:
                        logger.warning(f"Attempt {attempt + 1}/{MAX_JSON_PARSE_ATTEMPTS}: Error decoding JSON from LLM response: {e}")
                        logger.debug(f"Raw JSON string causing error: {current_json_to_parse[:500]}...")
        
                        # Apply self-correction heuristics
                        # Heuristic 1: Escape unescaped double quotes
                        # This regex looks for a double quote that is NOT preceded by a backslash
                        json_string = re.sub(r'(?<=[^\\])"', r'\\"', json_string) 
                        
                        # Heuristic 2: Escape newlines within string values (simple approach)
                        # This is a more aggressive replacement and might need refinement
                        json_string = json_string.replace("\n", "\\n")
        
                        # Heuristic 3: Remove trailing commas before '}' or ']' (simple approach)
                        json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
        
                        logger.warning(f"Attempt {attempt + 1}: Applied self-correction. Retrying...")
                ####### retry parse json
                
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                #return json.loads(output)
                return response_json
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded
                    if tries == total_tries:
                        # Last attempt failed. Give up
                        logger.error(
                            f"APIError: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                        )
                        break
                    else:
                        wait_time = tries * self.retry_wait_time
                        logger.warning(
                            f"APIError: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                        )
                        time.sleep(wait_time)
                else:
                    logger.error(f"APIError: {e}")
                    break
            except Exception as e:
                logger.error(f"Exception: {e}")
                traceback.print_exc()
                break

        return {}


class GoogleGeminiService(BaseGeminiService):
    gemini_api_key: Annotated[str, "The Google API key to use for the service."] = None

    def get_google_client(self, timeout: int):
        return genai.Client(
            api_key=self.gemini_api_key,
            http_options={"timeout": timeout * 1000},  # Convert to milliseconds
        )
