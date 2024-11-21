from functools import wraps
import markdown
import re
import uuid

from genservice.openai_client import client


# JSON to dialogue conversion
def convert_json_to_dialogue(data: dict):
    dialogue = ""

    for item in data:
        sentence = item.get('sentence', '')
        speaker_name = item.get('speaker_name', 'unknown')

        if len(sentence) >= 20:
            if speaker_name == 'speaker 1':
                speaker = "Interviewer"
            elif speaker_name == 'speaker 2':
                speaker = "Candidate"
            else:
                speaker = "Unknown"

            dialogue += f"{speaker}: {sentence}\n"

    return dialogue


# JSON minification
def process_json(data: dict):
    if not isinstance(data, dict):
        return data

    for item in data:
        if isinstance(item, dict):
            item.pop('speaker_id', None)
        else:
            pass

    return data


# OpenAI Request Decorator
def openai_request_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        payload = await func(*args, **kwargs)
        response = await client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload["temperature"],
            top_p=payload["top_p"]
        )
        print("response has been collected")
        choices = response.choices
        extracted_content = choices[0].message.content

        return extracted_content

    return wrapper


# OpenAI Requirement Handling
def parse_requirements(requirements: str):
    requirements_list = [line.strip(' ') for line in requirements.strip().split('\n') if line]
    categories_listed = []

    for category in requirements_list:
        categories_listed.append(category)

    return categories_listed


# Hardcoded variables for sections
general_requirements = [
    "Introduction",
    "Main responsibilities",
    "Git/CI/CD",
    "Scrum/Kanban/Waterfall",
    "SOLID/Object Oriented Programming/Design Patterns"
]


# High-order payload builder
def payload_builder(fsp: str, prompt: str, system_content: str, model: str, temperature: float, top_p: float):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Below is an example of the expected format. Do not quote this example in your response. Instead, follow this structure:"},
            {"role": "user", "content": fsp},
            {"role": "user", "content": "Here is the task with the actual data:"},
            {"role": "user", "content": prompt}
        ],
        "top_p": top_p,
        "temperature": temperature
    }

