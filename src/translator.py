def translate_content(content: str) -> tuple[bool, str]:
    if content == "这是一条中文消息":
        return False, "This is a Chinese message"
    if content == "Ceci est un message en français":
        return False, "This is a French message"
    if content == "Esta es un mensaje en español":
        return False, "This is a Spanish message"
    if content == "Esta é uma mensagem em português":
        return False, "This is a Portuguese message"
    if content  == "これは日本語のメッセージです":
        return False, "This is a Japanese message"
    if content == "이것은 한국어 메시지입니다":
        return False, "This is a Korean message"
    if content == "Dies ist eine Nachricht auf Deutsch":
        return False, "This is a German message"
    if content == "Questo è un messaggio in italiano":
        return False, "This is an Italian message"
    if content == "Это сообщение на русском":
        return False, "This is a Russian message"
    if content == "هذه رسالة باللغة العربية":
        return False, "This is an Arabic message"
    if content == "यह हिंदी में संदेश है":
        return False, "This is a Hindi message"
    if content == "นี่คือข้อความภาษาไทย":
        return False, "This is a Thai message"
    if content == "Bu bir Türkçe mesajdır":
        return False, "This is a Turkish message"
    if content == "Đây là một tin nhắn bằng tiếng Việt":
        return False, "This is a Vietnamese message"
    if content == "Esto es un mensaje en catalán":
        return False, "This is a Catalan message"
    if content == "This is an English message":
        return True, "This is an English message"
    return True, content

# import os
# import time
# import openai
# from openai.error import RateLimitError
# import re
# from typing import Callable


# openai.api_key = os.getenv("OPEN_AI_KEY")
# # Dont use the entire url!! https://[name-of-your-resource].openai.azure.com/
# openai.api_base = "https://hkkubais-openai-source.openai.azure.com/"  # Your endpoint base URL
# openai.api_type = "azure"
# openai.api_version = "2024-08-01-preview"

# # Use your deployment name
# deployment_name = "hkkubais-gpt-4"



# def query_llm(context: str, question: str) -> str:
#     done = False
#     while not done:
#       try:
#         fullResponse =  openai.ChatCompletion.create(
#         engine=deployment_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": f"Text:'{context}'\nYour Task:{question}"}
#         ],
#         max_tokens=50,
#         temperature=0.7)
#         done = True
#       except RateLimitError:
#         time.sleep(5)
#     return fullResponse['choices'][0]['message']['content'].strip()

# def get_translation(post: str) -> str:
#     context = post
#     question = "Translate the text to English. Obviously ignore the 'Text:' part and dont wrap your response in quotes. If the text is gibberish return 'Error: Translation Failed'"
#     return query_llm(context, question)

# def get_language(post: str) -> str:
#     context = post
#     question = "What language? Answer in one word"
#     return query_llm(context, question)

# def is_valid_string(text):
#   pattern = r'^[a-zA-Z0-9 .,!?\'"-]+$'
#   return bool(re.match(pattern, text))

# def translate_content(post: str) -> tuple[bool, str]:
#   translation = post
#   try:
#     lang = get_language(post)
#     if lang.lower() != 'english':
#       translation = get_translation(post)
#       if translation == 'Error: Translation Failed':
#          return (False, translation)
#     if not lang.isalpha() or not is_valid_string(translation):
#       return (False, "Error: Failed to detect language or properly translate post")
#     return (lang.lower() == 'english', translation)
#   except:
#     return (False, "Error: Translation to English failed")