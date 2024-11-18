from src.translator import translate_content
from sentence_transformers import SentenceTransformer, util
import openai
from mock import patch
from typing import Callable
from openai.error import RateLimitError

model = SentenceTransformer('all-MiniLM-L6-v2')

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
    bool1, string1 = expected_answer
    bool2, string2 = llm_response
    if bool1 != bool2:
        return 0.0
    sentences = [string1, string2]
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)
    sim_score = float(similarities[0][1])
    return min(sim_score, 1.0)

def test_llm_normal_response():
    eval_score = evaluate(translate_content, eval_single_response_complete, complete_eval_set)
    assert (eval_score > 90);

def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
    totalScore = 0
    for data in dataset:
        actual_response = query_fn(data['post'])
        score = eval_fn(actual_response, data['expected_answer'])
        # print("Expected: "+ data['expected_answer'])
        # print("Actual: "+ actual_response)
        # print("Score: ", score)
        totalScore += score
    return (totalScore*100)/(len(dataset))

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response1(mocker):
  # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "This is th%e trnslation"

  # TODO assert the expected behavior
    res = translate_content("Hier ist dein erstes Beispiel.")
    assert res == (False, "Error: Translation to English failed")

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response2(mocker):
  # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "This is the tr$nslation"

  # TODO assert the expected behavior
    res = translate_content("Hello! This is a test.")
    assert res == (False, "Error: Translation to English failed")

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response3(mocker):
  # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "Hello 😊"

  # TODO assert the expected behavior
    res = translate_content("Это просто проверка системы.")
    assert res == (False, "Error: Translation to English failed")

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response4(mocker):
  # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "Café"

  # TODO assert the expected behavior
    res = translate_content("Możesz mi powiedzieć, gdzie jest najbliższa stacja metra?")
    assert res == (False, "Error: Translation to English failed")

@patch.object(openai.ChatCompletion, 'create')
def test_llm_gibberish_response5(mocker):
  # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "1+1=3"
  # TODO assert the expected behavior
    res = translate_content("I have a question. When is the midterm?")
    assert res == (False, "Error: Translation to English failed")

complete_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "This is your first example.")
    },
    {
        "post": "Ceci est un test de traduction.",
        "expected_answer": (False, "This is a translation test.")
    },
    {
        "post": "¿Cómo estás hoy?",
        "expected_answer": (False, "How are you today?")
    },
    {
        "post": "今日は素晴らしい天気ですね。",
        "expected_answer": (False, "It's great weather today.")
    },
    {
        "post": "Это просто проверка системы.",
        "expected_answer": (False, "This is just a system check.")
    },
    {
        "post": "Estou ansioso para o evento de amanhã!",
        "expected_answer": (False, "I'm excited for tomorrow's event!")
    },
    {
        "post": "Możesz mi powiedzieć, gdzie jest najbliższa stacja metra?",
        "expected_answer": (False, "Can you tell me where the nearest metro station is?")
    },
    {
        "post": "Je suis actuellement en train d'apprendre une nouvelle langue.",
        "expected_answer": (False, "I am currently learning a new language.")
    },
    {
        "post": """أعتقد أن الاجتماعات التي نجريها كل أسبوع لها تأثير إيجابي كبير على الفريق، لأنها تمنحنا الفرصة لمناقشة التحديات التي نواجهها وتبادل الأفكار حول كيفية تحسين الأداء. هل يمكنك مساعدتي في تقديم اقتراحات إضافية يمكننا مناقشتها في الاجتماع القادم؟""",
        "expected_answer": (False, """I believe that the meetings we have every week\
 have a huge positive impact on the team, because\
 they give us the opportunity to discuss the\
 challenges we face and exchange ideas on how to\
 improve performance. Can you help me make\
 additional suggestions that we can discuss at the next meeting?""")
    },
    {
        "post": "우리는 오늘 저녁 회의에서 이 문제를 논의할 예정입니다.",
        "expected_answer": (False, "We plan to discuss this issue in the meeting tonight.")
    },
    {
        "post": "Sono felice di vederti dopo tutto questo tempo!",
        "expected_answer": (False, "I'm happy to see you after all this time!")
    },
    {
        "post": "今天的会议非常有趣，但我仍然有一些问题。",
        "expected_answer": (False, "Today's meeting was very interesting, but I still have some questions.")
    },
    {
        "post": "Merci beaucoup pour votre aide précieuse!",
        "expected_answer": (False, "Thank you very much for your valuable help!")
    },
    {
        "post": "¿Puedes recomendarme un buen restaurante en esta zona?",
        "expected_answer": (False, "Can you recommend a good restaurant in this area?")
    },
    {
        "post": "Ich habe gestern einen sehr interessanten Artikel über künstliche Intelligenz gelesen.",
        "expected_answer": (False, "Yesterday, I read a very interesting article about artificial intelligence.")
    },
    {
        "post": "I have a question. When is the midterm?",
        "expected_answer": (True, "I have a question. When is the midterm?")
    },
        {
        "post": "Can you help me finish this project by Friday?",
        "expected_answer": (True,"Can you help me finish this project by Friday?")
    },
    {
        "post": "I had an amazing time at the concert last night!",
        "expected_answer": (True,"I had an amazing time at the concert last night!")
    },
    {
        "post": "Don't forget to bring your laptop to the meeting tomorrow.",
        "expected_answer": (True,"Don't forget to bring your laptop to the meeting tomorrow.")
    },
    {
        "post": "What do you think about the new design proposal?",
        "expected_answer": (True,"What do you think about the new design proposal?")
    },
    {
        "post": "I need to buy some groceries on my way home today.",
        "expected_answer": (True,"I need to buy some groceries on my way home today.")
    },
    {
        "post": "It's been raining all day, and it's really cold outside.",
        "expected_answer": (True,"It's been raining all day, and it's really cold outside.")
    },
    {
        "post": "Could you send me the updated report before noon?",
        "expected_answer": (True,"Could you send me the updated report before noon?")
    },
    {
        "post": "I'm not sure if I can attend the event this weekend.",
        "expected_answer": (True,"I'm not sure if I can attend the event this weekend.")
    },
    {
        "post": "The new restaurant in town has great reviews.",
        "expected_answer": (True,"The new restaurant in town has great reviews.")
    },
    {
        "post": "I'm planning to take a vacation next month.",
        "expected_answer": (True,"I'm planning to take a vacation next month.")
    },
    {
        "post": "Please let me know if you have any questions.",
        "expected_answer": (True,"Please let me know if you have any questions.")
    },
    {
        "post": "I'm currently reading an interesting book on psychology.",
        "expected_answer": (True,"I'm currently reading an interesting book on psychology.")
    },
    {
        "post": "We should schedule a meeting to discuss the project details.",
        "expected_answer": (True,"We should schedule a meeting to discuss the project details.")
    },
    {
        "post": "I'm really excited about the new software update.",
        "expected_answer": (True,"I'm really excited about the new software update.")
    },
    {
        "post": "Do you have any recommendations for good movies to watch?",
        "expected_answer": (True,"Do you have any recommendations for good movies to watch?")
    },
    {
        "post": "awekjfbqlarwdmqg;kl53jgpfovWM",
        "expected_answer": (False ,"Error: Translation Failed")
    },
    {
        "post": "ثصمبىصثخحخحصنثبخحصثنب",
        "expected_answer": (False ,"Error: Translation Failed")
    },
    {
        "post": "ثصمبىrqwerqwerحص ثمصنبثصنثبخحصثنب",
        "expected_answer": (False ,"Error: Translation Failed")
    },
    {
        "post": "ثصم:::خحصثنب",
        "expected_answer": (False ,"Error: Translation Failed")
    },
    {
        "post": "ثصمبىصثخحخح!!!!!!!!!!!!!!!!بخحصثنب",
        "expected_answer": (False ,"Error: Translation Failed")
    }
]