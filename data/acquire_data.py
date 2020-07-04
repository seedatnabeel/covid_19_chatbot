import pandas as pd
from bs4 import BeautifulSoup
from requests import get

url = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub/q-a-detail/q-a-coronaviruses"

response = get(url)
soup = BeautifulSoup(response.text)

q_items = soup.findAll("a", {"class": "sf-accordion__link"})
a_items = soup.findAll("p", {"class": "sf-accordion__summary"})


def get_qa_pair(q, a, idx):
    return {
        "Question": q.get_text().strip(),
        "Answer": a.get_text().strip().split(".")[idx],
    }


q_a_pairs = []

for i in range(len(q_items)):
    n_answers = len(a_items[i].get_text().strip().split(".")[0:-1])

    for j in range(n_answers):
        q_a_pair = get_qa_pair(q_items[i], a_items[i], j)
        q_a_pairs.append(q_a_pair)

df = pd.DataFrame(q_a_pairs)

df.to_csv("Covid_19_Questions_Answers_WHO.csv")
