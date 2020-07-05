import unittest
from data.acquire_data import *

class Test_Acquire_Data(unittest.TestCase):

    def test_scrape_web_url(self):
        import bs4
        from bs4 import BeautifulSoup
        from requests import get

        url = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub/q-a-detail/q-a-coronaviruses"
        soup = scrape_web_url(url)
        self.assertIsInstance(soup, bs4.BeautifulSoup)

    def test_q_a_pairs_to_df(self):
        import bs4
        from bs4 import BeautifulSoup
        from requests import get

        url = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub/q-a-detail/q-a-coronaviruses"
        soup = scrape_web_url(url)
        df = q_a_pairs_to_df(soup)
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
