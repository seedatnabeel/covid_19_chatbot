import argparse
import logging

import pandas as pd
from bs4 import BeautifulSoup
from requests import get

logging.getLogger().setLevel(logging.INFO)


def scrape_web_url(url):
    """
    Scrapes web url with BeautifulSoup and returns a soup object

    Args:

    url(str): string of the url to scrape

    Returns:
        soup: soup object
    """

    logging.info("Getting soup from url")
    # get request on the url
    response = get(url)
    soup = BeautifulSoup(response.text)

    return soup


def q_a_pairs_to_df(soup):
    """
    Writes the QA pairs to a pandas dataframe

    Args:

    soup: soup object

    Returns:
        df: pandas dataframe of QA pairs
    """

    def get_qa_pair(q, a, idx):
        return {
            "Question": q.get_text().strip(),
            "Answer": a.get_text().strip().split(".")[idx],
        }

    logging.info("Writing QA pairs to a pandas df")
    # get the Question and Answers from the soup
    q_items = soup.findAll("a", {"class": "sf-accordion__link"})
    a_items = soup.findAll("p", {"class": "sf-accordion__summary"})

    # combine q and a pairs
    q_a_pairs = []
    for i in range(len(q_items)):
        n_answers = len(a_items[i].get_text().strip().split(".")[0:-1])

        for j in range(n_answers):
            q_a_pair = get_qa_pair(q_items[i], a_items[i], j)
            q_a_pairs.append(q_a_pair)

    return pd.DataFrame(q_a_pairs)


def write_df_to_h5(df, file_name):
    """
    Writes the dataframe to a .h5 file

    Args:

    df: pandas DataFrame
    file_name (str): name of the h5 file

    """
    logging.info("Writing df to h5 file")
    df.to_hdf(file_name, key="df", mode="w")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Acquire data into pandas df from a web url"
    )
    parser.add_argument(
        "--url",
        help="Web url to scrape",
        type=str,
        default="https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub/q-a-detail/q-a-coronaviruses",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Filename to name your H5",
        default="ref_faqs.h5",
    )

    args = parser.parse_args()

    url = args.url
    file_name = args.output_file_name

    soup = scrape_web_url(url)
    df = q_a_pairs_to_df(soup)
    write_df_to_h5(df, file_name)
