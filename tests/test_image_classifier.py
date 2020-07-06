import unittest
from lang_model.train_lang_embedding import *

class Test_Acquire_Data(unittest.TestCase):

    def test_load_ref_data(self):
        import pandas as pd
        df = load_ref_data("data/ref_faqs.h5")
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_preprocess_sentences(self):
        import pandas as pd
        input_sentences = ['covid', 'covid-19']
        sentences = preprocess_sentences(input_sentences)
        self.assertEqual('coronavirus', sentences[0])
        self.assertEqual('coronavirus', sentences[1])

if __name__ == '__main__':
    unittest.main()
