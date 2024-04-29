"""The script tests whether the imported module's functionalities work well.


Submitted by Candice Yao(jy3440)
This module implements the testings using module unittest. It runs three tests by accessing elements from a list.
Each string represents different aspects of problems we would want to catch.
"""
import unittest
import jy3440_summarized_text

summaries = ["This is a simple, but useful example! However, "
             "there are many cases not included here...",
             "This is pre-made lemonade. However, it tastes like "
             "it was 'homemade' a week ago.",
             """Testing the behavior of empty words ;: "; ""."""]


expected_outputs = ['T2s is a s4e, b1t u4l e5e! H5r, t3e a1e m2y c3s n1t i6d '
                    'h2e...',
                    "T2s is p6e l6e. H5r, it t4s l2e it w1s 'h6e' a w2k a1o.",
                    'T5g t1e b6r of e3y w3s ;: "; "".']


class TestSummarizeText(unittest.TestCase):
    def test_string_1(self):
        """running the first test on a normal string"""
        test_string = summaries[0]
        expected_output = expected_outputs[0]
        self.assertEqual(ma6918_summarized_text.summarize_text(test_string), expected_output)

    def test_string_2(self):
        """running the second test that might contain the '-' problem"""
        test_string = summaries[1]
        expected_output = expected_outputs[1]
        self.assertEqual(ma6918_summarized_text.summarize_text(test_string), expected_output)

    def test_string_3(self):
        """running the thrid test that might contain the empty-string problem"""
        test_string = summaries[2]
        expected_output = expected_outputs[2]
        self.assertEqual(ma6918_summarized_text.summarize_text(test_string), expected_output)


if __name__ == '__main__':
    unittest.main()
