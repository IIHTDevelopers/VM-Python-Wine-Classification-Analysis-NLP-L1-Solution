import unittest
import pandas as pd
from test.TestUtils import TestUtils
from winenlptemplate import process_reviews, train_model, vectorize_and_add_features, TfidfVectorizer

class ExceptionalTest(unittest.TestCase):

    def test_empty_dataframe_exception(self):
        # Create an instance of TestUtils for yaksha assertions
        test_obj = TestUtils()

        # Create an empty DataFrame with the required columns
        empty_df = pd.DataFrame(columns=['review_title', 'review_description', 'points', 'price'])

        try:
            # Trying to process reviews on an empty DataFrame
            processed_df = process_reviews(empty_df)  # This should handle empty DataFrame gracefully

            # Check if the DataFrame is still empty after processing
            if processed_df.empty:
                test_obj.yakshaAssert("TestEmptyDataFrame", True, "boundary")
                print("TestEmptyDataFrame = Passed")
            else:
                test_obj.yakshaAssert("TestEmptyDataFrame", False, "boundary")
                print("TestEmptyDataFrame = Failed")

        except Exception as e:
            # If any unexpected exception occurs, the test fails
            test_obj.yakshaAssert("TestEmptyDataFrame", False, "boundary")
            print(f"TestEmptyDataFrame = Failed with exception: {str(e)}")



if __name__ == '__main__':
    unittest.main()
