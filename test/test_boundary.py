import unittest
import pandas as pd
from test.TestUtils import TestUtils
from winenlptemplate import process_reviews, train_model, vectorize_and_add_features, TfidfVectorizer

class BoundaryTest(unittest.TestCase):

    def test_boundary_values(self):
        # Create an instance of TestUtils for yaksha assertions
        test_obj = TestUtils()

        # Create a DataFrame with boundary values for 'points' and 'price'
        boundary_df = pd.DataFrame({
            'review_title': ['Best wine', 'Worst wine'],
            'review_description': ['Amazing taste, high value', 'Terrible taste, low value'],
            'points': [0, 100],  # Boundary values for points
            'price': [0.0, 10000.0]  # Boundary values for price
        })

        try:
            # Process reviews
            processed_df = process_reviews(boundary_df)

            # Vectorize reviews using TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000)
            vectorizer.fit(processed_df['review_title'] + ' ' + processed_df['review_description'])

            # Prepare features and labels for training
            X_train = vectorize_and_add_features(processed_df, vectorizer)
            Y_train = [0, 1]  # Dummy labels for this test

            # Train the model with boundary values
            model = train_model(X_train, Y_train)

            # If the model trains successfully, the test passes
            test_obj.yakshaAssert("TestBoundaryValues", True, "boundary")
            print("TestBoundaryValues = Passed")
        except Exception as e:
            # If any exception occurs, the test fails
            test_obj.yakshaAssert("TestBoundaryValues", False, "boundary")
            print(f"TestBoundaryValues = Failed with exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()
