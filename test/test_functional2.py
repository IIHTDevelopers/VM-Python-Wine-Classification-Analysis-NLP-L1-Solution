import unittest
import pickle
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier
import os
from test.TestUtils import TestUtils  # Assuming this is a custom testing utility class
from winereview import (load_data, process_reviews, most_frequent_word, longest_review, average_review_length,
                       shortest_review, most_common_adjective, most_frequent_review_title, average_points,
                       most_frequent_country, train_model, save_model, load_model, vectorize_and_add_features,
                       TfidfVectorizer, LabelEncoder)

class NLPFunctionalTest(unittest.TestCase):

    def setUp(self):
        # Load the training dataset
        self.df_train = load_data('train500.csv')

        # Process the dataset for the review-related functions
        self.df_train = process_reviews(self.df_train)

        # Expected values (static)
        self.expected_most_common_word = "good"
        self.expected_longest_review = "excellent rich flavorful"
        self.expected_avg_review_length = 18.62
        self.expected_shortest_review = "bad light crisp"
        self.expected_most_common_adjective = "good"
        self.expected_most_common_title = "Very Good"
        self.expected_avg_points = 89.18
        self.expected_most_common_country = "Spain"

    def test_most_frequent_word(self):
        test_obj = TestUtils()
        try:
            actual_most_common_word = most_frequent_word(self.df_train)
            if actual_most_common_word == self.expected_most_common_word:
                test_obj.yakshaAssert("TestMostCommonWord", True, "functional")
                print("TestMostCommonWord: Passed")
            else:
                test_obj.yakshaAssert("TestMostCommonWord", False, "functional")
                print(f"TestMostCommonWord: Failed - Expected: {self.expected_most_common_word}, Got: {actual_most_common_word}")
        except Exception as e:
            test_obj.yakshaAssert("TestMostCommonWord", False, "functional")
            print(f"TestMostCommonWord: Failed due to exception - {str(e)}")

    def test_longest_review(self):
        test_obj = TestUtils()
        try:
            actual_longest_review = longest_review(self.df_train)
            if actual_longest_review == self.expected_longest_review:
                test_obj.yakshaAssert("TestLongestReview", True, "functional")
                print("TestLongestReview: Passed")
            else:
                test_obj.yakshaAssert("TestLongestReview", False, "functional")
                print(f"TestLongestReview: Failed - Expected: {self.expected_longest_review}, Got: {actual_longest_review}")
        except Exception as e:
            test_obj.yakshaAssert("TestLongestReview", False, "functional")
            print(f"TestLongestReview: Failed due to exception - {str(e)}")

    def test_average_review_length(self):
        test_obj = TestUtils()
        try:
            actual_avg_review_length = average_review_length(self.df_train)
            if actual_avg_review_length is not None:
                actual_avg_review_length = round(actual_avg_review_length, 2)
                if actual_avg_review_length == self.expected_avg_review_length:
                    test_obj.yakshaAssert("TestAverageReviewLength", True, "functional")
                    print("TestAverageReviewLength: Passed")
                else:
                    test_obj.yakshaAssert("TestAverageReviewLength", False, "functional")
                    print(f"TestAverageReviewLength: Failed - Expected: {self.expected_avg_review_length}, Got: {actual_avg_review_length}")
            else:
                raise ValueError("Function returned None")
        except Exception as e:
            test_obj.yakshaAssert("TestAverageReviewLength", False, "functional")
            print(f"TestAverageReviewLength: Failed due to exception - {str(e)}")

    def test_shortest_review(self):
        test_obj = TestUtils()
        try:
            actual_shortest_review = shortest_review(self.df_train)
            if actual_shortest_review == self.expected_shortest_review:
                test_obj.yakshaAssert("TestShortestReview", True, "functional")
                print("TestShortestReview: Passed")
            else:
                test_obj.yakshaAssert("TestShortestReview", False, "functional")
                print(f"TestShortestReview: Failed - Expected: {self.expected_shortest_review}, Got: {actual_shortest_review}")
        except Exception as e:
            test_obj.yakshaAssert("TestShortestReview", False, "functional")
            print(f"TestShortestReview: Failed due to exception - {str(e)}")

    def test_most_common_adjective(self):
        test_obj = TestUtils()
        try:
            actual_most_common_adjective = most_common_adjective(self.df_train)
            if actual_most_common_adjective == self.expected_most_common_adjective:
                test_obj.yakshaAssert("TestMostCommonAdjective", True, "functional")
                print("TestMostCommonAdjective: Passed")
            else:
                test_obj.yakshaAssert("TestMostCommonAdjective", False, "functional")
                print(f"TestMostCommonAdjective: Failed - Expected: {self.expected_most_common_adjective}, Got: {actual_most_common_adjective}")
        except Exception as e:
            test_obj.yakshaAssert("TestMostCommonAdjective", False, "functional")
            print(f"TestMostCommonAdjective: Failed due to exception - {str(e)}")

    def test_most_frequent_review_title(self):
        test_obj = TestUtils()
        try:
            actual_most_frequent_title = most_frequent_review_title(self.df_train)
            if actual_most_frequent_title == self.expected_most_common_title:
                test_obj.yakshaAssert("TestMostFrequentTitle", True, "functional")
                print("TestMostFrequentTitle: Passed")
            else:
                test_obj.yakshaAssert("TestMostFrequentTitle", False, "functional")
                print(f"TestMostFrequentTitle: Failed - Expected: {self.expected_most_common_title}, Got: {actual_most_frequent_title}")
        except Exception as e:
            test_obj.yakshaAssert("TestMostFrequentTitle", False, "functional")
            print(f"TestMostFrequentTitle: Failed due to exception - {str(e)}")

    def test_average_points(self):
        test_obj = TestUtils()
        try:
            actual_avg_points = average_points(self.df_train)
            if actual_avg_points is not None:
                actual_avg_points = round(actual_avg_points, 2)
                if actual_avg_points == self.expected_avg_points:
                    test_obj.yakshaAssert("TestAveragePoints", True, "functional")
                    print("TestAveragePoints: Passed")
                else:
                    test_obj.yakshaAssert("TestAveragePoints", False, "functional")
                    print(f"TestAveragePoints: Failed - Expected: {self.expected_avg_points}, Got: {actual_avg_points}")
            else:
                raise ValueError("Function returned None")
        except Exception as e:
            test_obj.yakshaAssert("TestAveragePoints", False, "functional")
            print(f"TestAveragePoints: Failed due to exception - {str(e)}")

    def test_most_frequent_country(self):
        test_obj = TestUtils()
        try:
            actual_most_frequent_country = most_frequent_country(self.df_train)
            if actual_most_frequent_country == self.expected_most_common_country:
                test_obj.yakshaAssert("TestMostFrequentCountry", True, "functional")
                print("TestMostFrequentCountry: Passed")
            else:
                test_obj.yakshaAssert("TestMostFrequentCountry", False, "functional")
                print(f"TestMostFrequentCountry: Failed - Expected: {self.expected_most_common_country}, Got: {actual_most_frequent_country}")
        except Exception as e:
            test_obj.yakshaAssert("TestMostFrequentCountry", False, "functional")
            print(f"TestMostFrequentCountry: Failed due to exception - {str(e)}")

    def test_model_file(self):
        test_obj = TestUtils()
        try:
            # Encode and vectorize the training data
            le_variety = LabelEncoder()
            self.df_train['variety_encoded'] = le_variety.fit_transform(self.df_train['variety'])
            vectorizer = TfidfVectorizer(max_features=1000)
            vectorizer.fit(self.df_train['review'])
            X_train = vectorize_and_add_features(self.df_train, vectorizer)
            Y_train = self.df_train['variety_encoded']

            # Train and save the model
            model = train_model(X_train, Y_train)
            save_model(model, 'wine_model_test.pkl')

            # Check if the model file exists and can be loaded
            if os.path.exists('wine_model_test.pkl'):
                loaded_model = load_model('wine_model_test.pkl')
                if isinstance(loaded_model, RandomForestClassifier):
                    test_obj.yakshaAssert("TestModelSave", True, "functional")
                    print("TestModelSave: Passed")
                else:
                    test_obj.yakshaAssert("TestModelSave", False, "functional")
                    print("TestModelSave: Failed - Model loaded is not a RandomForestClassifier")
            else:
                test_obj.yakshaAssert("TestModelSave", False, "functional")
                print("TestModelSave: Failed - Model file does not exist")
        except Exception as e:
            test_obj.yakshaAssert("TestModelSave", False, "functional")
            print(f"TestModelSave: Failed due to exception - {str(e)}")


if __name__ == '__main__':
    unittest.main()
