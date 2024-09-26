import unittest
import pickle
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier
import os
from test.TestUtils import TestUtils  # Assuming this is a custom testing utility class
from WineClassification import (load_data, process_reviews, most_frequent_word, longest_review, average_review_length,
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
        actual_most_common_word = most_frequent_word(self.df_train)
        if actual_most_common_word == self.expected_most_common_word:
            test_obj.yakshaAssert("TestMostCommonWord", True, "functional")
            print("Most frequent word: Passed")
        else:
            test_obj.yakshaAssert("TestMostCommonWord", False, "functional")
            print("Most frequent word: Failed")

    def test_longest_review(self):
        test_obj = TestUtils()
        actual_longest_review = longest_review(self.df_train)
        if actual_longest_review == self.expected_longest_review:
            test_obj.yakshaAssert("TestLongestReview", True, "functional")
            print("Longest review: Passed")
        else:
            test_obj.yakshaAssert("TestLongestReview", False, "functional")
            print("Longest review: Failed")

    def test_average_review_length(self):
        test_obj = TestUtils()
        actual_avg_review_length = round(average_review_length(self.df_train), 2)
        if actual_avg_review_length == self.expected_avg_review_length:
            test_obj.yakshaAssert("TestAverageReviewLength", True, "functional")
            print("Average review length: Passed")
        else:
            test_obj.yakshaAssert("TestAverageReviewLength", False, "functional")
            print("Average review length: Failed")

    def test_shortest_review(self):
        test_obj = TestUtils()
        actual_shortest_review = shortest_review(self.df_train)
        if actual_shortest_review == self.expected_shortest_review:
            test_obj.yakshaAssert("TestShortestReview", True, "functional")
            print("Shortest review: Passed")
        else:
            test_obj.yakshaAssert("TestShortestReview", False, "functional")
            print("Shortest review: Failed")

    def test_most_common_adjective(self):
        test_obj = TestUtils()
        actual_most_common_adjective = most_common_adjective(self.df_train)
        if actual_most_common_adjective == self.expected_most_common_adjective:
            test_obj.yakshaAssert("TestMostCommonAdjective", True, "functional")
            print("Most common adjective: Passed")
        else:
            test_obj.yakshaAssert("TestMostCommonAdjective", False, "functional")
            print("Most common adjective: Failed")

    def test_most_frequent_review_title(self):
        test_obj = TestUtils()
        actual_most_frequent_title = most_frequent_review_title(self.df_train)
        if actual_most_frequent_title == self.expected_most_common_title:
            test_obj.yakshaAssert("TestMostFrequentTitle", True, "functional")
            print("Most frequent review title: Passed")
        else:
            test_obj.yakshaAssert("TestMostFrequentTitle", False, "functional")
            print("Most frequent review title: Failed")

    def test_average_points(self):
        test_obj = TestUtils()
        actual_avg_points = round(average_points(self.df_train), 2)
        if actual_avg_points == self.expected_avg_points:
            test_obj.yakshaAssert("TestAveragePoints", True, "functional")
            print("Average points of wines: Passed")
        else:
            test_obj.yakshaAssert("TestAveragePoints", False, "functional")
            print("Average points of wines: Failed")

    def test_most_frequent_country(self):
        test_obj = TestUtils()
        actual_most_frequent_country = most_frequent_country(self.df_train)
        if actual_most_frequent_country == self.expected_most_common_country:
            test_obj.yakshaAssert("TestMostFrequentCountry", True, "functional")
            print("Most frequent country: Passed")
        else:
            test_obj.yakshaAssert("TestMostFrequentCountry", False, "functional")
            print("Most frequent country: Failed")

    def test_model_file(self):
        test_obj = TestUtils()

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
                print("Model file save and load: Passed")
            else:
                test_obj.yakshaAssert("TestModelSave", False, "functional")
                print("Model file save and load: Failed")
        else:
            test_obj.yakshaAssert("TestModelSave", False, "functional")
            print("Model file save and load: Failed")

if __name__ == '__main__':
    unittest.main()
