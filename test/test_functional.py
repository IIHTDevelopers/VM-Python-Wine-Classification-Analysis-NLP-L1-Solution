import unittest
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from test.TestUtils import TestUtils  # Assuming this is a custom testing utility class
from winenlptemplate import (load_data, process_reviews, most_frequent_word, longest_review, average_review_length,
                         shortest_review, most_common_adjective, most_frequent_review_title,
                         most_frequent_country, train_model, save_model, load_model, vectorize_and_add_features,
                         TfidfVectorizer, LabelEncoder, most_liked_variety_by_points)


class NLPFunctionalTest(unittest.TestCase):

    def setUp(self):
        # Load the training dataset
        self.df_train = load_data('train500likes.csv')
        self.df_test = load_data('test500.csv')  # Assuming this is your test data file

        # Process the dataset for the review-related functions
        self.df_train = process_reviews(self.df_train)
        self.df_test = process_reviews(self.df_test)

        # Expected values (static)
        self.expected_most_common_word = "bad"
        self.expected_longest_review = "amazing taste rich full flavor"
        self.expected_avg_review_length = 22.55
        self.expected_shortest_review = "bad bit sour"
        self.expected_most_common_adjective = "bad"
        self.expected_most_common_title = "Not bad"
        self.expected_most_common_country = "Spain"
        self.expected_most_liked_variety = "Syrah"  # Expected most liked variety
        self.expected_avg_points = 92.21  # Expected average points for the most liked variety

    def test_most_frequent_word(self):
        test_obj = TestUtils()
        try:
            actual_most_common_word = most_frequent_word(self.df_train)
            if actual_most_common_word == self.expected_most_common_word:
                test_obj.yakshaAssert("TestMostCommonWord", True, "functional")
                print("TestMostCommonWord: Passed")
            else:
                test_obj.yakshaAssert("TestMostCommonWord", False, "functional")
                print(f"TestMostCommonWord: Failed ")
        except Exception as e:
            test_obj.yakshaAssert("TestMostCommonWord", False, "functional")
            print(f"TestMostCommonWord: Failed ")

    def test_longest_review(self):
        test_obj = TestUtils()
        try:
            actual_longest_review = longest_review(self.df_train)
            if actual_longest_review == self.expected_longest_review:
                test_obj.yakshaAssert("TestLongestReview", True, "functional")
                print("TestLongestReview: Passed")
            else:
                test_obj.yakshaAssert("TestLongestReview", False, "functional")
                print(f"TestLongestReview: Failed ")
        except Exception as e:
            test_obj.yakshaAssert("TestLongestReview", False, "functional")
            print(f"TestLongestReview: Failed ")

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
                    print(f"TestAverageReviewLength: Failed ")
            else:
                raise ValueError("Function returned None")
        except Exception as e:
            test_obj.yakshaAssert("TestAverageReviewLength", False, "functional")
            print(f"TestAverageReviewLength: Failed ")

    def test_shortest_review(self):
        test_obj = TestUtils()
        try:
            actual_shortest_review = shortest_review(self.df_train)
            if actual_shortest_review == self.expected_shortest_review:
                test_obj.yakshaAssert("TestShortestReview", True, "functional")
                print("TestShortestReview: Passed")
            else:
                test_obj.yakshaAssert("TestShortestReview", False, "functional")
                print(f"TestShortestReview: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestShortestReview", False, "functional")
            print(f"TestShortestReview: Failed")

    def test_most_common_adjective(self):
        test_obj = TestUtils()
        try:
            actual_most_common_adjective = most_common_adjective(self.df_train)
            if actual_most_common_adjective == self.expected_most_common_adjective:
                test_obj.yakshaAssert("TestMostCommonAdjective", True, "functional")
                print("TestMostCommonAdjective: Passed")
            else:
                test_obj.yakshaAssert("TestMostCommonAdjective", False, "functional")
                print(f"TestMostCommonAdjective: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestMostCommonAdjective", False, "functional")
            print(f"TestMostCommonAdjective :: Failed")

    def test_most_frequent_review_title(self):
        test_obj = TestUtils()
        try:
            actual_most_frequent_title = most_frequent_review_title(self.df_train)
            if actual_most_frequent_title == self.expected_most_common_title:
                test_obj.yakshaAssert("TestMostFrequentTitle", True, "functional")
                print("TestMostFrequentTitle: Passed")
            else:
                test_obj.yakshaAssert("TestMostFrequentTitle", False, "functional")
                print(f"TestMostFrequentTitle: Failed ")
        except Exception as e:
            test_obj.yakshaAssert("TestMostFrequentTitle", False, "functional")
            print(f"TestMostFrequentTitle: Failed ")

    def test_most_frequent_country(self):
        test_obj = TestUtils()
        try:
            actual_most_frequent_country = most_frequent_country(self.df_train)
            if actual_most_frequent_country == self.expected_most_common_country:
                test_obj.yakshaAssert("TestMostFrequentCountry", True, "functional")
                print("TestMostFrequentCountry: Passed")
            else:
                test_obj.yakshaAssert("TestMostFrequentCountry", False, "functional")
                print(f"TestMostFrequentCountry: Failed ")
        except Exception as e:
            test_obj.yakshaAssert("TestMostFrequentCountry", False, "functional")
            print(f"TestMostFrequentCountry: Failed ")

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

            # Train and save the model with a random seed for consistency
            model = RandomForestClassifier(n_estimators=250, random_state=42)
            model.fit(X_train, Y_train)
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
            print(f"TestModelSave: Failed")

    # New test case for the most liked variety based on average points
    def test_most_liked_variety_by_points(self):
        test_obj = TestUtils()
        try:
            # Check if the 'variety' column exists in df_test before encoding
            if 'variety' not in self.df_test.columns:
                raise ValueError("'variety' column not found in the test dataset")

            # Encode the 'variety' column into 'variety_encoded'
            le_variety = LabelEncoder()
            self.df_test['variety_encoded'] = le_variety.fit_transform(self.df_test['variety'])

            # Check if 'variety_encoded' was created successfully
            if 'variety_encoded' not in self.df_test.columns:
                raise ValueError("'variety_encoded' column not created in the test dataset")

            # Vectorize the reviews in the test set using the trained vectorizer
            vectorizer = TfidfVectorizer(max_features=1000)
            vectorizer.fit(self.df_train['review'])  # Fit on training data
            X_test = vectorize_and_add_features(self.df_test, vectorizer)

            # Train the model using the training data with a random seed
            X_train = vectorize_and_add_features(self.df_train, vectorizer)
            Y_train = self.df_train['variety_encoded']
            model = RandomForestClassifier(n_estimators=250, random_state=42)
            model.fit(X_train, Y_train)

            # Predict the most liked variety based on the average points
            actual_most_liked_variety = most_liked_variety_by_points(self.df_test, model, vectorizer, le_variety)

            # Validate the result
            if actual_most_liked_variety == self.expected_most_liked_variety:
                test_obj.yakshaAssert("TestMostLikedVarietyByPoints", True, "functional")
                print("TestMostLikedVarietyByPoints: Passed")
            else:
                test_obj.yakshaAssert("TestMostLikedVarietyByPoints", False, "functional")
                print(
                    f"TestMostLikedVarietyByPoints: Failed ")

        except Exception as e:
            test_obj.yakshaAssert("TestMostLikedVarietyByPoints", False, "functional")
            print(f"TestMostLikedVarietyByPoints: Failed ")


if __name__ == '__main__':
    unittest.main()
