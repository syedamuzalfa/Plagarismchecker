import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
nltk.download('averaged_perceptron_tagger')


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_words]

    # Return preprocessed text
    return " ".join(lemmatized_words)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)

def calculate_similarity(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Compute Jaccard similarity
    words_text1 = set(preprocessed_text1.split())
    words_text2 = set(preprocessed_text2.split())
    intersection = words_text1.intersection(words_text2)
    union = words_text1.union(words_text2)
    jaccard_similarity = len(intersection) / len(union)

    return jaccard_similarity

def main():
    text1 = "This is a sample text to test plagiarism detection."
    text2 = "This is a test to check for plagiarism."

    similarity = calculate_similarity(text1, text2)
    print("Similarity:", similarity)

    threshold = 0.3  # Set your threshold for plagiarism detection
    if similarity >= threshold:
        print("Plagiarism detected!")
    else:
        print("No plagiarism detected.")

if __name__ == "__main__":
    main()
