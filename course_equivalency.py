from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


# The categorize_similarity function was not provided in the code snippet, so I'm redefining it here for completeness
def categorize_similarity(score):
    """Categorize the cosine similarity score."""
    if score <= 0.2:
        return "Not Similar"
    elif score <= 0.5:
        return "Somewhat Similar"
    elif score <= 0.7:
        return "Similar"
    else:
        return "Very Similar"


# The find_closest_courses_with_scores_updated function will be slightly modified to take
# both title and description as parameters and return the closest matches for both.
def find_closest_courses_with_scores_updated(title, description, csv_file='csi_courses_f23.csv'):
    # Load the CSV
    df_updated = pd.read_csv(csv_file)

    # Process for title
    if 'title' not in df_updated.columns:
        raise ValueError("'title' column not found in the CSV.")

    tfidf_vectorizer_title = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(df_updated['title'])
    title_vectorized = tfidf_vectorizer_title.transform([title])
    cosine_similarities_title = linear_kernel(title_vectorized, tfidf_matrix_title).flatten()
    top_title_indices = cosine_similarities_title.argsort()[-3:][::-1]
    closest_titles = [
        (df_updated['CODE'].iloc[i], cosine_similarities_title[i], categorize_similarity(cosine_similarities_title[i]))
        for i in top_title_indices]

    # Process for description
    if 'description' not in df_updated.columns:
        raise ValueError("'description' column not found in the CSV.")

    tfidf_vectorizer_description = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df_updated['description'])
    description_vectorized = tfidf_vectorizer_description.transform([description])
    cosine_similarities_description = linear_kernel(description_vectorized, tfidf_matrix_description).flatten()
    top_description_indices = cosine_similarities_description.argsort()[-3:][::-1]
    closest_descriptions = [(df_updated['CODE'].iloc[i], cosine_similarities_description[i],
                             categorize_similarity(cosine_similarities_description[i]))
                            for i in top_description_indices]

    return closest_titles, closest_descriptions


# Streamlit UI
st.title('Course Matcher')

# User input for title and description
user_input_title = st.text_input("Please enter a course title:")
user_input_description = st.text_area("Please enter a course description:")

if user_input_title and user_input_description:
    closest_titles, closest_descriptions = find_closest_courses_with_scores_updated(user_input_title,
                                                                                    user_input_description)

    st.write("Closest course matches for title are:")
    for title in closest_titles:
        st.write(f"Code: {title[0]}, Score: {title[1]:.4f}, Similarity: {title[2]}")

    st.write("\nClosest course matches for description are:")
    for description in closest_descriptions:
        st.write(f"Code: {description[0]}, Score: {description[1]:.4f}, Similarity: {description[2]}")
