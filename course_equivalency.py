from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

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

def find_closest_courses_with_scores_updated(title, description, df):
    # Process for title
    if 'title' not in df.columns:
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

    # Return the indices for the top 3 courses along with scores and categories for title
    closest_titles = [(i, cosine_similarities_title[i], categorize_similarity(cosine_similarities_title[i]))
                      for i in top_title_indices]

    # Return the indices for the top 3 courses along with scores and categories for description
    closest_descriptions = [
        (i, cosine_similarities_description[i], categorize_similarity(cosine_similarities_description[i]))
        for i in top_description_indices]

    return closest_titles, closest_descriptions


# Load the CSV outside the function
df_updated = pd.read_csv('csi_courses_f23.csv')

# Streamlit UI
st.title('Course Matcher')

# Initialize or get the state of our input fields
user_input_title = st.session_state.get('input_title', '')
user_input_description = st.session_state.get('input_description', '')

# User input for title and description
user_input_title = st.text_input("Please enter a course title:", value=user_input_title)
user_input_description = st.text_area("Please enter a course description:", value=user_input_description)

# Search button
search_button = st.button('Search')

# Reset button (formerly Clear Inputs)
reset_button = st.button('Reset')

# Handle reset button
if reset_button:
    user_input_title = ''
    user_input_description = ''
    st.session_state.input_title = ''
    st.session_state.input_description = ''
    st.experimental_rerun()

# Update session state values for inputs
st.session_state.input_title = user_input_title
st.session_state.input_description = user_input_description

if search_button and user_input_title and user_input_description:
    closest_titles, closest_descriptions = find_closest_courses_with_scores_updated(user_input_title, user_input_description, df_updated)

    # Display title matches in table format
    st.write("Closest course matches for title are:")
    title_data = [{"Title": df_updated['title'].iloc[title[0]],
                   "Code": df_updated['CODE'].iloc[title[0]],
                   "Score": f"{title[1]:.4f}",
                   "Similarity": title[2]} for title in closest_titles]
    st.table(pd.DataFrame(title_data))

    st.write("")  # Add a space

    # Display description matches in table format
    st.write("Closest course matches for description are:")
    description_data = [{"Title": df_updated['title'].iloc[description[0]],
                         "Code": df_updated['CODE'].iloc[description[0]],
                         "Score": f"{description[1]:.4f}",
                         "Similarity": description[2]} for description in closest_descriptions]
    st.table(pd.DataFrame(description_data))
