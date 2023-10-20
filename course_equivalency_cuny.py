from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os

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

def get_catalog_link(url):
    """Return a formatted link for the catalog or a 'No Catalog Link' text."""
    if url and not pd.isna(url) and url.strip():
        return f"[Catalog]({url})"
    else:
        return "No Catalog Link"


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

    tfidf_vectorizer_description = TfidfVectorizer(stop_words='english', max_features=5000)
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


#

# Streamlit UI
st.title('CUNY Course Matcher')

# Dictionary mapping codes to college names
college_names = {
    "CUNY01": "All CUNY Colleges",
    "BAR01": "Baruch College",
    "BCC01": "Bronx Community College",
    "BKL01": "Brooklyn College",
    "BMC01": "Borough of Manhattan Community College",
    "CSI01": "College of Staten Island",
    "CTY01": "City College",
    "HOS01": "Hostos Community College",
    "HTR01": "Hunter College",
    "JJC01": "John Jay College of Criminal Justice",
    "KCC01": "Kingsborough Community College",
    "LAG01": "LaGuardia Community College",
    "LEH01": "Lehman College",
    "MEC01": "Medgar Evers College",
    "MHC01": "Macaulay Honors College",
    "NCC01": "Guttman Community College",
    "NYT01": "New York City College of Technology",
    "QCC01": "Queensborough Community College",
    "QNS01": "Queens College",
    "SLU01": "School of Labor & Urban Studies",
    "SPS01": "School of Professional Studies",
    "YRK01": "York College"
}

# List all files in the 'cuny colleges' directory
all_files = os.listdir('cuny colleges')

# Convert file names to college names using the dictionary
display_names = [college_names[file.split('.')[0]] if file.split('.')[0] in college_names else file for file in all_files]

selected_colleges = []

# Use a loop to create a checkbox for each college and capture which are selected
for college_code, college_name in college_names.items():
    if college_name in display_names:  # Make sure the college is in display_names
        if st.sidebar.checkbox(college_name, value=False):  # Default value is unchecked
            selected_colleges.append(college_name)

# Initialize or get the state of our input fields
user_input_title = st.session_state.get('input_title', '')
user_input_description = st.session_state.get('input_description', '')

# Text input for course title
user_input_title = st.text_input('Enter the course title:', user_input_title)
# Text input for course description
user_input_description = st.text_area('Enter the course description:', user_input_description)

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

for selected_college in selected_colleges:
    # Convert back from college name to file name when reading the CSV
    selected_file = [key for key, value in college_names.items() if value == selected_college][0] + ".csv"

    # Load the selected CSV from the 'cuny colleges' directory
    df_updated = pd.read_csv(os.path.join('cuny colleges', selected_file))

    df_updated['description'].fillna("", inplace=True)

    if search_button and user_input_title and user_input_description:
        closest_titles, closest_descriptions = find_closest_courses_with_scores_updated(user_input_title,
                                                                                        user_input_description,
                                                                                        df_updated)


        # Helper function to apply styling based on similarity
        def style_text(similarity, text):
            if similarity == "Similar":
                return f"<span style='color: green'>{text}</span>"
            elif similarity == "Very Similar":
                return f"<span style='color: green; font-weight: bold'>{text}</span>"
            else:
                return text


        # Assuming you've already extracted the college name as done before:
        selected_code = selected_file.split('.')[0]  # assuming file has a .csv extension
        college_name = college_names.get(selected_code,
                                         selected_code)  # get the college name from the dictionary or use the code if not found

        # Determine if "All CUNY Colleges" is selected
        all_cuny_selected = "All CUNY Colleges" in selected_colleges

        # Use the college_name in the string to display the message
        st.markdown(
            f"Closest course matches for title at <span style='color:red; font-weight:bold;'>{college_name}</span> are:",
            unsafe_allow_html=True)

        # Adjusting the table header for titles to include "College"
        title_table = "| Title | College | Code | Score | Similarity | College Course Catalog | T-REX |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        for title in closest_titles:
            # If "All CUNY Colleges" is selected, use the actual college name from the data for that specific row
            if all_cuny_selected:
                actual_college_code = df_updated['college'].iloc[
                    title[0]]  # Assuming the column in the file that has the college codes is named 'college'
                display_college_name = college_names.get(actual_college_code, actual_college_code)
            else:
                display_college_name = college_name
            styled_title = style_text(title[2], df_updated['title'].iloc[title[0]])
            title_table += f"| {styled_title} | {display_college_name} | {df_updated['CODE'].iloc[title[0]]} | {title[1]:.4f} | {title[2]} | {get_catalog_link(df_updated['url'].iloc[title[0]])} | {df_updated['t_rex'].iloc[title[0]]} |\n"
        st.markdown(title_table, unsafe_allow_html=True)

        st.write("")  # Add a space

        # Adjusting the table header for descriptions to include "College"
        st.markdown(
            f"Closest course matches for description at <span style='color:red; font-weight:bold;'>{college_name}</span> are:",
            unsafe_allow_html=True)
        description_table = "| Title | College | Code | Score | Similarity | College Course Catalog | T-REX |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        for description in closest_descriptions:
            # Similar adjustment for the description table
            if all_cuny_selected:
                actual_college_code = df_updated['college'].iloc[description[0]]
                display_college_name = college_names.get(actual_college_code, actual_college_code)
            else:
                display_college_name = college_name
            styled_description = style_text(description[2], df_updated['title'].iloc[description[0]])
            description_table += f"| {styled_description} | {display_college_name} | {df_updated['CODE'].iloc[description[0]]} | {description[1]:.4f} | {description[2]} | {get_catalog_link(df_updated['url'].iloc[description[0]])} | {df_updated['t_rex'].iloc[description[0]]} |\n"
        st.markdown(description_table, unsafe_allow_html=True)

        # Adding a space
        st.write("")

        # Adding a horizontal line
        st.markdown("---")






