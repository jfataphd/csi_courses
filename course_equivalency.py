from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
from difflib import get_close_matches
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS



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

# Helper function to apply styling based on similarity
def style_text(similarity, text):
    if similarity == "Similar":
        return f"<span style='color: green'>{text}</span>"
    elif similarity == "Very Similar":
        return f"<span style='color: green; font-weight: bold'>{text}</span>"
    else:
        return text

def get_catalog_link(url):
    """Return a formatted link for the catalog or a 'No Catalog Link' text."""
    if url is not None and not pd.isna(url) and isinstance(url, str) and url.strip():
        return f"[Catalog]({url})"
    else:
        return "No Catalog Link"

@st.cache_data
def fetch_course_data_from_cuny01():
    """Fetch course data from the CUNY01 CSV."""
    df_cuny01 = pd.read_csv(os.path.join('suny_courses.csv'))
    course_data = {}

    for _, row in df_cuny01.iterrows():
        title = row['title'].strip()
        course_detail = {
            "code": row['CODE'],
            "college": row['college'],
            "description": row['description']
        }

        if title not in course_data:
            course_data[title] = []
        course_data[title].append(course_detail)

    return course_data

@st.cache_data
def find_closest_courses_with_scores_updated(title, description, df, max_features_title=1000, max_features_description=5000):
    # Determine if "College of Staten Island" is selected
    all_cuny_selected = "College of Staten Island" in selected_college
    # Process for title
    if 'title' not in df.columns:
        raise ValueError("'title' column not found in the CSV.")

    max_features_title = 2000 if all_cuny_selected else 1000

    # Custom stop words list
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['independent', 'honors', 'introductory', 'advanced', 'study', 'introduction', 'elementary',
                         'fundamental', 'seminar', 'I', 'II', 'intermediate', 'special', 'topics', 'i', 'ii', 'studies',
                         'research', 'modern', 'internship', 'service', 'studies', 'workshop', 'directed', 'senior', 'seminars',
                         'comparative', 'co-requisite', 'prerequisite', 'requisite', 'mth123', 'mth125', 'general', 'hours', 'credits']

    tfidf_vectorizer_title = TfidfVectorizer(stop_words=custom_stop_words, max_features=max_features_title)
    tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(df['title'])  # Changed df_updated to df
    title_vectorized = tfidf_vectorizer_title.transform([title])
    cosine_similarities_title = linear_kernel(title_vectorized, tfidf_matrix_title).flatten()
    top_title_indices = cosine_similarities_title.argsort()[-11:][::-1] if all_cuny_selected else cosine_similarities_title.argsort()[-4:][::-1]
    # closest_titles = [
    #     (df['CODE'].iloc[i], cosine_similarities_title[i], categorize_similarity(cosine_similarities_title[i]))
    #     for i in top_title_indices]

    # Process for description
    if 'description' not in df.columns:  # Changed df_updated to df
        raise ValueError("'description' column not found in the CSV.")

    max_features_description = 10000 if all_cuny_selected else 5000  # Adjust max_features based on selection

    tfidf_vectorizer_description = TfidfVectorizer(stop_words=custom_stop_words, max_features=max_features_description)
    tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df['description'])  # Changed df_updated to df
    description_vectorized = tfidf_vectorizer_description.transform([description])
    cosine_similarities_description = linear_kernel(description_vectorized, tfidf_matrix_description).flatten()
    top_description_indices = cosine_similarities_description.argsort()[-11:][::-1] if all_cuny_selected else cosine_similarities_description.argsort()[-4:][::-1]
    # closest_descriptions = [(df['CODE'].iloc[i], cosine_similarities_description[i],
    #                          categorize_similarity(cosine_similarities_description[i]))
    #                         for i in top_description_indices]

    # Return the indices for the top 3 courses along with scores and categories for title
    closest_titles = [(i, cosine_similarities_title[i], categorize_similarity(cosine_similarities_title[i]))
                      for i in top_title_indices]

    # Return the indices for the top 3 courses along with scores and categories for description
    closest_descriptions = [
        (i, cosine_similarities_description[i], categorize_similarity(cosine_similarities_description[i]))
        for i in top_description_indices]

    return closest_titles, closest_descriptions


def format_course_code(input_code):
    # Capitalize all letters in the string
    input_code = input_code.upper()

    # Insert a space between the last letter and the first number, if it doesn't exist
    formatted_code = re.sub(r'([A-Z])(\d)', r'\1 \2', input_code)

    return formatted_code

#

# Streamlit UI
st.title('CSI Course Selector and Matcher')

st.sidebar.markdown("## Step 1: Please Select College(s) for Similarity Screening")


# Dictionary mapping codes to college names
college_names = {
    "CSI01": "College of Staten Island"
}

# List all files in the 'cuny colleges' directory
all_files = os.listdir('cuny colleges')

# Convert file names to college names using the dictionary
display_names = [college_names[file.split('.')[0]] if file.split('.')[0] in college_names else file for file in all_files]

selected_colleges = []

# Reduce loop overheads for creating checkboxes
display_name_set = set(display_names)  # Convert to set for O(1) lookup

# Use a loop to create a checkbox for each college and capture which are selected
for college_code, college_name in college_names.items():
    if college_name in display_name_set:
        default_value = True if college_name == "College of Staten Island" else False
        if st.sidebar.checkbox(college_name, value=default_value):
            selected_colleges.append(college_name)



# ... [the same as before, no changes]

# Fetching course data from csi_coursesf23.csv
course_data = fetch_course_data_from_cuny01()
input_choice = st.radio("Step 2: Select input method:", ["SUNY Course Title", "SUNY Course Code", "Any Course Title/Description"])

# Default initialization
user_input_title = ""
user_input_description = ""
options_list = []
user_input_code = ""

if input_choice == "SUNY Course Title":
    user_input_title = st.text_input('Step 3: Type part or all of the course title and then hit ENTER to see possible CSI matches:', '')
    search_key = user_input_title
    search_field = 'title'
elif input_choice == "SUNY Course Code":
    user_input_code = st.text_input('Step 3: Enter the course code to see possible CSI matches:', '')
    search_key = format_course_code(user_input_code)
    search_field = 'code'
    # Here you already determine the corresponding_title based on the course code
    for title, courses in course_data.items():
        for course in courses:
            if course['code'] == search_key:
                corresponding_title = title
                break
else:  # Non-CSI Course Title
    user_input_title = st.text_input('Step 3: Enter the any course title/description:', '')
    search_key = None
    search_field = 'none'

matches = []
# Initialize matched_title
matched_title = ""
options_list = []

if search_field == 'title':
    matches = get_close_matches(search_key, course_data.keys(), n=21)
    for match in matches:
        for course_detail in course_data[match]:
            description = f"{course_detail['code']} - {course_detail['college']} - {match}"
            options_list.append(description)

    # Sort options_list by college
    options_list.sort(key=lambda x: x.split(" - ")[1])

    selected_description = st.selectbox(
        'Select a SUNY course below (repeat if necessary) to populate the Course Description box:',
        options_list,
        key="selectbox_title"
    )

    for description in options_list:
        if selected_description == description:
            matched_title = description.rsplit(" - ", 1)[-1]
            for course in course_data[matched_title]:
                if course['code'] in description:
                    user_input_description = course['description']
                    break

    # Capturing the title from the selected description for use in similarity comparison.
    search_input_title = matched_title

elif search_field == 'code':
    options_list = [f"{course['code']} - {course['college']} - {title}"
                    for title, courses in course_data.items()
                    for course in courses if course['code'] == search_key]

    # Sort options_list by college
    options_list.sort(key=lambda x: x.split(" - ")[1])

    selected_description = st.selectbox(
        'Select a SUNY course below (repeat if necessary) to populate the Course Description box:',
        options_list,
        key="selectbox_code"
    )

    for description in options_list:
        if selected_description == description:
            matched_title = description.rsplit(" - ", 1)[-1]
            for course in course_data[matched_title]:
                if course['code'] in description:
                    user_input_description = course['description']
                    break

# Populate dropdown or use user-input description
elif search_field != 'none':
    selected_description = st.selectbox(
        'Select a SUNY course below (repeat if necessary) to populate the Course Description box:',
        options_list,
        key="selectbox_general"
    )

    for description in options_list:
        if selected_description == description:
            matched_title = description.rsplit(" - ", 1)[-1]
            for course in course_data[matched_title]:
                if course['code'] in description:
                    user_input_description = course['description']
                    break

# Display the course description text area
description_label = 'Course Description:'
if input_choice == "Any Course Title/Description":
    description_label = 'Step 3b: Enter the any course description:'

user_input_description = st.text_area(
    description_label,
    value=user_input_description,
    max_chars=None,
    height=200,
    key="course_description"
)

# Search button
search_button = st.button('Step 4: Search for similarity between course above and CSI courses')

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

    if search_button:
        # Instead, we adjust it to:
        if search_field == 'code':
            search_input_title = corresponding_title
        elif search_field == 'title':
            # This is already set in the previous block, so we don't need to change it
            pass
        else:
            search_input_title = user_input_title

        # Now use search_input_title for similarity scores
        closest_titles, closest_descriptions = find_closest_courses_with_scores_updated(search_input_title, user_input_description, df_updated)

        # Filter closest_titles and closest_descriptions to only include "Similar" and "Very Similar" results
        closest_titles = [title for title in closest_titles if title[2] in ["Similar", "Very Similar"]]
        closest_descriptions = [desc for desc in closest_descriptions if desc[2] in ["Similar", "Very Similar"]]

        # For Descriptions
        st.markdown(
            f"Closest course matches for description at <span style='color:red; font-weight:bold;'>{selected_college}</span> are:",
            unsafe_allow_html=True)

        if not closest_descriptions:
            st.markdown("<b>NO SIMILAR DESCRIPTIONS FOUND</b>", unsafe_allow_html=True)
        else:
            description_table = "| Title | College | Code | Description Similarity | College Course Catalog | T-REX |\n| --- | --- | --- | --- | --- | --- |\n"
            for description in closest_descriptions:
                actual_college_code = df_updated['college'].iloc[description[0]]
                display_college_name = college_names.get(actual_college_code, actual_college_code)
                styled_description = style_text(description[2], df_updated['title'].iloc[description[0]])
                description_table += f"| {styled_description} | {display_college_name} | {df_updated['CODE'].iloc[description[0]]} | {description[2]} | {get_catalog_link(df_updated['url'].iloc[description[0]])} | {df_updated['t_rex'].iloc[description[0]]} |\n"
            st.markdown(description_table, unsafe_allow_html=True)

        st.write("")  # Add a space

        # For Titles
        st.markdown(
            f"Closest course matches for title at <span style='color:red; font-weight:bold;'>{selected_college}</span> are:",
            unsafe_allow_html=True)

        if not closest_titles:
            st.markdown("<b>NO SIMILAR TITLES FOUND</b>", unsafe_allow_html=True)
        else:
            title_table = "| Title | College | Code | Title Similarity | College Course Catalog | T-REX |\n| --- | --- | --- | --- | --- | --- |\n"
            for title in closest_titles:
                actual_college_code = df_updated['college'].iloc[title[0]]
                display_college_name = college_names.get(actual_college_code, actual_college_code)
                styled_title = style_text(title[2], df_updated['title'].iloc[title[0]])
                title_table += f"| {styled_title} | {display_college_name} | {df_updated['CODE'].iloc[title[0]]} | {title[2]} | {get_catalog_link(df_updated['url'].iloc[title[0]])} | {df_updated['t_rex'].iloc[title[0]]} |\n"
            st.markdown(title_table, unsafe_allow_html=True)

        # Adding a space
        st.write("")

        # Adding a horizontal line
        st.markdown("---")
        # Adding a horizontal line
        st.markdown("---")

