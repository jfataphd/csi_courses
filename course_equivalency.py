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
def find_closest_courses_with_scores_updated(title, description, df, selected_college):
    # Determine if "All CUNY Colleges" is selected
    all_cuny_selected = "All CUNY Colleges" in selected_college

    # Process for title
    if 'title' not in df.columns:
        raise ValueError("'title' column not found in the CSV.")
    max_features_title = 2000 if all_cuny_selected else 1000

    # Custom stop words list
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['independent', 'honors', 'introductory', 'advanced', 'study', 'introduction', 'elementary',
                     'fundamental', 'seminar', 'I', 'II', 'intermediate', 'special', 'topics', 'i', 'ii', 'studies',
                     'research', 'modern', 'internship', 'service', 'studies', 'workshop', 'directed', 'senior',
                     'seminars', 'comparative', 'studies', 'description', 'topic', 'topics', 'registrar',
                     'adv', 'catalog', 'see', 'for', 'study', 'intro', 'basic',
                     'registrars', 'advanced', 'principles', 'department', 'course', 'students', 'techniques', 'coop',
                     'faculty', 'member', 'project', 'credits', 'identified', 'instructors', 'announced', 'semester',
                     'repeated', 'eva', 'zora', 'dryden', 'creditsacquisition', 'pilot', 'topical', 'immediate', 'qulaifies',
                     'option', 'different', 'provided', 'id', 'qc', 'sp', '3008', '3456', '3610', '17', '11', '114', 'qualifies',
                     'accompanieducation', 'student', 'students', 'theory', 'analysis', 'work', 'lecture', 'college', 'skills',
                     'field', 'lit', 'program', 'required', 'approved', 'soc', 'sem', 'covered', 'determined', 'offering',
                     '&', 'continuing', '(in', '2', '1', 'vt:', 'iii', 'topics:', 'capstone', 'fieldwork', 'internship:',
                     'practicum', 'stdy', 'tech', 'stdies', 'internship', 'englishL)', 'v', 'ital', 'internsh', 'arts,',
                     'iv', 'fundamentals', 'indiv-instr', 'dsg', 'meth', 'psy', 'gen', 'la', 'sem:', 'melville', 'literature:',
                     'lab', 'laboratory', 'select', 'sel', 'amer', 'major', 'subject', 'new', 'hist', 'sci', 'learn', 'msp',
                     'credit', 'tutorial', 'individual', 'description', 'hunter', 'including', 'methods', 'mod', 'permission',
                     'description.', 'literature', 'hours', 'study:', 'various', 'use', 'various', 'maximum', 'issues', 'include',
                     'mat', 'using', 'report', 'majors', 'minors', 'hour', 'art.', 'music.', 'used', 'number', 'art,',
                     'self', 'history.', 'point', 'design,', 'prerequisite:', 'art:', 'technology,', 'writing,', 'bio', 'enlish',]

    tfidf_vectorizer_title = TfidfVectorizer(stop_words=custom_stop_words, max_features=max_features_title)
    tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(df['title'])
    title_vectorized = tfidf_vectorizer_title.transform([title])
    cosine_similarities_title = linear_kernel(title_vectorized, tfidf_matrix_title).flatten()
    top_20_title_indices = cosine_similarities_title.argsort()[-20:][::-1]

    # Process for description
    if 'description' not in df.columns:
        raise ValueError("'description' column not found in the CSV.")
    max_features_description = 10000 if all_cuny_selected else 5000

    tfidf_vectorizer_description = TfidfVectorizer(stop_words=custom_stop_words, max_features=max_features_description)
    tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df['description'])
    description_vectorized = tfidf_vectorizer_description.transform([description])
    cosine_similarities_description = linear_kernel(description_vectorized, tfidf_matrix_description).flatten()

    # Calculating combined scores
    combined_scores = []
    for index in top_20_title_indices:
        title_sim_score = cosine_similarities_title[index]
        desc_sim_score = cosine_similarities_description[index]
        combined_score = 0.75 * title_sim_score + 0.25 * desc_sim_score
        combined_scores.append((index, combined_score))

    # Sorting by combined score and extracting top 20
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_combined = combined_scores[:20]

    # Creating a list of tuples (index, title_similarity, description_similarity, combined_score)
    results = [(i[0], cosine_similarities_title[i[0]], cosine_similarities_description[i[0]], i[1]) for i in
               top_combined]
    return results


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

selected_colleges = []

# Use a loop to create a checkbox for each college and capture which are selected
for college_code, college_name in college_names.items():
    default_value = True if college_name == "College of Staten Island" else False
    if st.sidebar.checkbox(college_name, value=default_value):
        selected_colleges.append(college_name)

# ... [the same as before, no changes]

# Fetching course data from csi_coursesf23.csv
course_data = fetch_course_data_from_cuny01()

# Default initialization for description_label
description_label = 'Course Description:'

# Define the Reset button immediately
# reset_button = st.button('Reset')


# If the Reset button is pressed, initialize values to empty or defaults.
# if reset_button:
#     # Input fields
#     user_input_title = ''
#     user_input_description = ''
#     user_input_code = ''

#     # Checkboxes and select boxes
#     selected_colleges = []
#     for college_code, college_name in college_names.items():
#         default_value = True if college_name == "College of Staten Island" else False
#         if st.sidebar.checkbox(college_name, value=default_value, key=college_code):
#             selected_colleges.append(college_name)

#     # Session state
#     if 'input_title' in st.session_state:
#         del st.session_state['input_title']
#     if 'input_description' in st.session_state:
#         del st.session_state['input_description']

#     # Rerun the app to refresh the page immediately.
#     st.rerun()

input_choice = st.radio("Step 2: Select input method:",
                            ["SUNY Course Title", "SUNY Course Code", "Any Course Title/Description"])

# Default initialization
user_input_title = ""
user_input_description = ""
options_list = []
user_input_code = ""

if input_choice == "SUNY Course Title":
    user_input_title = st.text_input(
        'Step 3: Type part or all of the course title and then hit ENTER to see possible CSI matches:', '')
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

# # Display the course description text area
# description_label = 'Course Description:'
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


for selected_college in selected_colleges:
    # Convert back from college name to file name when reading the CSV
    selected_file = [key for key, value in college_names.items() if value == selected_college][0] + ".csv"

    # Load the selected CSV from the 'cuny colleges' directory
    df_updated = pd.read_csv(os.path.join('cuny colleges_original', selected_file))
    df_updated['description'].fillna("", inplace=True)

    if search_button:
        # Adjust search_input_title based on search_field
        if search_field == 'code':
            search_input_title = matched_title
        elif search_field == 'title':
            # Already set in the previous block
            pass
        else:
            search_input_title = user_input_title

        # Using the updated function
        closest_courses = find_closest_courses_with_scores_updated(search_input_title, user_input_description, df_updated, selected_college)

        # Displaying the results
        st.markdown(f"### Top 20 Similar Courses for {selected_college}:")
        results_table = "| Title | College | Code | Title Similarity | Description Similarity | Combined Score | Catalog Link | T-REX |\n"
        results_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        for course in closest_courses:
            index = course[0]
            title_sim = course[1]
            desc_sim = course[2]
            combined_score = course[3]
            results_table += f"| {df_updated['title'].iloc[index]} | {df_updated['college'].iloc[index]} | {df_updated['CODE'].iloc[index]} | {title_sim:.2f} | {desc_sim:.2f} | {combined_score:.2f} | {get_catalog_link(df_updated['url'].iloc[index])} | {df_updated['t_rex'].iloc[index]} |\n"
        st.markdown(results_table, unsafe_allow_html=True)

        # Adding a space
        st.write("")

        # Adding a horizontal line
        st.markdown("---")
        # Adding a horizontal line
        st.markdown("---")

