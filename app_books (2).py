import streamlit as st
import pandas as pd
import numpy as np
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODELS
# =========================================================

@st.cache_resource
def load_models():

    with open(
        "recommendation_models.pkl",
        "rb"
    ) as file:

        model_bundle = pickle.load(
            file
        )

    with open(
        "books.pkl",
        "rb"
    ) as file:

        books = pickle.load(
            file
        )

    return model_bundle, books


try:

    model_bundle, books = load_models()

except FileNotFoundError:

    st.error(
        """
        Model files not found.

        Make sure these files are in the same
        folder as app.py:

        recommendation_models.pkl
        books.pkl
        """
    )

    st.stop()


# =========================================================
# EXTRACT SAVED OBJECTS
# =========================================================

svd_model = model_bundle[
    "svd_model"
]

user_factors_svd = model_bundle[
    "user_factors_svd"
]

book_factors_svd = model_bundle[
    "book_factors_svd"
]

nmf_model = model_bundle[
    "nmf_model"
]

user_factors_nmf = model_bundle[
    "user_factors_nmf"
]

book_factors_nmf = model_bundle[
    "book_factors_nmf"
]

user_to_index = model_bundle[
    "user_to_index"
]

book_to_index = model_bundle[
    "book_to_index"
]

index_to_user = model_bundle[
    "index_to_user"
]

index_to_book = model_bundle[
    "index_to_book"
]


# =========================================================
# CLEAN BOOK DETAILS
# =========================================================

books = books.drop_duplicates(
    subset="ISBN"
).copy()

books["ISBN"] = (
    books["ISBN"]
    .astype(str)
)

book_details = books.set_index(
    "ISBN"
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_book_details(isbn):

    if isbn not in book_details.index:

        return {
            "title": "Unknown",
            "author": "Unknown",
            "publisher": "Unknown",
            "year": "Unknown",
            "image": None
        }

    row = book_details.loc[
        isbn
    ]

    return {
        "title":
            row.get(
                "Book-Title",
                "Unknown"
            ),

        "author":
            row.get(
                "Book-Author",
                "Unknown"
            ),

        "publisher":
            row.get(
                "Publisher",
                "Unknown"
            ),

        "year":
            row.get(
                "Year-Of-Publication",
                "Unknown"
            ),

        "image":
            row.get(
                "Image-URL-M",
                None
            )
    }


# =========================================================
# SVD RECOMMENDATION FUNCTION
# =========================================================

def recommend_svd(
    user_id,
    n=10
):

    if user_id not in user_to_index:

        return pd.DataFrame()

    user_index = user_to_index[
        user_id
    ]

    # Predict scores for all books
    scores = np.dot(
        book_factors_svd,
        user_factors_svd[
            user_index
        ]
    )

    scores = np.clip(
        scores,
        1,
        10
    )

    # Sort highest scores first
    sorted_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for book_index in sorted_indices:

        isbn = index_to_book[
            book_index
        ]

        details = get_book_details(
            isbn
        )

        recommendations.append({
            "ISBN":
                isbn,

            "Book-Title":
                details[
                    "title"
                ],

            "Book-Author":
                details[
                    "author"
                ],

            "Publisher":
                details[
                    "publisher"
                ],

            "Year":
                details[
                    "year"
                ],

            "Image":
                details[
                    "image"
                ],

            "Predicted-Rating":
                round(
                    float(
                        scores[
                            book_index
                        ]
                    ),
                    3
                )
        })

        if len(
            recommendations
        ) >= n:

            break

    return pd.DataFrame(
        recommendations
    )


# =========================================================
# NMF RECOMMENDATION FUNCTION
# =========================================================

def recommend_nmf(
    user_id,
    n=10
):

    if user_id not in user_to_index:

        return pd.DataFrame()

    user_index = user_to_index[
        user_id
    ]

    scores = np.dot(
        book_factors_nmf,
        user_factors_nmf[
            user_index
        ]
    )

    scores = np.clip(
        scores,
        1,
        10
    )

    sorted_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for book_index in sorted_indices:

        isbn = index_to_book[
            book_index
        ]

        details = get_book_details(
            isbn
        )

        recommendations.append({
            "ISBN":
                isbn,

            "Book-Title":
                details[
                    "title"
                ],

            "Book-Author":
                details[
                    "author"
                ],

            "Publisher":
                details[
                    "publisher"
                ],

            "Year":
                details[
                    "year"
                ],

            "Image":
                details[
                    "image"
                ],

            "Predicted-Rating":
                round(
                    float(
                        scores[
                            book_index
                        ]
                    ),
                    3
                )
        })

        if len(
            recommendations
        ) >= n:

            break

    return pd.DataFrame(
        recommendations
    )


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================

def display_recommendations(
    recommendations
):

    if recommendations.empty:

        st.warning(
            "No recommendations found."
        )

        return

    books_per_row = 5

    for start in range(
        0,
        len(recommendations),
        books_per_row
    ):

        row_data = recommendations.iloc[
            start:
            start + books_per_row
        ]

        columns = st.columns(
            len(row_data)
        )

        for column, (_, book) in zip(
            columns,
            row_data.iterrows()
        ):

            with column:

                image = book[
                    "Image"
                ]

                if (
                    pd.notna(image)
                    and str(
                        image
                    ).startswith(
                        "http"
                    )
                ):

                    try:

                        st.image(
                            image,
                            use_container_width=True
                        )

                    except Exception:

                        st.write(
                            "📕"
                        )

                else:

                    st.write(
                        "📕"
                    )

                st.markdown(
                    f"### {book['Book-Title']}"
                )

                st.write(
                    f"**Author:** "
                    f"{book['Book-Author']}"
                )

                st.write(
                    f"**Publisher:** "
                    f"{book['Publisher']}"
                )

                st.write(
                    f"**Year:** "
                    f"{book['Year']}"
                )

                st.write(
                    f"⭐ Predicted Rating: "
                    f"{book['Predicted-Rating']}"
                )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📚 Book Recommender"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🤖 SVD Recommendations",
        "🔢 NMF Recommendations",
        "📊 Model Information",
        "📚 Books Dataset"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "📚 Book Recommendation System"
    )

    st.write(
        """
        This Machine Learning project recommends
        books based on user-rating patterns.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Users in Model",
        f"{len(user_to_index):,}"
    )

    col2.metric(
        "Books in Model",
        f"{len(book_to_index):,}"
    )

    col3.metric(
        "Book Records",
        f"{len(books):,}"
    )

    st.divider()

    st.subheader(
        "Models Used"
    )

    st.write(
        """
        ### Truncated SVD

        Matrix factorization model used to
        learn hidden relationships between
        users and books.

        ### NMF

        Non-negative Matrix Factorization used
        as another recommendation model.

        Both models generate personalized
        recommendations for users.
        """
    )

    st.subheader(
        "Technology"
    )

    st.write(
        """
        Python • Pandas • NumPy •
        Scikit-learn • SciPy • Streamlit
        """
    )


# =========================================================
# SVD PAGE
# =========================================================

elif page == "🤖 SVD Recommendations":

    st.title(
        "🤖 SVD Book Recommendations"
    )

    st.write(
        """
        Select a User ID to generate
        personalized recommendations using
        the Truncated SVD model.
        """
    )

    available_users = sorted(
        list(
            user_to_index.keys()
        )
    )

    selected_user = st.selectbox(
        "Select User ID",
        available_users
    )

    number = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        key="svd_slider"
    )

    if st.button(
        "Recommend with SVD",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating recommendations..."
        ):

            recommendations = recommend_svd(
                selected_user,
                number
            )

        st.success(
            f"Recommendations for User "
            f"{selected_user}"
        )

        display_recommendations(
            recommendations
        )


# =========================================================
# NMF PAGE
# =========================================================

elif page == "🔢 NMF Recommendations":

    st.title(
        "🔢 NMF Book Recommendations"
    )

    st.write(
        """
        Select a User ID to generate
        recommendations using the
        Non-negative Matrix Factorization model.
        """
    )

    available_users = sorted(
        list(
            user_to_index.keys()
        )
    )

    selected_user = st.selectbox(
        "Select User ID",
        available_users,
        key="nmf_user"
    )

    number = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        key="nmf_slider"
    )

    if st.button(
        "Recommend with NMF",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating recommendations..."
        ):

            recommendations = recommend_nmf(
                selected_user,
                number
            )

        st.success(
            f"NMF Recommendations for User "
            f"{selected_user}"
        )

        display_recommendations(
            recommendations
        )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif page == "📊 Model Information":

    st.title(
        "📊 Model Information"
    )

    col1, col2 = st.columns(
        2
    )

    col1.metric(
        "Total Users",
        len(
            user_to_index
        )
    )

    col2.metric(
        "Total Books",
        len(
            book_to_index
        )
    )

    st.divider()

    st.subheader(
        "SVD Model"
    )

    st.write(
        "User Factor Matrix:",
        user_factors_svd.shape
    )

    st.write(
        "Book Factor Matrix:",
        book_factors_svd.shape
    )

    st.subheader(
        "NMF Model"
    )

    st.write(
        "User Factor Matrix:",
        user_factors_nmf.shape
    )

    st.write(
        "Book Factor Matrix:",
        book_factors_nmf.shape
    )

    st.divider()

    model_data = pd.DataFrame({
        "Model": [
            "Truncated SVD",
            "NMF"
        ],

        "Purpose": [
            "Latent-factor recommendation",
            "Non-negative matrix factorization"
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True
    )


# =========================================================
# BOOK DATASET PAGE
# =========================================================

elif page == "📚 Books Dataset":

    st.title(
        "📚 Books Dataset"
    )

    st.write(
        "Total Books:",
        len(
            books
        )
    )

    search = st.text_input(
        "Search Book Title"
    )

    if search:

        filtered_books = books[
            books[
                "Book-Title"
            ]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered_books = (
            books.head(
                100
            )
        )

    display_columns = [
        column
        for column in [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Year-Of-Publication",
            "Publisher"
        ]
        if column in books.columns
    ]

    st.dataframe(
        filtered_books[
            display_columns
        ],
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
    ">
        <b>📚 Book Recommendation System</b>
        <br>
        Machine Learning • SVD • NMF • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)