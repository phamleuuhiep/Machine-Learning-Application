import streamlit as st
import mysql.connector
import pymysql
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sweetviz as sv
import csv
import json
import openpyxl
import os
import time
import base64
from io import StringIO, BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from ydata_profiling import ProfileReport
from streamlit_lottie import st_lottie

#### ------ HELPER FUNCTION SECTION ------- ####

### --- DEFINE GLOBAL VARIABLES ---
# -- Define a base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -- Session variable to determine if user is logged in or NOT
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False

### --- FUNCTION FOR RENDER ANIMATION ---  
# --- Function to load Lottie animations from online url
def load_lottie_url(url):
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# --- Function for loading lottie picture from json file in directory
def load_lottie_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)

# Load some Lottie animations for better design
lottie_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")  
lottie_animation_sidebar = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_zw0djhar.json")  # Sidebar animation
lottie_animation_intro = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_zw0djhar.json")  # Footer animation
lottie_business = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_data_hello = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
lottie_animation_main_1 = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")
success_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_s2lryxtd.json")
file_upload_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_j1adxtyb.json")

### --- FUNCTION FOR CREATE A SIDEBAR FOR EVERY PAGE ---
def create_side_bar():
        # --- Sidebar with Buttons ---
    st.sidebar.markdown("## Sales Data App")  # Add a bold header for the sidebar
    Logo_path = os.path.join(BASE_DIR, "images", "Logo.png")
    st.sidebar.image(
        Logo_path,  # Replace with your image URL or path
        use_container_width=True,
    )
    st.sidebar.markdown(
    """
    Welcome to the **Sales Data Analysis App**! 
    """
    )
    st.sidebar.markdown("---")  

    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("📖 Introduction", use_container_width=True):
        set_page("Introduction")
    if st.sidebar.button("📂 File Upload", use_container_width=True):
        set_page("File Upload")
    if st.sidebar.button("📊 Data Analysis", use_container_width=True):
        set_page("Data Analysis")

    st.sidebar.markdown("---")  
    st.sidebar.markdown("### About This App")
    st.sidebar.write(
        "This application provides powerful tools for analyzing and visualizing sales data. "
        "Navigate using the buttons above to explore its features."
    )

    # Add an image or logo to the sidebar
    Home_image_path = os.path.join(BASE_DIR, "images", "Home_image.png")
    st.sidebar.image(
        Home_image_path,  
        use_container_width=True,
    )

    # Contact info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Contact")
    st.sidebar.write("📧 Email: support@salesanalysisapp.com")
    st.sidebar.write("📞 Phone: +84 839569959")

### --- FUNCTION FOR LOGIN AND REGISTER ACCOUNT---

# -- Get connection from Database, 
# -- DB info are securely place in the secrets.toml
# @st.cache_resource
# def init_connection():
#     try:
#         return mysql.connector.connect(**st.secrets["mysql"])
#     except mysql.connector.Error as err:
#         st.error(f"Failed to connect to database: {err}")
#         return None
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = "252325"
DB_NAME = "daktdl"


def init_connection():
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT)
        print("✅ Database connection successful! ✅")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e} ❌")
        return None


# Function to check user credentials
def authenticate_user(username, password):
    cursor = None
    conn = None
    try:
        # Connect to the database
        conn = init_connection()
        cursor = conn.cursor()

        # SQL query
        query = "SELECT * FROM useraccount WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        
        # Fetch records
        user = cursor.fetchone()
        if (user):
            conn.close()
            st.cache_resource.clear() 
            return user
        else: return user #reurns the user record if found, otherwise None

    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None
    finally:
        if cursor:
            cursor.close()
            
# -- Function to check the username
def check_username_exist(username, gmail):
    cursor = None
    try:
        conn = init_connection()
        cursor = conn.cursor()

        # SQL query to check if the user exists
        query1 = "SELECT * FROM useraccount WHERE username = %s"
        cursor.execute(query1, (username,))
        # Fetch one record
        user = cursor.fetchone()
        cursor.close()
        if ( user != None): return 1
        cursor = conn.cursor()
        # SQL query to check if the email exists
        query2 = "SELECT * FROM useraccount WHERE email = %s"
        cursor.execute(query2, (gmail,))
        # Fetch one record
        user = cursor.fetchone()
        cursor.close()
        if ( user != None): return 2
        return 0
    
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None

# -- Function to register new user
def add_new_user(gmail, username, password):
    try:
        conn = init_connection()
        cursor = conn.cursor()

        query = "INSERT INTO useraccount (username, password, email) VALUES (%s, %s, %s)"
        print((username, password, gmail) )
        cursor.execute(query, (username, password, gmail) )
        conn.commit()
   
        cursor.close()
        conn.close()
        st.cache_resource.clear() 
    
    except mysql.connector.Error as err:
        st.error(f"Database Error: {err}")
        return None

### --- FUNCTION FOR CHECKING FILE VALIDITY ---
# Check if file is csv
def is_valid_csv(file_stream):
    """Check if the file content is valid CSV."""
    try:
        file_stream.seek(0) 
        content = file_stream.read().decode('utf-8')  
        file_stream.seek(0)  
        csv.Sniffer().sniff(content)  
        return True
    except Exception:
        return False
# Check if file is excel
def is_valid_excel(file_stream):
    """Check if the file content is valid Excel."""
    try:
        file_stream.seek(0) 
        pd.read_excel(file_stream)  
        return True
    except Exception:
        return False
# Check if file is in xlsx
def is_valid_xlsx(file_path):
    try:
        wb = openpyxl.load_workbook(file_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

### --- FUNCTION FOR PREPROCCESSING DATA ---

def filter_missing_value(data):
    method = st.selectbox("Choose method", ["Drop rows", "Fill with mean"])
    if method == "Drop rows":
        data = data.dropna()
    else:
        #Handle numeric columns
        for col in data.select_dtypes(include=["float", "int"]).columns:
            if data[col].isna().all():
                st.warning(f"Column '{col}' has only missing values and cannot be filled with mean.")
            else:
                data[col] = data[col].fillna(data[col].mean())
        
        #Handle non-numeric columns
        for col in data.select_dtypes(exclude=["float", "int"]).columns:
            if data[col].isna().any():
                data[col] = data[col].fillna("NaN")
    
    return data

def normalize_data(data):
    method = st.selectbox("Normalization Method", ["Min-Max Scaling", "Z-Score Standardization"])
    column = st.selectbox("Select Column to Normalize", data.select_dtypes(include=["float", "int"]).columns)
    
    #Scales data to a fixed range [0, 1]
    if method == "Min-Max Scaling":
        scaler = MinMaxScaler()
        data[[column]] = scaler.fit_transform(data[[column]])

    #Center data around 0, standard deviation of 1
    elif method == "Z-Score Standardization":
        scaler = StandardScaler()
        data[[column]] = scaler.fit_transform(data[[column]])

    return data

def transform_data_types(data):
    columns = st.multiselect("Select Columns", data.columns)
    target_type = st.selectbox("Convert to", ["int", "float", "string", "datetime", "category"])
    for col in columns:
        if target_type == "int":
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
        elif target_type == "float":
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
        elif target_type == "string":
            data[col] = data[col].astype(str)
        elif target_type == "datetime":
            data[col] = pd.to_datetime(data[col], errors="coerce")
        elif target_type == "category":
            data[col] = data[col].astype("category")

    return data

def preprocess_data(data):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.checkbox("Filter missing values"):
            filter_missing_value(data)
    with col2:
        if st.checkbox("Normalize values"):
            normalize_data(data)
    with col3:
        if st.checkbox("Transform Data Types"):
            transform_data_types(data)
    
    return data

def save_processed_data(data, uploaded_file):

    os.makedirs("Preprocessed_Files", exist_ok=True)

    if uploaded_file.name.endswith(".csv"):
        output_path = "Preprocessed_Files/preprocessed_data.csv"
        data.to_csv(output_path, index=False)

    else:
        output_path = "Preprocessed_Files/preprocessed_data.xlsx"   
        data.to_excel(output_path, index=False)

    return output_path

### --- FUNCTION FOR DATA VISUALISATION ---
def pie_chart(data: pd.DataFrame, sales_col: str, region_col: str):
    try:
        data = data.dropna(subset=[region_col, sales_col])
        data[sales_col] = pd.to_numeric(data[sales_col], errors='coerce')
        data = data.dropna(subset=[sales_col])  # Drop rows where sales are NaN after conversion
        region_sales = data.groupby(region_col)[sales_col].sum().reset_index()

        # Check whether DataFrame is empty
        if region_sales.empty:
            st.warning(f"No data available for plotting with {region_col} and {sales_col}.")
            return

        # Pie chart
        fig = px.pie(
            region_sales,
            names=region_col,
            values=sales_col,
            labels={sales_col: "Total Sales", region_col: "Region"}
        )
        st.plotly_chart(fig)

    except:
        st.warning("No Pie Chart can be drawn from these columns.")


def bar_chart(data: pd.DataFrame, category_col: str, sales_col: str):
    try:
        data = data.dropna(subset=[category_col, sales_col])
        data[sales_col] = pd.to_numeric(data[sales_col], errors='coerce')
        data = data.dropna(subset=[sales_col])  # Drop NaN
        category_sales = data.groupby(category_col)[sales_col].sum().reset_index()
        if category_sales.empty:
            st.warning(f"No data available for plotting with {category_col} and {sales_col}.")
            return
        
        # Bar chart
        fig = px.bar(
            category_sales,
            x=category_col,
            y=sales_col,
            title="Sales By Category",
            labels={sales_col: "Total Sales", category_col: "Category"},
            color=category_col
        )
        st.plotly_chart(fig)
    except:
        st.warning("No Bar Chart can be drawn from these columns.")

def tree_map(data: pd.DataFrame, sales_col: str, region_col: str, category_col: str):
    try:
        data = data.dropna(subset=[category_col, sales_col, region_col])
        
        # Ensure the sales column is numeric
        data[sales_col] = pd.to_numeric(data[sales_col], errors='coerce')
        data = data.dropna(subset=[sales_col])  # Drop rows where sales are NaN after conversion
        # Group data by region and category and sum sales
        treemap_data = data.groupby([region_col, category_col])[sales_col].sum().reset_index()
        if treemap_data.empty:
            st.warning("No data available for plotting with these columns.")
            return
        fig = px.treemap(
            treemap_data,
            path=[region_col, category_col],
            values=sales_col,
            title="Sales Breakdown",
            color=sales_col,
            color_continuous_scale="Viridis",
            labels={sales_col: "Total Sales"}
        )
        st.plotly_chart(fig)
    except:
        st.warning("No Tree Map can be drawn from these columns.")

def bar_chart_2(x_data, y_data, x_label, y_label):
    chart_container = st.empty()


    data = pd.DataFrame({'X': x_data, 'Y': y_data})

    try:
        avg_data = data.groupby('X', as_index=False)['Y'].mean()
        # Create Plotly bar chart
        fig = px.bar(
            avg_data,
            x='X',
            y='Y',
            color='X',  # Color based on the unique categories in 'X'
            color_discrete_sequence=px.colors.qualitative.Set2, 
            title=f'Average {y_label} Per {x_label}',
            labels={'X': x_label, 'Y': f'Average {y_label}'}
        )

        # Customize layout
        fig.update_layout(
            xaxis_title=f'{x_label} Category',
            yaxis_title=f'Average {y_label} Data',
            xaxis_tickangle=45,
            margin=dict(l=20, r=20, t=50, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zerolinecolor='rgba(0,0,0,0.3)')
        )

        # Render the chart in Streamlit
        chart_container.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Unable to create the chart: {str(e)}")

def export_report(data, file_name="data_report.html"):
    report = sv.analyze(data)
    #export html
    report.show_html(filepath=file_name)
    return file_name

def data_classification(data, target_col, selected_columns):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix

    # Select relevant columns
    x = data[selected_columns]
    y = data[target_col]

    # Convert datetime columns to numerical values
    for col in x.select_dtypes(include=["datetime64"]).columns:
        x[col] = x[col].astype('int64') // 10**9  # Convert to seconds since epoch

    # Convert target column if it's datetime
    if pd.api.types.is_datetime64_any_dtype(y):
        y = y.astype('category').cat.codes

    # One-hot encode non-numeric columns
    non_numeric_cols = x.select_dtypes(include=["object", "category"]).columns
    x = pd.get_dummies(x, columns=non_numeric_cols)

    # Ensure only numeric columns are included
    x = x.select_dtypes(include=["int64", "float64"])

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=27)

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train the model
    model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Display classification report
    st.write("### Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.format(precision=2))

    # Confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)


#### ------- END HELPER FUNCTION SECTION ------- #####

#### ------- PAGE DESIGN SECTION ------ ####

def login_page():
    # --- Styling ---
    
    st.markdown(
    """
    <style>
    .login-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 50px auto;
        padding: 30px;
        background: linear-gradient(135deg, #4e54c8, #8f94fb);
        color: white;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
        max-width: 450px;
        font-size: 24px;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .login-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .login-input {
        width: 100%;
        margin: 10px 0;
    }
    .login-footer {
        margin-top: 20px;
        font-size: 14px;
        color: #ddd;
    }
    .login-button {
        background-color: #ff7b54;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .login-button:hover {
        background-color: #ff4d2d;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )


    # --- Login Container ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st_lottie(lottie_business, height=150, key="login-animation-1")
    with col2:
        st.markdown(
        """
        <div class="login-container">
            <div class="login-title">🔐 User Login 🔐</div>
        </div>
        """,
        unsafe_allow_html=True,
        )
    with col3:
        st_lottie(lottie_business, height=150, key="login-animation-2")

    # User Input Fields
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border = True):           
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", placeholder="Enter your password", type="password")

            # Feedback Placeholder
            login_feedback = st.empty()

            col_1, col_2, col_3 = st.columns([1, 2, 1])
            with col_2:
                if st.button("Login",type="primary", use_container_width=True):
                    if username.strip() == "" or password.strip() == "":
                        login_feedback.error("Please fill in both username and password.")
                    else:
                        if ( authenticate_user(username, password)):
                            st.success("Login successful! Redirecting...")
                            st.session_state["is_logged_in"] = True
                            set_page("Home")
                        else:
                            login_feedback.error("Invalid username or password!")

    col1, col2, col3 = st.columns([1,4,1])
    with col1:
        # Back to Home Button
        if st.button("Back to Home",use_container_width=True,icon="👈"):
            set_page("Home")
    with col3:
        # Login Footer with Registration Link
        if st.button("Register",use_container_width=True, icon = "👉"):
            set_page("Register")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Footer Section ---
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class='footer'>
        © 2024 Sales Data Analysis App for Data Engineering Project.
    </div>
    """,
    unsafe_allow_html=True,
    )
    create_side_bar()



# ---REGISTER PAGE---

def register_page():
    # --- Styling ---
    st.markdown(
        """
        <style>
        .register-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: 50px auto;
            padding: 30px;
            background: linear-gradient(135deg, #ff7b54, #ff9770);
            color: white;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
            max-width: 500px;
            font-size: 20px;
            font-weight: bold;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .register-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .register-input {
            width: 100%;
            margin: 10px 0;
        }
        .register-button {
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .register-button:hover {
            background-color: #45a049;
        }
        .register-footer {
            margin-top: 20px;
            font-size: 14px;
            color: #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Register Container ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st_lottie(lottie_data_hello, height=150, key="register-animation-1")
    with col2:
        st.markdown(
            """
            <div class="register-container">
                <div class="register-title">✨ Create Account ✨</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st_lottie(lottie_data_hello, height=150, key="register-animation-2")

    # Input Fields
    col1, col2 = st.columns([3, 2])
    with col1:
        email = st.text_input("Email", placeholder="Please enter your email", key="email-input")
        username = st.text_input("Username", placeholder="Choose a username", key="username-input")
        password = st.text_input("Password", placeholder="Choose a password", type="password", key="password-input")
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", key="confirm-password-input")
    with col2:
        lottie_image = load_lottie_from_file(os.path.join(BASE_DIR, "images", "animals.json"))
        st_lottie(lottie_image, loop=True, height=300)
    # Feedback Placeholder
    register_feedback = st.empty()

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Register", type = 'primary', use_container_width=True, icon = "👉"):
            if not email or not username or not password or not confirm_password:
                register_feedback.error("Please fill in all fields.")
            elif password != confirm_password:
                register_feedback.error("Passwords do not match.")
            else:
                if not email or not username or not password or not confirm_password:
                    register_feedback.error("Please fill in all fields.")
                elif password != confirm_password:
                    register_feedback.error("Passwords do not match.")
                else:
                        rs = check_username_exist(username, email)
                        if (rs == 1): st.error("Username already exists! Please choose another.")
                        elif (rs == 2): st.error("Email already registered! Please use a different email.")
                        else:
                            st.success("Registration successful! You can now log in.")
                            add_new_user(email, username, password)
                            print("Success")
        
        # --- Back to Login Button ---
        if st.button("Back to Login",type='primary', key="back-to-login", use_container_width=True, icon = "👈"):
            set_page("Login")

    # Footer Section
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #555;
        }
        </style>
        <div class='footer'>
            © 2024 Sales Data Analysis App for Data Engineering Project.
        </div>
        """,
        unsafe_allow_html=True,
    )
    create_side_bar()

def home_page():
    # --- Page Styling ---
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
            .header {
                text-align: center;
                font-size: 50px;
                color: black;
                margin-bottom: 20px;
            }
            .hero-section {
                text-align: center;
                margin-bottom: 30px;
            }
            .card {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                text-align: center;
                width: 100%;
                color: black;
            }
            .card:hover {
                background: #e7f3e9;
                cursor: pointer;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                font-size: 14px;
                color: #888;
            }
            .background-decor {
                position: absolute;
                top: 15%;
                left: 10%;
                width: 150px;
                height: 150px;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.9), rgba(240, 240, 240, 0.5));
                border-radius: 50%;
                z-index: -1;
                animation: float 6s ease-in-out infinite;
            }
            .background-decor2 {
                position: absolute;
                bottom: 20%;
                right: 15%;
                width: 200px;
                height: 200px;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(230, 230, 230, 0.3));
                border-radius: 50%;
                z-index: -1;
                animation: float-reverse 8s ease-in-out infinite;
            }
            .login-button {
                position: absolute;
                top: 20px;
                right: 20px;
                font-size: 20px;
                padding: 10px 30px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            .login-button:hover {
                background-color: #45a049;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Decorative Background Elements ---
    st.markdown(
        """
        <div class="background-decor"></div>
        <div class="background-decor2"></div>
        """,
        unsafe_allow_html=True,
    )

    # --- Page Header ---
    st.markdown("<div class='header'>🚀 Sales Data Analysis Application 📊</div>", unsafe_allow_html=True)

    # --- Hero Section ---
    st.markdown(
        """
        <div class='hero-section'>
            <h2>Discover Insights. Empower Decisions.</h2>
            <p>Upload your sales data and uncover trends with our interactive visualizations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="hero-animation")

    # --- Login Button ---
    col1, col2, col3 = st.columns([1.5,1,1.5])
    with col2:
        if not st.session_state.get("is_logged_in", False):
            if st.button("Login",type="primary",icon="😃", use_container_width=True):
                set_page("Login")
        else:
            st.success(f"Welcome back, {st.session_state.get('username', 'User')}!")

    # --- Add sidebar ---
    create_side_bar()

    # --- Footer Section ---
    st.markdown(
        """
        <div class='footer'>
            © 2024 Sales Data Analysis App for Data Engineering Project.
        </div>
        """,
        unsafe_allow_html=True,
    )



def introduction_page():
    # --- Page Styling ---
    st.markdown(
        """
        <style>
            .intro-container {
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .intro-header {
                text-align: center;
                font-size: 40px;
                color: #4CAF50;
                margin-bottom: 30px;
            }
            .intro-list {
                list-style: none;
                padding: 0;
                margin: 0;
                font-size: 18px;
                color: #333;
            }
            .intro-list li {
                padding: 10px 0;
                border-bottom: 1px solid #ddd;
                position: relative;
            }
            .intro-list li:last-child {
                border-bottom: none;
            }
            .intro-list li::before {
                content: "✔";
                position: absolute;
                left: -30px;
                font-size: 18px;
                color: #4CAF50;
            }
            .button-container {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 30px;
            }
            .hero-animation {
                margin: 20px auto;
                display: block;
                height: 200px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Animated Header ---
    st.markdown("<div class='intro-header'>Welcome to the Sales Data Analysis App 📖</div>", unsafe_allow_html=True)

    # --- Introduction Section ---
    col1, col2, col3 = st.columns([0.4,1,0.4])
    with col2:
        st.markdown(
            """
            <div class='intro-container'>
                <ul class='intro-list'>
                    <li>Accept a limited types of files: CSV, XLS, XLSX and as big as 200MB.</li>
                    <li>Offer a wide range of methods for pre-proccessing and visualization of data.</li>
                    <li>Support user in exploring correlations between attributes.</li>
                    <li>Support data classification for better understanding of the data.</li>
                    <li>Summarize all information in a comprehensie report that available for download.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # --- Navigation Buttons ---
    col1, col2,col3 = st.columns([1,3,1])
    if col1.button("Back to Home",type="primary",icon="😃", use_container_width=True):
        set_page("Home")
    if col3.button("Go to File Upload",type="primary",icon="😃", use_container_width=True):
        set_page("File Upload")

    # --- Footer Section ---
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class='footer'>
        © 2024 Sales Data Analysis App for Data Engineering Project.
    </div>
    """,
    unsafe_allow_html=True,
    )
    create_side_bar()
        

# ----- DATA UPLOAD PAGE ----- #

def file_upload_page():
    # --- Page Styling ---
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
            .header {
                text-align: center;
                font-size: 50px;
                color: black;
                margin-bottom: 20px;
            }
            .file-upload-section {
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-container {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                margin: 0 auto;
                width: 60%;
                color: black;
            }
            .upload-container:hover {
                background: #e7f3e9;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 0;
                display: inline-block;
            }
            .button:hover {
                background-color: #45a049;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                font-size: 14px;
                color: #888;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Decorative Background Elements ---
    st.markdown(
        """
        <div class="background-decor"></div>
        <div class="background-decor2"></div>
        """,
        unsafe_allow_html=True,
    )

    # --- Page Header ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st_lottie(file_upload_animation, height=100, key="upload_animation-left")
    with col2:
        st.markdown("<div class='header'>Upload Your Data</div>", unsafe_allow_html=True)
    with col3:
        st_lottie(file_upload_animation, height=100, key="upload_animation-right")
    

    # --- File Upload Section ---
    if not st.session_state.get("is_logged_in"):
        st.warning("You must log in to access this page.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Go to Login", use_container_width=True):
                set_page("Login")
        return

    st.markdown(
        """
        <div class="file-upload-section">
            <p>📂 Upload your dataset (CSV, XLS, XLSX) to start analyzing sales data 📂</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a file (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"], label_visibility="collapsed"
    )

    if uploaded_file:
        max_size = 200 * 1024 * 1024  # 200MB
        if uploaded_file.size > max_size:
            st.error(
                f"File is too large! Maximum file size is 200MB. Your file size is {uploaded_file.size / (1024 * 1024):.2f} MB."
            )
            return

        try:
            # Process the uploaded file
            if uploaded_file.name.endswith(".csv"):
                if not is_valid_csv(uploaded_file):
                    st.error("The uploaded file is not a valid CSV.")
                    return
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith("xlsx"):
                if not is_valid_xlsx(uploaded_file):
                    st.error("The uploaded file is not a valid XLSX.")
                    return
                data = pd.read_excel(uploaded_file)
            else:
                if not is_valid_excel(uploaded_file):
                    st.error("The uploaded file is not a valid Excel file.")
                    return
                data = pd.read_excel(uploaded_file)

            st.session_state["data"] = data
            st.success("File uploaded successfully!")
            st_lottie(success_animation, height=150, key="success_animation")

            col_1, col_2 = st.columns([1,0.2])
            with col_1:
                st.subheader("Preview of Uploaded Data")
            with col_2:
                # Create a download button
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download File ",
                    icon = "📄",
                    data=csv,
                    file_name='downloaded_data.csv',
                    mime='text/csv'
                )
            st.dataframe(data.head(50))

            # Generate Profile Report
            col_1, col_2 = st.columns([1,3])
            with col_1:
                if st.button("Generate Data Profile Report"):
                    profile = ProfileReport(data, title="CSGO Report", explorative=True)
                    report_path = "Report.html"
                    profile.to_file(report_path)
                    st.success(f"Successfully create the report, you can download it by click the button below.")
                    with open(report_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Report",
                            data=file,
                            file_name="Report.html",
                            mime="text/html",
                        )

            if st.checkbox("Start Data Preprocessing"):
                preprocess_data(data)
                if st.button("Revert all changes"):
                    data = st.session_state["data"]
                    st.success("Changes reverted")
                    col_1, col_2 = st.columns([1,0.2])
                    with col_1:
                        st.subheader("Preview of Processed Data")
                    with col_2:
                        # Create a download button
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="Download File ",
                            icon = "📄",
                            data=csv,
                            file_name='downloaded_data.csv',
                            mime='text/csv',
                            key = 'button2'
                        )
                    st.dataframe(data.head(50))
                else:
                    col_1, col_2 = st.columns([1,0.2])
                    with col_1:
                        st.subheader("Preview of Processed Data")
                    with col_2:
                        # Create a download button
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="Download File ",
                            icon = "📄",
                            data=csv,
                            file_name='downloaded_data.csv',
                            mime='text/csv',
                            key = 'button3'
                        )
                    st.dataframe(data.head(50))

                save_processed_data(data, uploaded_file)
                st.success("Preprocessed Data saved")

            st.session_state["data"] = data
            col_1, col_2, col_3 = st.columns([1.5,1,1.5])
            with col_2:
                if st.button("Proceed to Data Analysis", use_container_width=True):
                    set_page("Data Analysis")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a dataset to proceed.")

    # --- Sidebar with Buttons ---
    st.sidebar.markdown(
    """
    ### Instruction:
    1. **Log In**: Make sure you are logged in to access all features.
    2. **Upload Data**: Upload your sales data file (CSV, XLS, or XLSX).
    3. **Preview Data**: Review the uploaded data in the main interface.
    4. **Preprocess Data**: Enable preprocessing to clean or modify your dataset.
    5. **Analyze Data**: Proceed to analyze your processed data.
    
    
    **Tips**:
    - Ensure your file is under 200MB.
    - Use the preprocessing section to clean your dataset before analysis.
    - Contact support if you encounter any issues.
    """
    )
    st.sidebar.markdown("---")  
    st.sidebar.markdown("### About Uploading Files:")

    # --- Add an interactive element ---
    if st.sidebar.button("📂 Learn More About Uploading Files"):
        st.sidebar.info(
        """
        **File Requirements**:
        - Supported formats: CSV, XLS, XLSX.
        - Maximum size: 200MB.
        - Ensure data is clean and formatted properly.
        
        **Need Help?** Reach out to our support team for assistance!
        """
    )

    st.sidebar.markdown("---")  
    
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("📖 Introduction", use_container_width=True):
        set_page("Introduction")
    if st.sidebar.button("📂 File Upload", use_container_width=True):
        set_page("File Upload")
    if st.sidebar.button("📊 Data Analysis", use_container_width=True):
        set_page("Data Analysis")

    st.sidebar.markdown("---") 
    # Add an image or logo to the sidebar
    Login1_path = os.path.join(BASE_DIR, "images", "Login_1.png")
    st.sidebar.image(
        Login1_path, 
        use_container_width=True,
    )

    # Additional static note or contact info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Contact")
    st.sidebar.write("📧 Email: support@salesanalysisapp.com")
    st.sidebar.write("📞 Phone: +84 839569959")

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("Back to Home", use_container_width=True):
            set_page("Home")

    # --- Footer Section ---
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class='footer'>
        © 2024 Sales Data Analysis App for Data Engineering Project.
    </div>
    """,
    unsafe_allow_html=True,
    )




# ----- DATA ANALYSIS PAGE ----- #
def data_analysis_page():
    st.title("Data Analysis 📊")
    
    st.header("1. Data Visualization")
    if "data" not in st.session_state:
        st.warning("No data uploaded! Please upload data on the File Upload page.")
    
    create_side_bar()
    # --- Login Check Section ---
    if not st.session_state.get("is_logged_in"):
        st.warning("You must log in to access this page.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Go to Login", use_container_width=True):
                set_page("Login")
        return

    data = st.session_state["data"]
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    
    st.subheader("Data Profile Report")
    if st.button("Generate Data Profile Report"):
        profile = ProfileReport(data, title="CSGO Report", explorative=True)
        report_path = "Report.html"
        profile.to_file(report_path)
        st.success(f"Successfully create the report, you can download it by click the button below.")
        with open(report_path, "rb") as file:
            btn = st.download_button(
                label="Download Report",
                data=file,
                file_name="Report.html",
                mime="text/html",
            )

    # Scatter #
    if not numeric_cols.empty:
        st.subheader("Scatter Plot")
        st.markdown("Select X and Y axes to create a scatter plot.")
        col_1, col_2, col_3 = st.columns([1,1,1])
        with col_1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col_2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        with col_3:
            category_col = st.selectbox("Category Column (optional)", data.columns.insert(0, None), key="scatter_category")
        
        if x_col and y_col:
            fig = px.scatter(data, x=x_col, y=y_col, color=category_col)
            st.plotly_chart(fig)
    else:
        st.warning("No numeric columns available for scatter plot visualization.")

    st.subheader("Pie Chart by Category")
    col1, col2 = st.columns([1,1])
    with col1:
        Category_data = st.selectbox("Select Category", data.columns, key= "pie1")
    with col2:
        Sales_data = st.selectbox("Select Sales Data", data.columns, key="pie2")
    pie_chart(data, Sales_data, Category_data )
        
    st.subheader("Bar Chart (Total Sales Trends)")
    col1, col2 = st.columns([1,1])
    with col1:
        X_data = st.selectbox("Select Category", data.columns, key = "total trends 1")
    with col2:
        Y_data = st.selectbox("Select Sales Data", data.columns, key = "total trends 2")
    bar_chart(data, X_data,Y_data)

    st.subheader("Bar Chart (Avg Sales Trends)")
    col1, col2 = st.columns([1,1])
    with col1:
        X_data = st.selectbox("Select Category", data.columns, key = "avg trends 1")
    with col2:
        Y_data = st.selectbox("Select Sales Data", data.columns, key = "avg trends 2")
    bar_chart_2(data[X_data], data[Y_data], X_data, Y_data)

    st.subheader("TreeMap")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        X_data = st.selectbox("Select Sales Column", data.columns)
    with col2:
        Y_data = st.selectbox("Select Main Category", data.columns)
    with col3:
        Z_data = st.selectbox("Select Sub Category", data.columns)
    tree_map(data, X_data, Y_data, Z_data)
        
    st.subheader("Line Graph (Trend over time)")
    col1, col2 = st.columns([1,1])
    with col1:
        datetime_data = st.selectbox("Select Day Time", data.columns, key = "line 1")
    with col2:
        sales_data = st.selectbox("Select Sales Data", data.columns, key = "line 2")
    try:
        data[datetime_data] = pd.to_datetime(data[datetime_data], errors="coerce")
        data = data.dropna(subset=[datetime_data])  # Remove invalid dates
        sales_trend = data.groupby(datetime_data)[sales_data].sum().reset_index()
        fig = px.line(sales_trend, x=datetime_data, y=sales_data, title="Sales Trend Over Time")
        st.plotly_chart(fig)
    except:
        st.warning("No line graph can be drawn.")
    
    st.markdown("---")
    st.header("2. Further Analysis")
    st.markdown("Explore the correlation between numeric columns in your dataset.")
    col1, col2 = st.columns([2,2])
    with col1:
        if st.button("🔥 Go to Correlation Heatmap and Data Classification 🔥",use_container_width=True):
            set_page("Correlation Heatmap")
    # --- Footer Section ---
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class='footer'>
        © 2024 Sales Data Analysis App for Data Engineering Project.
    </div>
    """,
    unsafe_allow_html=True,
    )


def correlation_heatmap_page():
    st.title("Correlation Heatmap 🔥")
    create_side_bar()
    if "data" not in st.session_state:
        st.warning("No data uploaded! Please upload data on the File Upload page.")
        if st.button("Go to File Upload"):
            set_page("File Upload")
        return
    
    data = st.session_state["data"]
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    
    if not numeric_cols.empty:
        st.subheader("Correlation Matrix")
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation analysis.")
    st.subheader("Machine Learning Classification")
    col1, col2 = st.columns([1,3])
    with col1:
        # Classification Model
        target_col = st.selectbox("Select Target Column", data.columns)
    with col2:
        # Allow users to drop unnecessary columns
        all_columns = data.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to include in the classification model:",
            [col for col in all_columns if col != target_col],  # Exclude the target column
        )
    if st.button("Start Data Classification"):
        try:
            data_classification(data, target_col, selected_columns)
        except Exception as e:
            st.error(f"Error in classification: {e}")

    # --- Footer Section ---
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Back to Data Analysis Page"):
            set_page("File Upload")
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class='footer'>
        © 2024 Sales Data Analysis App for Data Engineering Project.
    </div>
    """,
    unsafe_allow_html=True,
    )

##### --- END PAGE DESIGN SECTION --- ####

##### --- PAGE NAVIGATION SECTION --- ####
def set_page(page_name):
    """Set the current page in the session state."""
    st.session_state["current_page"] = page_name
    st.rerun()

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

st.set_page_config(page_title="Sales Data Analysis", layout="wide")

# Handle navigation
if st.session_state["current_page"] == "Home":
    home_page()
elif st.session_state["current_page"] == "Introduction":
    introduction_page()
elif st.session_state["current_page"] == "File Upload":
    file_upload_page()
elif st.session_state["current_page"] == "Data Analysis":
    data_analysis_page()
elif st.session_state["current_page"] == "Correlation Heatmap":
    correlation_heatmap_page()
elif st.session_state["current_page"] == "Login":
    login_page()
elif st.session_state["current_page"] == "Register":
    register_page()

# show log out button
if st.session_state.get("is_logged_in"):
    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.success("Logged out successfully.")
        set_page("Login")

##### --- END PAGE NAVIGATION SECTION --- ####
