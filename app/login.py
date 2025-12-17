import streamlit as st

# A simple user database (you can replace it with a real database later)
users_db = {
    "patient": {},  # stores patient usernames and passwords
    "doctor": {}    # stores doctor usernames and passwords
}

# Function to register a new user (both patient and doctor)
def register_user(role, username, password):
    if username in users_db[role]:
        st.error("This username is already taken!")
        return False
    users_db[role][username] = password
    st.success(f"User {username} registered as {role} successfully!")
    return True

# Function to log in an existing user (both patient and doctor)
def login_user(role, username, password):
    if username in users_db[role] and users_db[role][username] == password:
        st.session_state.username = username  # Store username in session state
        st.session_state.role = role          # Store role (patient/doctor)
        st.session_state.authenticated = True  # Set the user as authenticated
        st.success(f"Welcome back, {username} ({role})!")
        return True
    st.error("Invalid username or password!")
    return False

# Function to log out the current user
def logout_user():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.experimental_rerun()  # Rerun to reset the app

# Function to handle user registration and login logic
def user_authentication():
    # Show login or registration options
    st.title("Login / Register")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = ""

    if st.session_state.authenticated:
        # If the user is authenticated, show the role and provide the logout option
        st.sidebar.write(f"Welcome, {st.session_state.username} ({st.session_state.role.capitalize()})")
        if st.sidebar.button("Logout"):
            logout_user()
    else:
        # Ask user to either login or register
        login_or_register = st.selectbox("Choose an option", ["Login", "Register"])

        role = st.selectbox("Select your role", ["patient", "doctor"])

        # Handle login logic
        if login_or_register == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username and password:
                    login_user(role, username, password)
                else:
                    st.error("Please enter both username and password!")

        # Handle registration logic
        elif login_or_register == "Register":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            if st.button("Register"):
                if username and password and confirm_password:
                    if password == confirm_password:
                        register_user(role, username, password)
                    else:
                        st.error("Passwords do not match!")
                else:
                    st.error("Please enter all required fields!")

