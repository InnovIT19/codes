import re

# Function to validate phone number
def is_valid_phone_number(phone_number):
    # Check if the phone number is exactly 10 digits
    return bool(re.match(r'^\d{10}$', phone_number))

# Function for gender selection
def select_gender():
    while True:
        gender_input = input("Enter your gender (1 for male / 2 for female): ").strip()
        if gender_input == '1':
            return 'male'
        elif gender_input == '2':
            return 'female'
        else:
            print("Invalid input. Please enter '1' for male or '2' for female.")

# Function for user registration
def user_registration(existing_phone_numbers, existing_usernames):
    user_data = {}

    while True:
        phone_number = input("Enter your phone number (10 digits): ").strip()
        if not is_valid_phone_number(phone_number):
            print("Invalid phone number. Please enter a 10-digit phone number.")
            continue

        if phone_number in existing_phone_numbers:
            print("Already a customer.")
            return None

        gender = select_gender()

        # Store user data
        user_data['phone_number'] = phone_number
        user_data['gender'] = gender

        print("\nRegistration Successful!")
        print(f"Phone Number: {phone_number}")
        print(f"Gender: {gender.capitalize()}")

        break

    while True:
        username = input("Enter your username: ").strip()
        if username in existing_usernames:
            print("Username is already in use. Please try a different one.")
            continue

        user_data['username'] = username
        existing_usernames.add(username)

        print("\nUsername registration Successful!")
        print(f"Username: {username}")

        break

    return user_data

# Function for user login
def user_login(existing_phone_numbers, user_data):
    while True:
        phone_number = input("Enter your phone number to log in (10 digits): ").strip()
        if not is_valid_phone_number(phone_number):
            print("Invalid phone number. Please enter a 10-digit phone number.")
            continue

        if phone_number not in existing_phone_numbers:
            print("Phone number not registered. Please register first.")
            break

        username = input("Enter your username: ").strip()
        if username != user_data.get(phone_number, {}).get('username'):
            print("Username does not match. Please try again.")
            continue

        print("Login successful.")
        break

# Sets to keep track of existing phone numbers and usernames
existing_phone_numbers = set()
existing_usernames = set()
user_data = {}

# Run the user registration function
if __name__ == "__main__":
    registration_data = user_registration(existing_phone_numbers, existing_usernames)
    if registration_data:
        existing_phone_numbers.add(registration_data['phone_number'])
        user_data[registration_data['phone_number']] = registration_data
        print("\nRegistered User Data:", registration_data)

    # Simulate login after registration
    user_login(existing_phone_numbers, user_data)