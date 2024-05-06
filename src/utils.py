import os
import pickle

# Function to save a Python object to a file using pickle
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # Extract the directory path from the given file path
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        # Open the file in binary write mode and use pickle to dump the object into it
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except:
        print("error in saving object at {}".format(file_path))
        
# Function to load a Python object from a file using pickle      
def load_object(file_path):
    try:
        # Open the file in binary read mode and use pickle to load the object from it
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except:
        print("error in loading object at {}".format(file_path))