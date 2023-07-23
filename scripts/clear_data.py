import os
import shutil

class FileDeleter:
    '''
    A class for deleting all files and directories within a specified directory.

    Attributes:
        directory_path (str): The path to the directory from which contents will be deleted.

    Methods:
        __init__(self, directory_path='data/processed/'): Initializes the FileDeleter with a directory path.
        delete_all_in_directory(self): Deletes all files and directories within the specified directory.
    '''
    def __init__(self, directory_path = 'data/processed/'):
        '''
        Initializes the FileDeleter with a directory path.

        Args:
            directory_path (str): The path to the directory from which contents will be deleted.
        '''
        self.directory_path = directory_path

    def delete_all_in_directory(self):
        '''
        Deletes all files and directories within the specified directory.
        '''
        # List all files and directories in the directory
        contents_list = os.listdir(self.directory_path)

        # Iterate through each content (file or directory) and delete it
        for content_name in contents_list:
            content_path = os.path.join(self.directory_path, content_name)
            if os.path.isfile(content_path):
                try:
                    os.remove(content_path)
                    print(f"Deleted file: {content_path}")
                except Exception as e:
                    print(f"Failed to delete file: {content_path} - {e}")
            elif os.path.isdir(content_path):
                try:
                    shutil.rmtree(content_path)
                    print(f"Deleted directory: {content_path}")
                except Exception as e:
                    print(f"Failed to delete directory: {content_path} - {e}")
