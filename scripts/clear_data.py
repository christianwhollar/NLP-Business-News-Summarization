import os
import shutil

class FileDeleter:
    def __init__(self, directory_path = 'data/processed/'):
        self.directory_path = directory_path

    def delete_all_in_directory(self):
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
