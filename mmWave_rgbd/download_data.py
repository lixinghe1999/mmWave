import requests
import os

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

if __name__ == "__main__":
    # TAKE ID FROM SHAREABLE LINK
    file_id = '16rdlZSHpIzYXEBHTW40V75RPN-yzn0oZ'
    # DESTINATION FILE ON YOUR DISK
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fp = '1_person_walking_128loops.bin'
    download_file_from_google_drive(file_id, os.path.join(data_dir, fp))