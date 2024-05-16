import requests

def generate_image(self, prompt):
    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={
            "authorization": f"Bearer sk-Y7VUr9D0ikr7fjin6MBb1oAVTHO8eOHjqTGUDpScLbddBih6",
            "accept": "image/*"
        },
        files={"none": ''},
        data={
            "prompt": "dog wearing black glasses",
            "output_format": "jpg",
        },
    )

    if response.status_code == 200:
        with open(".images/generated_image.jpg", 'wb') as file:
            file.write(response.content)
    else:
        raise Exception(str(response.json()))