from pytube import YouTube

# YouTube video URL
url = 'https://www.youtube.com/watch?v=DS90yQt3E5w'

# Download the video
yt = YouTube(url)
stream = yt.streams.get_highest_resolution()
stream.download('video.mp4')