import cohere

def parse_srt(file_path):
    indices = []
    timestamps = []
    subtitles = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        subtitle_text = ''
        for i in range(0, len(lines), 4):
            index = int(lines[i].strip())
            indices.append(index)

            timestamp = lines[i + 1].strip()
            timestamp = ' '.join(timestamp.split())  # Remove extra whitespaces
            timestamps.append(timestamp)

            text = lines[i + 2].strip()
            subtitles.append(text)

    return indices, timestamps, subtitles

    
indices, timestamps, subtitles = parse_srt("output_subtitles.srt")
print(str(subtitles))
translated_subtitles = []
def aya_translation(input_text, cohere_api_key):
    co = cohere.Client(cohere_api_key) # This is your trial API key
    fix_prompt = 'Translate each sentence into French, output only the translation:'
    response = co.generate(
        model='c4ai-aya',
        prompt=fix_prompt + input_text,
        max_tokens=20000,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    print(response.generations[0].text)
    return response.generations[0].text

count = 1
for i in subtitles:
    try:
        translated_subtitles.append(aya_translation(i, "BXyTrgsV2PMbRuvDYu9ZwfLHObeTkR4SvPoZAvtf"))
        count += 1
    except:
        break

def generate_srt(indices, timestamps, translated_subtitles, output_file_path):
    with open(output_file_path, 'w') as file:
        for i in range(count):
            file.write(str(indices[i]) + '\n')
            file.write(timestamps[i] + '\n')
            file.write(translated_subtitles[i] + '\n\n')

output_file_path = "translated_subtitles.srt"
generate_srt(indices, timestamps, translated_subtitles, output_file_path)