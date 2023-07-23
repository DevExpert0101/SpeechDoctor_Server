from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from pydub import AudioSegment
import scipy.signal
import soundfile as sf
import whisper
import json
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import time
import os
import datetime


# available models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
model = whisper.load_model("small")

model_path = "vosk-model-en-us-0.22"
model_vosk = Model(model_path) 
filter_words = ['well','oh', 'um' 'er' ,'ah', 'uh', 'hmm', 'like', 'actually', 'basically', 'seriously', 'literally', 'totally', 'clearly', 'you see', 'you know'
    , 'i mean', 'you know what I mean','yeah', 'at the end of the day', 'believe me', 'i guess' , 'i suppose', 'or something', 'okay' , 'so', 'right' , 'hmm' , 'uh' ,'huh']


print('Server is started ...')

app = FastAPI()

class Word:
    ''' A class representing a word from the JSON format for vosk speech recognition API '''

    def __init__(self, dict):
        '''
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        '''

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self):
        ''' Returns a string describing this instance '''
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf*100)

class AudioProcessResult(BaseModel):
    filename: str
    duration_seconds: float  # You can add more fields depending on the processing you want to do

def process_audio(folder_path: str, file_name: str):
    
    start_ = time.time()    

    fname = file_name.split('.')[0]
    # print(fname)
    src = folder_path + "/" + file_name
    dst = folder_path + "/" + file_name.replace('m4a', 'wav')
    sound = AudioSegment.from_file(src)
    sound.export(dst, format="wav")

    # Load the original audio file
    audio, original_sample_rate = sf.read(dst)

    # Set the target sample rate
    target_sample_rate = 16000

    # Calculate the resampling ratio
    resampling_ratio = target_sample_rate / original_sample_rate

    # Resample the audio to the target sample rate
    resampled_audio = scipy.signal.resample(audio, int(len(audio) * resampling_ratio))

    # Save the resampled audio to a new file
    sf.write(folder_path + "/" + fname + "_resampled.wav", resampled_audio, target_sample_rate)


    sound = AudioSegment.from_wav(folder_path + "/" + fname + "_resampled.wav")
    sound = sound.set_channels(1)
    sound.export(folder_path + "/" + fname + "_resampled_mono.wav", format="wav")

    result = model.transcribe(folder_path + "/" + fname + "_resampled_mono.wav")
    # print(result['text'])
    sentences = []
    json_sentence = []

    tmp = ""
    start_time = result['segments'][0]['start']
    end_time = result['segments'][0]['end']

    for segment in result['segments']:
      tmp += segment['text']
      if "." in segment['text']:
        sentences.append(tmp)
        end_time = segment['end']
        json_sentence.append({'start': start_time, 'end': end_time, 'sentence': tmp})
        tmp = ""
        start_time = end_time

    sentence_num = len(sentences)

    print('The number of sentence is : ', sentence_num)
    
    # json_sentence_data
    json_sentence_data = json.dumps(json_sentence)
    with open(folder_path + "/" +'sentence_json_data.json', 'w') as outfile:
       outfile.write(json_sentence_data)

    
    # Part for using VOSK model
     
    audio_filename = folder_path + "/" + fname + "_resampled_mono.wav"


    wf = wave.open(audio_filename, "rb")
    rec = KaldiRecognizer(model_vosk, wf.getframerate())
    rec.SetWords(True)

    # get the list of JSON dictionaries
    results = []

    # recognize speech using vosk model
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)

    # convert list of JSON dictionaries to list of 'Word' objects
    list_of_Words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for obj in sentence['result']:
            w = Word(obj)  # create custom Word object
            list_of_Words.append(w)  # and add it to list

    wf.close()  # close audiofile

    # output to the screen

    word_num = len(list_of_Words)
    json_words = []
    for word in list_of_Words:
        # print(word.to_string())
        json_words.append({'start': word.start, 'end': word.end, 'word': word.word})

    print(json_words)     

    word_num = len(list_of_Words)
    json_words = []
    for word in list_of_Words:
        # print(word.to_string())
        json_words.append({'start': word.start, 'end': word.end, 'word': word.word})

    print("The number of words is : ", len(json_words))

    json_words_data = json.dumps(json_words)
    with open(folder_path + "/" + 'words_json_data_final.json', 'w') as outfile:
        outfile.write(json_words_data)

    pause = []
    for i in range(len(list_of_Words) - 1):
        word = list_of_Words[i]
        next_word = list_of_Words[i + 1]
        if word.end < next_word.start - 0.5:
            pause.append({'start': word.end, 'end': next_word.start})

    num_pause = len(pause)
    print("The number of puases is : ", len(pause))
    

    json_pause_data = json.dumps(pause)
    with open(folder_path + "/" + 'pause_json_data_Gujrati_en.json', 'w') as outfile:
        outfile.write(json_pause_data)

    
    
    filtered_words = []
    for word in list_of_Words:
        w = word.to_string().split()
        if w[0].lower() in filter_words:
            print(word.to_string())
            filtered_words.append({'start': word.start, 'end': word.end, 'filtered_word': word.word})

    num_filtered_words = len(filtered_words)
    print("The number of filtered words : ", num_filtered_words)
    

    json_filtered_data = json.dumps(filtered_words)
    with open(folder_path + "/" + 'filtered_json_data_Gujrati_en.json', 'w') as outfile:
        outfile.write(json_filtered_data)


    end_ = time.time() - start_
    print("--- %s seconds ---" % end_)
    
    return json_sentence_data, json_words, json_pause_data, json_filtered_data, end_

@app.post("/upload/audio/")
async def upload_audio_file(file: UploadFile = File(...)):

    c_directory = os.getcwd()
    
    c_year = str(datetime.date.today().year)
    c_date = str(datetime.date.today().month) + '-' + str(datetime.date.today().day)
    
    folder_name = c_directory + f"/data/{c_year}/{c_date}/{file.filename.split('.')[0]}"
    fname = folder_name + f"/{file.filename}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Save the uploaded audio file to a temporary location

    with open(fname, "wb") as buffer:
        buffer.write(await file.read())    

    # Process the uploaded audio file
    sentence_json = process_audio(folder_name, file.filename)
    

    # Return the processing result
    # return AudioProcessResult(filename=file.filename, duration_seconds=duration_seconds)
    return sentence_json
