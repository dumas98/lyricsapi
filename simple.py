from fastapi import FastAPI
import spotipy
import string
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from googletrans import Translator
import re
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

app = FastAPI()

def musixmatch_api(name, artist):
    musixmatch_key = 'e3aaefc9e200cd82edcea97d33eeaf65'
    try:
        url = f'https://api.musixmatch.com/ws/1.1/track.search?q_artist={artist}&q_track={name}&apikey={musixmatch_key}'
        response = requests.get(url=url).json()
        track_id = response['message']['body']['track_list'][0]['track']['track_id']
        url_track_id = f"https://api.musixmatch.com/ws/1.1/track.lyrics.get?track_id={track_id}&apikey={musixmatch_key}"
        response_lyrics = requests.get(url=url_track_id).json()
        lyrics = response_lyrics['message']['body']['lyrics']['lyrics_body']
        lyrics_clean = lyrics.split('\n...')[0].replace("\n", ". ")
        translator = Translator()
        translation = translator.translate(lyrics_clean).text.replace(" . ", "").replace(" .", "")
        return translation
    except:
        return 0

def clean_lyrics(lyrics):
    pattern = r'\.\S'
    matches = re.findall(pattern,lyrics)
    if matches:
        for match in matches:
            temp_list = list(match)
            temp_list.insert(1," ")
            to_replace = "".join(temp_list)
            lyrics = lyrics.replace(match, to_replace)

    ignore = ',.:;?!\''
    for char in string.punctuation:
        if char not in ignore:
            lyrics = lyrics.replace(char, ' ') # Remove Punctuation
    lyrics = re.sub(r'\r\n', '. ', lyrics)
    lyrics = re.sub(r'\n', '. ', lyrics)
    lyrics = lyrics.replace('  ', ' ')

    return lyrics

def read_text(text):
    filedata = text
    article = filedata.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return article

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=1):
    stop_words = stopwords.words('english')
    summarize_text = []

    #Read text and split it
    sentences =  read_text(text)

    #Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    #Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    #Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarize_text.append("".join(ranked_sentence[i][1]))

    return "".join(summarize_text)


@app.get("/get_info/{spotify_track_uri}")
async def lyrics(spotify_track_uri):
    spotify_client_id = '8082fc33d8584af9ba70f63656416be0'
    spotify_client_secret = '05a8d68d55544585b59341715856052f'
    spotify_track_id = spotify_track_uri.replace('spotify:track:','')
    token = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret).get_access_token()['access_token']
    sp = spotipy.Spotify(token)
    meta = sp.track(spotify_track_id)
    name = meta['name']
    artist = meta['album']['artists'][0]['name']

    lyrics = musixmatch_api(name, artist)

    if lyrics:
        lyrics_clean = clean_lyrics(lyrics)
        summary = generate_summary(lyrics_clean)

        return {'summary':summary, 'clean_lyrics': lyrics_clean}
    else:
        return 0
