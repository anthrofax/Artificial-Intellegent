# 2. Mengimport library yang relevan
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os


stemmer = LancasterStemmer()

import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.estimator import regression
import pickle
import random
import json

# 3. Membaca file json
# Menggunakan json.loads() dan menyimpan isinya ke variable bernama 'intents' (type: dict)
data_file = open("maksud.json").read()
intents = json.loads(data_file)

# 4. Pra-pemrosesan data
words = []
classes = []
documents = []
ignore = ["?"]

# Melakukan perulangan pada setiap kalimat pada intents pattern
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Melakukan tokenisasi setiap kata di dalam kalimat
        w = nltk.word_tokenize(pattern)

        # Menambahkan kata ke dalam list 'words'
        words.extend(w)

        # Menambahkan 'words' ke 'documents'
        documents.append((w, intent["tag"]))

        # Menambahkan tag ke dalam classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Melakukan stemming and mengubah setiap kata menjadi huruf kecil, dan juga menghapus duplikasi
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# Menghapus duplikasi pada classes
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unquie stemmed words", words)

# Menentukan data training
training = []
output = []

# Membuat array kosong untuk output
output_empty = [0] * len(classes)

# Membuat training set, bag of words di setiap kalimat
for doc in documents:
    # Inisialisas bag of words
    bag = []

    # List dari kata yang telah tertokenisasi untuk suatu pola
    pattern_words = doc[0]

    # Melakukan stem pada setiap kata
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    # Membuat array Bag of Words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

        # Output adalah '1 untuk tag saat ini dan '0  untuk tag sisanya
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

# Mengacak fitur dan membuat nya menjadi np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Membuat training list
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# # 6. Membangun model jaringan syaraf tiruan
   
tf.compat.v1.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
net = regression(net)

# Mendefinisikan model dan mengatur tensorboard
model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")

# Memeriksa apakah model sudah ada
if os.path.isfile("model-indo.tflearn.index") == False:
    # Memulai pelatihan
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model-indo.tflearn")
    # Menyimpan data yang diperlukan untuk penggunaan di masa mendatang
    pickle.dump(
        {"words": words, "classes": classes, "train_x": train_x, "train_y": train_y},
        open("training_data", "wb"),
    )

# 7. Mempersiapkan data dan model
# Memulihkan semua stuktur data
data = pickle.load(open("training_data", "rb"))
words = data["words"]
classes = data["classes"]
train_x = data["train_x"]
train_y = data["train_y"]

# Memuat model yang telah disimpan
model.load("./model-indo.tflearn")


# 8. Membuat fungsi untuk menangani input pengguna
def clean_up_sentence(sentence):
    # Melakukan tokenisasi pada pola
    sentence_words = nltk.word_tokenize(sentence)

    # Melakukan stem pada setiap kata
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words


# Mengembalikan array bag of words: 0 atau 1 untuk setiap kata di dalam bag yang ada pada kalimat
def bow(sentence, words, show_etails=False):
    # Melakukan tokenisasi pada pola
    sentence_words = clean_up_sentence(sentence)

    # Men-generate bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_etails:
                    print("found in bag: %s" % w)

    return np.array(bag)


ERROR_THRESHOLD = 0.30


def classify(sentence):
    # Men-generate probabilitas dari model
    results = model.predict([bow(sentence, words)])[0]

    # Memfilter prediksi yang di bawah batasan
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]

    # Mensortir berdasarkan probabilitas
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))

    # Mengembalikan tuple yang berisikan intent dan probabilitasnya
    return return_list


def response(sentence, userID="123", show_details=False):
    results = classify(sentence)

    # Jika kita memiliki klasifikasi maka cocokan dengan tag intent
    if results:
        # Ulang selama masih ada kecocokan dengan proses
        while results:
            for i in intents["intents"]:
                # find a tag matching the first result
                if i["tag"] == results[0][0]:
                    # A random response from the intent
                    return print(random.choice(i["responses"]))

            results.pop(0)


print("Tekan 0 jika kamu tidak ingin chat dengan chatbot kami")
while True:
    message = input("")

    if message == "0":
        break

    result = response(message)
    print(result)
