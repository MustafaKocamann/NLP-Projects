# Importing the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

data = [
    "Bugün hava yürüyüş yapmak için çok güzel",
    "Sabah erken kalkıp kahvaltı hazırladım",
    "Otobüs biraz geç geldiği için işe geç kaldım",
    "Akşam eve giderken markete uğramam gerekiyor",
    "Bugün çok yoğun bir gün geçirdim",
    "Kahvemi alıp bilgisayarın başına oturdum",
    "Telefonumun şarjı çok hızlı bitiyor",
    "Dışarı çıkmadan önce hava durumuna baktım",
    "Arkadaşlarımla hafta sonu buluşmayı planladık",
    "Bugün trafikte uzun süre beklemek zorunda kaldım",
    "Öğle yemeğinde evden getirdiğim yemeği yedim",
    "Akşam yemeğinden sonra kısa bir yürüyüş yaptım",
    "Sabah işe giderken müzik dinlemeyi seviyorum",
    "Bugün kendimi biraz yorgun hissediyorum",
    "Hafta sonu için küçük bir tatil planı yaptık",
    "Eve gelince ilk iş üstümü değiştirdim",
    "Bugün toplantı beklediğimden uzun sürdü",
    "Yeni bir kitap almaya karar verdim",
    "Akşam erken yatmayı düşünüyorum",
    "Sabah kahvesi olmadan kendime gelemiyorum",
    "Bugün yağmur yağacağı söyleniyor",
    "İşten sonra spor salonuna gittim",
    "Evde temizlik yapmam gereken çok iş var",
    "Telefonla uzun bir görüşme yaptım",
    "Bugün işler planladığım gibi gitmedi",
    "Akşam arkadaşım beni aradı",
    "Sabah alarmı duymadığım için geç uyandım",
    "Bugün biraz kendime zaman ayırmak istiyorum",
    "Yemek yaparken müzik dinledim",
    "Akşam televizyon izleyerek dinlendim",
    "Bugün dışarıda yemek yemeye karar verdik",
    "Sabah yürüyüşü bana iyi geliyor",
    "İnternetten birkaç şey sipariş verdim",
    "Bugün hava beklediğimden daha soğuk",
    "Eve geç kalacağımı haber verdim",
    "Akşam yemeğini ailemle birlikte yedik",
    "Bugün yapılacaklar listem çok kabarık",
    "Kahvaltıda çay içmeyi tercih ediyorum",
    "Bugün kendimi daha motive hissediyorum",
    "Akşamüstü kısa bir mola verdim",
    "Sabah işe başlamadan önce maillerimi kontrol ettim",
    "Bugün biraz erken çıkmayı planlıyorum",
    "Eve dönerken müzik dinledim",
    "Bugün yeni bir şey öğrenmek istiyorum",
    "Akşam yemeği için ne yapacağımıza karar veremedik",
    "Sabah spor yapmak bana enerji veriyor",
    "Bugün dışarı çıkmak istemiyorum",
    "Akşam uzun süre telefonla vakit geçirdim",
    "Yarın için plan yapmam gerekiyor"
    "Sabah pencereden içeri güneş ışığı giriyordu",
    "Bugün dışarı çıkmadan önce montumu aldım",
    "İşe gitmeden önce kısa bir telefon görüşmesi yaptım",
    "Akşam yemeği için evde bir şeyler hazırladım",
    "Bugün işler beklediğimden daha hızlı bitti",
    "Eve dönerken yolda eski bir arkadaşla karşılaştım",
    "Sabah kahvaltısını aceleyle yaptım",
    "Bugün bilgisayar başında uzun süre oturdum",
    "Akşamüstü biraz dinlenmeye karar verdim",
    "Markete girince ihtiyacım olmayan şeyler aldım",
    "Bugün toplantıya zamanında yetiştim",
    "Sabah uyandığımda hava kapalıydı",
    "Eve gelince biraz müzik açtım",
    "Bugün kendime küçük bir hedef koydum",
    "Akşam yemeğinden sonra çay içtik",
    "Sabah işe giderken yağmur başladı",
    "Bugün çok fazla mesaj aldım",
    "Eve erken gelmek bana iyi hissettirdi",
    "Akşam televizyon karşısında uyuyakaldım",
    "Bugün yapılacak işleri sıraya koydum",
    "Sabah alarmı ikinci kez çaldı",
    "Bugün dışarıda çok rüzgar vardı",
    "Eve döndüğümde yorgunluktan hemen uzandım",
    "Akşam arkadaşlarla kısa bir sohbet ettik",
    "Bugün bilgisayarda dosyaları düzenledim",
    "Sabah yürürken kahve aldım",
    "Akşam için plan yapmadık",
    "Bugün kendimi daha sakin hissediyorum",
    "Eve gelince haberleri izledim",
    "Sabah işe başlamadan önce notlar aldım",
    "Bugün uzun bir liste hazırladım",
    "Akşam yemeği biraz geç hazırlandı",
    "Sabah dışarı çıkarken anahtarımı unuttum",
    "Bugün telefonum sürekli çaldı",
    "Eve dönerken hava kararmıştı",
    "Akşam sessiz bir ortamda dinlendim",
    "Bugün yeni bir alışkanlık denemek istiyorum",
    "Sabah bilgisayarı açar açmaz çalışmaya başladım",
    "Akşam yemeğinden sonra kısa bir yürüyüş yaptık",
    "Bugün dışarıda fazla kalmadım",
    "Eve gelince üstümü değiştirip rahatladım",
    "Sabah kahvemi ayakta içtim",
    "Bugün biraz dalgın hissediyorum",
    "Akşam erken uyumaya karar verdim",
    "Sabah planladığım her şeyi yapamadım",
    "Bugün işler biraz karışıktı",
    "Eve geç saatte döndüm",
    "Akşam telefonumun sesini kapattım",
    "Yarın için hazırlanmayı düşünüyorum"
]

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1  # +1 for padding token

input_sequences = []
for text in data:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print(f"input_sequences: \n{input_sequences}")

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, max_sequence_len, padding='pre')
print(f"max_sequence_len: {max_sequence_len}")

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes = total_words)

# LSTM Model
model = Sequential()
model.add(Embedding(total_words, 50, input_length= X.shape[1]))
model.add(LSTM(100))
model.add(Dense(total_words, activation = "softmax")) # multiclass classification problem

# Compile the model
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)


# Training the model
model.fit(X, y, epochs = 100, verbose = 1)

# Generation text test
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis= -1)[0]
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text = seed_text + " " + predicted_word
    return seed_text

print(generate_text("Bugün", 3))