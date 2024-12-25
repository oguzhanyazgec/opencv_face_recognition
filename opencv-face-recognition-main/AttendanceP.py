import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime


# Yüz resimlerinin bulunduğu dizin yolu
path = 'faces'

# Yüz resimlerini depolamak için boş listeler oluşturulur
images = []
classNames = []

# Dizindeki dosyaların listesi alınır
myList = os.listdir(path)
print(myList)

# Yüz resimlerinin ve isimlerinin yüklenmesi
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


# Yüzleri kodlayan fonksiyon
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Katılımı işaretleme fonksiyonu
def markAttendance(name, min_match, max_match):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtsString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtsString},{min_match},{max_match}')


# Resimleri kodlayın
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Kamerayı başlatın
cap = cv2.VideoCapture(0)

# Ana döngü
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Görüntüdeki yüzleri tanıma ve kodlama
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFaces, faceLoc in zip(encodesCurFrame, faceCurFrame):
        # Tanımlanan yüzlerle eşleşmeleri karşılaştırma
        matches = face_recognition.compare_faces(encodeListKnown, encodeFaces)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFaces)
        matchIndex = np.argmin(faceDis)

        name = "Unknown"

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            min_match = min(faceDis) * 100
            max_match = max(faceDis) * 100

        # Yüzün etrafına dikdörtgen çizme ve ismi ekleme
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Katılımı işaretleme
        markAttendance(name, min_match, max_match)

    # Görüntüyü gösterme
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakma ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()