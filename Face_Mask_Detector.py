import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout

# --- Data Loading ---
categories = ['with_mask', 'without_mask']
data = []

for category in categories:
    path = os.path.join('train', category)
    label = categories.index(category)
    
    if not os.path.exists(path):
        print(f"Directory not found: {path}")
        continue

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Convert BGR to RGB so training matches VGG16 expectations
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            data.append((img, label))
        else:
            print(f"Skipping invalid file: {file}")

print(f"Successfully loaded {len(data)} images.")
random.shuffle(data)

X = []
y = []  
for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X) 
y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Model Building ---
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.trainable = False 

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(128, activation='relu')) # A dense layer for better feature learning
model.add(Dropout(0.5))                  # Dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# --- Helper Functions ---
def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        return img[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def detect_face_mask(img):
    # Ensure live crop is also RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    y_pred = model.predict(img.reshape(1, 224, 224, 3), verbose=0)
    return 1 if y_pred[0][0] > 0.5 else 0

# --- Real-time Detection Loop ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    face_img, coords = detect_face(frame)
    
    if face_img is not None:
        x, y, w, h = coords
        y_pred = detect_face_mask(face_img)
        
        # Drawing rectangle and label
        if y_pred == 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            draw_label(frame, 'Mask', (x, y-10), (0, 255, 0))
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            draw_label(frame, 'No Mask', (x, y-10), (0, 0, 255))
            
    cv2.imshow('Face Mask Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()

