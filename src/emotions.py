
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def construct_model(model):
    print("constructing model...")
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()

#load trained model
def test_model(model):
    print("testing model...")
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary mapping class labels with corresponding emotions
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Neutral", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    draw_rect = (0, 0, 0, 0)
    while True:
    # Capture frame
        ret, frame = cap.read()
        if not ret:
            break;
        
        # Find haar cascade to draw bounding box around face
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.05, minNeighbors=3)

        face = (0, 0, 0, 0)
        for (x, y, w, h) in faces:
            x1, y1, w1, h1 = face
            if w1*h1 < w*h:
                face = (x, y, w, h)
            
    
        x_pad = 10
        y_pad = 40
        x, y, w, h = face
        print(face)
        if w*h > 0:
            draw_rect = face
            roi_gray = gray[max(y-y_pad, 0):min(y+h+y_pad, frame.shape[1]-1), max(x-x_pad, 0):min(x+w+x_pad, frame.shape[0]-1)]
            if (roi_gray.shape[0] > 48 and roi_gray.shape[1] > 48):
                small_gray = cv2.resize(roi_gray, (48, 48))
                cropped_img = np.expand_dims(np.expand_dims(small_gray, -1), 0)
                prediction = model.predict(cropped_img)
                print(prediction)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        x, y, w, h = draw_rect
        cv2.rectangle(frame, (x-x_pad, y-y_pad), (x+w+x_pad*2, y+h+y_pad*2), (255, 0, 0), 2)
    
        cv2.imshow('Video', cv2.resize(frame,(500,500),interpolation = cv2.INTER_CUBIC))
        #cv2.imshow('Video', small_gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def train_model(model):
    print("training model...")
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')
        
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, epsilon=1e-6),metrics=['accuracy'])

    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)

    plot_model_history(model_info)
    model.save_wights('model.h5')


# create model
model = Sequential()
construct_model(model)
#train_model(model)
test_model(model)



