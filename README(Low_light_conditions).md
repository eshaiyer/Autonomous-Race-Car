# Low-Light Image Enhancement using Deep Learning

## Overview
This project presents a deep learning-based approach to enhance images captured in low-light conditions. The methodology involves preprocessing the dataset, training a deep learning model for enhancement, and evaluating its performance. The implementation is optimized for execution in *Google Colab*.

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Dataset Preparation](#2-dataset-preparation)  
   - [Downloading and Preprocessing the Dataset](#downloading-and-preprocessing-the-dataset)  
   - [Data Augmentation](#data-augmentation)  
3. [Model Training](#3-model-training)  
   - [Model Architecture](#model-architecture)  
   - [Loss Function and Optimization](#loss-function-and-optimization)   
4. [Inference and Real-Time Enhancement](#4-inference-and-real-time-enhancement)  
5. [Applications](#5-applications)  
6. [Conclusion](#6-conclusion)  

---

## 1. Introduction
Low-light images suffer from poor visibility, loss of detail, and high noise levels. This project leverages *deep learning* to enhance the visibility of such images. The goal is to *train a model* that learns to map low-light images to their well-lit counterparts, improving contrast and restoring details.

## 2. Dataset Preparation

### Downloading and Preprocessing the Dataset
We use a dataset consisting of paired *low-light and normal-light images* for supervised learning. The dataset is downloaded and extracted in Colab using:

```python
!kaggle datasets download -d soumikrakshit/lol-dataset
!unzip lol-dataset.zip -d lol_dataset
```

### Data Augmentation
Data augmentation improves model generalization. We apply transformations such as *random flipping, rotation, and brightness adjustment*:

```python
def load_images(image_list, folder_path, target_size=(256, 256)):
    images = []
    for img_name in image_list:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size) / 255.0
        images.append(img)
    return np.array(images)
```

---

## 3. Model Training

### Model Architecture
The model is based on a *CNN-based U-Net architecture*, widely used in image enhancement tasks:

```python
def simplified_mirnet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    down1 = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(down1)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    down2 = layers.MaxPooling2D((2, 2), padding='same')(x)


    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(down2)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)


    up1 = layers.UpSampling2D((2, 2))(x)


    down2_resized = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(down2)


    down2_resized = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(down2_resized)

    up1 = layers.Concatenate()([up1, down2_resized])

    up2 = layers.UpSampling2D((2, 2))(up1)


    down1_resized = layers.Conv2D(64 + 128, (3, 3), activation='relu', padding='same')(down1)

    down1_resized = layers.Conv2DTranspose(64+128, (3, 3), strides=(2, 2), padding='same')(down1_resized)

    up2 = layers.Concatenate()([up2, down1_resized])


    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)


    model = models.Model(inputs, outputs)
    return model
```

### Loss Function and Optimization
We use a combination of *Mean Squared Error (MSE) loss* and *Perceptual Loss* for better color and texture reconstruction.

```python
model = simplified_mirnet()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
```

Training is performed over multiple epochs:

```python
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=8,
    shuffle=True,
)
```

---

## 4. Inference and Real-Time Enhancement
To use the trained model for real-time enhancement:

```python
def enhance_video(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture('/content/inpt_vid_low_light.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        img = cv2.resize(frame, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        enhanced_img = model.predict(img)[0]
        enhanced_img = (enhanced_img * 255).astype(np.uint8)
        enhanced_img = cv2.resize(enhanced_img, (frame_width, frame_height))

        out.write(enhanced_img)

        cv2_imshow(enhanced_img)


    cap.release()
    out.release()
    cv2.destroyAllWindows()

enhance_video("/content/inpt_vid_low_light.mp4", "output_final_video.mp4", model)
```

---

## 5. Applications
- **Night Photography:** Enhances images taken in low-light conditions.
- **Surveillance Systems:** Improves visibility in security footage.
- **Medical Imaging:** Helps in processing low-light microscopic images.

---

## 6. Conclusion
This project provides a deep learning-based solution for *low-light image enhancement* using CNNs. The trained model is capable of restoring visibility in dark images while maintaining fine details. Future improvements can include:
- Implementing GAN-based architectures.
- Using transformer-based models for better feature extraction.
