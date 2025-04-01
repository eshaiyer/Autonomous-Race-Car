# Lane Detection and Tracking Model

This project implements a neural network architecture with a lane tracking mechanism to identify and monitor lane markings in real time from driving footage

## Dataset and Preprocessing

### Lane Dataset
The LaneDataset class prepares road images and their corresponding lane masks for training it. Workflow of LaneDataset class:


-Loads frames and their matching lane masks

-Resizes images to 128×128 pixels

-Normalizes pixel values to the range [0-1]

-Processes masks to distinguish between the current lane and other lanes

-Handles merged lanes and lane splits using horizon splicing and bounded boxes

### process_mask

It plays an important role in the LaneDataset Class and has the functions of:

-Identifies lane contours in the binary mask

-Separates lanes at the horizon when they appear merged

-Calculates lane centroids to distinguish between the current lane and other lanes

### Example output of lane processing of lane mask into current and other lanes:

![image](https://github.com/user-attachments/assets/20a06661-0710-441d-80d7-5ad562585ad5)


## Model Architecture

## Model Architecture

### LaneNet

The core of the model is `LaneNet`, a convolutional neural network (CNN) designed for lane detection consisting of an encoder-decoder structure:

-Encoder: Two downsampling blocks that reduce spatial dimensions while increasing feature depth (3→64→128 channels)

-Decoder: Two upsampling blocks that restore spatial dimensions while decreasing feature depth (128→64→2 channels)

-Output: Two-channel sigmoid output representing:

    Channel 0: Current/ego lane probability map
    Channel 1: Other lanes probability map

### Example raw data for test image under model:

![image](https://github.com/user-attachments/assets/163c498c-233e-4410-b193-78a43b1209a7)


## LaneTracker

The LaneTracker class monitors lane positioning over time to detect lane changes. Workflow of LaneTracker class:

-Maintains a buffer of recent lane positions

-Calculates the centroid of the current lane in a region of interest

-Compares recent lane positions with historical positions

-Triggers an alert when the difference exceeds a threshold (potential lane departure)

-LaneTracker also provides steering directions and degree of steering required for vehicle to stay in the centre of the lane with reference to intitial position of vehicle

### Example directions overlay on output 

![image](https://github.com/user-attachments/assets/d8c7a03a-49d4-4881-91da-6927f4de23ee)


## Training

The training workflow:

-Splits the dataset into 80% training and 20% validation

-Uses BCE loss for the segmentation task

-Employs Adam optimizer with learning rate scheduling

-Calculates IoU (Intersection over Union) as an additional performance metric

-Saves checkpoints and the best model based on validation loss

-Implements early stopping logic

## Inference

### process_frame

The process_frame function involves the following steps:

-Resizes and normalizes input frames

-Feeds them through the trained model

-Thresholds the output to create binary masks

-Processes contours to create smooth lane lines

-Overlays the detected lanes on the original frame

-Displays alerts for lane change detection

### process_video

The process_video function:

-Loads the trained model

-Initializes the lane tracker

-Opens the input video and creates an output video writer

-Processes each frame through the detection pipeline

-Writes the processed frames to the output video

-Includes progress tracking and logging

### Example output of raw data image after frame processing:

![image](https://github.com/user-attachments/assets/0889fbff-0a57-4df9-bcb5-f67e9a080275)

### Example output frame after video input
![Screenshot 2025-03-30 104616](https://github.com/user-attachments/assets/def34f58-06bc-46f4-8767-dca0dd7388ba)



## Main code execution

The main script handles:

-Checking for existing trained models or checkpoints

-Training or resuming training if needed

-Selecting the appropriate model for inference

-Testing on a single frame to verify the model works correctly

-Processing the full video and saving the result

