# Covid Vision

## Inspiration
Due to COVID-19 everyone on the planet has been affected. Millions of people have died and many businesses have suffered. As the economy opens slowly, it is important that we take proper precautions so everyone's safe. Therefore we present, Covid Vision, a computer vision powered CCTV alarm system that enforces social distancing.  

## What it does
Covid Vision identifies humans in CCTV footage and calculates the distance among each other. If the distance between any 2 humans is less than 2 m, the application raises an alarm in that particular area and warns the people to maintain social distancing.

## How I built it
The application has been built using Python, OpenCV and Tensorflow. A ResNet-50 model was trained using a Google Compute Engine GPU instance and trained using the Tensorflow framework. Inference is run on every frame of any CCTV video to identify and warn humans.

## Challenges I ran into
One of the most challenges tasks was training the ResNet model for human detection. Figuring out the distance between humans based on different locations in the camera video was another challenging task. 

## What's next for Covid Vision
Prototyping and testing the solution for different scenes like grocery stores, public parks and more.
