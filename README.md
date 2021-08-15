# ImageK-Means

This is a program used to classify given images into different categories. The program takes already classified images and uses them as training data and uses k-means clustering to classify the images.

The program classifies the testing images and outputs the image name and its classification into a text file. By looking at the output test file, I calculated the programs accuracy as 40%. The feature extractor of the program converts each image into smaller rectangular smaples and converts the smaller images into numerical vector by pixel values. All the vectors of an image is considered as its feature.
