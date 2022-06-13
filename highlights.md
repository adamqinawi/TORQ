# Highlights
## This file was added to represent what I believe to be highlights of the project

# Coding style
Throughout the code, all non-trivial functions have docstrings as such  

```python3
	def train(self,trainingset, load_vocab = False):
        '''function to train model using BOVW algorithm

        parameters:
            -trainingset: list of images in trainingset in form ((image,label),(image, label)...)
            -save: boolean representing whether to save vocabulary and extracted features
            -load_vocab: boolean representing whether to load vocabulary and extracted features from file (debug feature)
        
        returns None
        '''
 
```

# Training function 

```python3
    def __extract_features(self, trainingset):
        '''function to extract features from training set passed using SIFT

        parameters:
            -trainingset: set of training images in the form ((image,label), (image,label)...)

        returns:
            array of SIFT descriptors in the form (((image1feature1,image1feaure2...),image1label),
                                                    ((image2feature1,image2feature2...),image2label)
                                                    ...)
            '''
        train_features =[]
        features_progress = progressBar("Extracting Features: ",len(trainingset), unit="Image")
        for datapoint in trainingset:
            image = datapoint[0]
            label = datapoint[1]
            sift = cv2.SIFT_create() #instantiates a SIFT engine
            keypoints, descriptors = sift.detectAndCompute(image,None) #extracting features

            train_features.append([descriptors,label])
            features_progress.increment()
        return train_features
```

# Predicting function
This function predicts the label using KNN algorithm by finding distances to known datapoints in bag of visual words algorithm and finding K number of minimum distance neighbors using Euclidean distance  

```python3
        '''function to predict label of given image against trained model

        parameters:
        -input_image: image to predict 
        -number_of_neighbors: number of neighbors to compare against in knn search

        returns:
        label predicted by model
        '''
        image = input_image
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(image,None) # find features of input image
        histogram = self.__create_histogram(descriptors)
        dataset = np.array([histogram for histogram,label in self.feature_histogram]) 
        distances = np.sum(np.square(histogram - dataset), axis=1) #find euclidean distance on whole numpy array
        indexes = distances.argsort()[:number_of_neighbors] # find k minimum elements
        labels = [self.feature_histogram[index][1] for index in indexes]
        return statistics.mode(labels)
```

There is of course a lot more we can extract from this code but these I believe to be sufficiently complex
