import sys
import statistics
import numpy as np
from sklearn.cluster import KMeans

class progressBar:
    '''helper class to print pretty progress bars'''
    def __init__(self,string,steps,unit="",size = 40,output=sys.stdout) -> None:
        self.steps = steps
        self.size = size
        self.output = output
        self.progress = 0
        self.string = string
        self.unit = unit

    def increment(self):
        self.progress+=1
        stepCount = int(self.size * self.progress / self.steps)
        self.output.flush()
        self.output.write("%s[%s%s] %s %i/%i\r" % (self.string, "█"*stepCount, " "*(self.size-stepCount),self.unit, self.progress, self.steps))
        self.output.flush()
        if self.progress == self.steps:
            self.output.write("\n")
            self.output.flush

class KNNModel:

    def __init__(self, number_of_k_clusters) -> None:
        self.vocabulary = []
        self.image_histograms = []
        self.vocab_savepath="vocab.npy"
        self.features_savepath = "features.npy"
        self.feature_histogram=[]
        self.kmeans =  KMeans(n_clusters = number_of_k_clusters)

    def train(self,trainingset, load_vocab = False):
        '''function to train model using BOVW algorithm

        parameters:
            -trainingset: list of images in trainingset in form ((image,label),(image, label)...)
            -save: boolean representing whether to save vocabulary and extracted features
            -load_vocab: boolean representing whether to load vocabulary and extracted features from file (debug feature)
        
        returns None
        '''
        train_features = [] #list to hold extracted SIFT features of training images
        if load_vocab:
            self.vocabulary = np.load(self.vocab_savepath, allow_pickle = True)
            train_features = np.load(self.features_savepath,allow_pickle = True)
            print("Features and vocabulary loaded from file")
        else:
            train_features = self.__extract_features(trainingset)

        descriptor_list = [descriptor for image,_ in train_features for descriptor in image]
        self.vocabulary = self.__cluster(descriptor_list)

        histogram_progress = progressBar("Generating Tr. Histograms:",
                len(train_features),unit="Image")
        for image,label in train_features:
            histogram = self.__create_histogram(image)
            self.feature_histogram.append([histogram,label])
            histogram_progress.increment()

    def __cluster(self, descriptors):
        '''function to perfrom kmeans clustering of descriptors to find vocabulary

        parameters:
            -descriptors: list of all unordered descriptors

        returns:
            array of descriptors to be used as vocabulary
        '''
        kmeans = self.kmeans
        print("Finding feature clusters to form vocabulary")
        kmeans.fit(descriptors)
        print("Clusters found, vocabulary generated")
        vocabulary = kmeans.cluster_centers_
        return vocabulary

    def __create_histogram(self, descriptors):
        '''function to create histogram against known vocabulary

        parameters:
            -descriptors: list of descriptors from one image to compare against vocabulary

        returns:
            array representing histogram against known vocab
        '''
        kmeans = self.kmeans
        number_of_words= len(kmeans.cluster_centers_)
        histogram = np.zeros(number_of_words)
        descriptors = np.array(descriptors, dtype=float)
        for index in kmeans.predict(descriptors):
            histogram[index] += 1
        return histogram

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
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image,None)

            train_features.append([descriptors,label])
            features_progress.increment()
        return train_features

    def predict_label(self, input_image, number_of_neighbors):
        '''function to predict label of given image against trained model

        parameters:
        -input_image: image to predict 
        -number_of_neighbors: number of neighbors to compare against in knn search

        returns:
        label predicted by model
        '''
        image = input_image
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(image,None)
        histogram = self.__create_histogram(descriptors)
        dataset = np.array([histogram for histogram,label in self.feature_histogram]) 
        distances = np.sum(np.square(histogram - dataset), axis=1)
        indexes = distances.argsort()[:number_of_neighbors]
        labels = [self.feature_histogram[index][1] for index in indexes]
        return statistics.mode(labels)

    def test(self,testing_set, kneighbors):
        '''function to test entire training set against model

        parameters:
            -testing_set: testing dataset in form ((image, label), (image,label)...)
            -kneighbors: number of neighbors to consider in KNN search

        returns:
            accuracy of model as fraction of correct classifications'''
        number_of_samples = len(testing_set)
        progress = progressBar("Testing: ",number_of_samples, "Image")
        correct_classifications = 0
        for sample in testing_set:

            image = sample[0]
            true_label = sample[1]
            predicted_label = self.predict_label(image,kneighbors)
            if true_label == predicted_label:
                correct_classifications+= 1
            progress.increment()

        accuracy = float(correct_classifications) / number_of_samples
        return accuracy

def read_datasets():
    '''module reads all the datasets and returns them as 2d lists where
    each entry of main list is a list of [image data, label]

    Returns:
    -training_set: training dataset
    -validation_set: validation dataset
    -testing_set: testing dataset
    '''
    trainpath = 'TrainingSet/'
    validationpath = 'ValidationSet/'
    testpath = 'TestSet/'

    train = []
    validation = []
    test = []

    for label in os.listdir(trainpath):
        for image in os.listdir(trainpath + label):
            train.append((cv2.imread(trainpath + label +'/' + image),label))
    print("Training Set Loaded Successfully")

    for label in os.listdir(validationpath):
        for image in os.listdir(validationpath + label):
            validation.append((cv2.imread(validationpath + label +'/' + image),label))
    print("Validation Set Loaded Successfully")

    for label in os.listdir(testpath):
        for image in os.listdir(testpath + label):
            test.append((cv2.imread(testpath + label +'/' + image),label))
    print("Testing Set Loaded Successfully")

    return train,validation,test

