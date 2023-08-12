# REVERSE_IMAGE_SEARCH_IN_FASHION_PRODUCTS
 Search for similar or related images from the trained data based on an input image. 

## HOW TO RUN THE CODES
1)Run feature_extraction.ipynb code first. The data set used in this code is from kaggle(link...https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).
2)Then download the pickle file for the feature extraction.
3)Next run the "reverse_image_search.ipynb"  by replacing the pickle file and test images with your own data set if you want.






## Detailed overview of the project


(A)In feature _extraction.ipynb 


1. **Import Libraries**: The code begins by importing the required libraries, including `seaborn`, `tensorflow`, `keras`, `cv2` (OpenCV), `numpy`, `matplotlib.pyplot`, and `os`.

2. **Create ResNet50 Model**: A ResNet50 model is loaded from Keras's `tensorflow.keras.applications.resnet50` module. The model is set to be non-trainable by freezing its layers. Then, a `Sequential` model is created to wrap the pre-trained ResNet50 model, followed by a `GlobalMaxPooling2D` layer. This new sequential model will be used for feature extraction.

3. **Load and Process Image**: An image (`1636.jpg`) is read using OpenCV, converted to a NumPy array, resized to a target size of (224, 224), and then preprocessed using the `preprocess_input` function to prepare it for the ResNet50 model.

4. **Feature Extraction**: The preprocessed image is passed through the ResNet50-based model to extract features. The resulting feature vector is flattened, normalized using the L2 norm, and stored in the `result_norm` variable.

5. **Define Feature Extraction Function**: A function called `feature_extraction` is defined to encapsulate the feature extraction process. This function takes an image path and a model as input, performs the necessary preprocessing, passes the image through the model, and returns the normalized feature vector.

6. **Extract Features from Multiple Images**: The code then loops through a directory containing images (`images_with_product_ids`), extracts features from each image using the `feature_extraction` function, and appends the normalized feature vectors to the `feature_list`. The filenames of the images are also stored in the `filename` list.

7. **Save Extracted Features**: Finally, the extracted feature vectors and corresponding filenames are saved using the `pickle` module. The `feature_list` is saved as `feature_list.pkl`, and the `filename` list is saved as `filename_list.pkl`.


**(B) In reverse_image_serach.ipynb**


1. **Importing Libraries**: The code starts by importing various libraries, including `seaborn`, `tensorflow`, `keras`, `os`, `cv2_imshow` (from Google Colab patches), `zipfile`, `Image` (from PIL), `numpy`, `pickle`, and `streamlit`.

2. **Extracting ZIP File**: The code extracts a ZIP file named `images_with_product_ids.zip` using the `zipfile` module. The contents of this file are extracted to a specific directory.

3. **Creating ResNet50 Model**: Similar to your previous code, a ResNet50 model is loaded, configured to be non-trainable, and wrapped with a `GlobalMaxPooling2D` layer in a `Sequential` model.

4. **Loading and Preprocessing Test Image**: A test image is loaded from a specified path, resized, and preprocessed using the `preprocess_input` function.

5. **Feature Extraction Function**: The `feature_extraction` function is defined, which takes an image path and model as input, extracts features from the image using the model, normalizes them, and returns the normalized feature vector.

6. **Extracting Features from Files**: Features are extracted from all files in the specified directory (`/content/images_with_product_ids`) using the `feature_extraction` function. The extracted features and corresponding filenames are stored.

7. **Loading Extracted Features and Filenames**: The extracted features and filenames are loaded from previously saved pickle files.

8. **Nearest Neighbors Model**: A Nearest Neighbors model (`model_knn`) is created using the extracted feature list.

9. **Testing and Displaying Similar Images**: A test image is provided, and its features are extracted. The nearest neighbors of the test image are found using the Nearest Neighbors model. The original test image and its nearest neighbor images are displayed.

10. **Streamlit Web App**: The code transitions to creating a Streamlit web app using the `streamlit` library. The app displays an interface where users can upload an image. When an image is uploaded, its features are extracted, and the system recommends similar fashion images based on the uploaded image.

11. **Uploading and Processing Uploaded Image**: If a user uploads an image through the app, the image is displayed, and its features are extracted.

12. **Recommendation**: The system then recommends similar images based on the extracted features using the previously created Nearest Neighbors model.

13. **Displaying Recommended Images**: The top recommended images are displayed in columns using Streamlit's layout components.






