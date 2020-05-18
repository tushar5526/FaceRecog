# FaceRecog
## Face Recogntion project based on VGG 16 for MLops task

### Project Structure 

***faceData***

 - **train**
   - n1
     - (training images for n1)
   - n2
     - (training images for n2)
     
 - **test**
   - n1
     - (test images for n1)
   - n2
     - (test images for n2)
   
 
You can collect and label your own data by using ***click_your_photo.py***, this script collects 100 photos of your by default and saves it in a folder named FaceData, the images names are by default stored as ***1_(image_count).jpg***
 
First project of mine with tensorflow on Docker container, it is a face recoginition model build using keras on TF backend
The results can be tested using Pillow as I am having issues with running OpenCV on container, but will get on it in future 
