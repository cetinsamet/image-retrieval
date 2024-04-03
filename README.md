# TY Case Study

The proposed solution for the provided case study can be divided into three stages:

1 - Download all images of objects of the products that are provided within the JSON file.  
2 - Extract the features of all images using a pre-trained ResNet model and save all features into a vector database.  
3 - For a given query image, perform a similarity search by first extracting the features of the query image and then performing a search over the vector database.  

## Installation
1. Clone the repository to your local first as follows    
```    
$ git clone https://github.com/Trendyol-DataScienceHiring/SametCetin.git
```

2. Then recreate the virtual environment on your local using the provided ```.yml``` file and activate the venv as follows    
```
$ cd SametCetin  
$ conda env create -f environment.yml
$ conda activate trendyol-case
```

3. Add the path of your own working directory as WORK_DIR in the configuration file ```src/config/settings.py```as follows
```
$ cd src
$ nano config/settings.py  # or use your favorite text editor and add the path of your own working directory

WORK_DIR = "PATH-TO-YOUR-OWN-WORKING-DIRECTORY"
```

4. Create a ```data``` folder as follows
```
mkdir ../data
```

5. Download data from [here](https://cdn.dsmcdn.com/ty1075/ds/cases/DScase_computervision.zip) and move ```export.json``` under ```data``` folder.

## Run
Please perform following steps sequentially to reproduce the proposed solution.

### Downloading images
Running ```download_images.py``` downloads all the images of objects of the products using the provided URLs and saves them under an image folder ```data/images```.
The script uses the multiprocessing library to download the images in parallel (by default the number of processes are 4, please use ```--n_process``` flag to customize).
```    
$ python3 download_images.py
```

If you are on MacOS and getting the crash error ("**+[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug**"), please run the following command and try to rerun the download script.
```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

### Building vector database
Running ```build_vector_database.py``` extracts the features of all downloaded images by using the selected pre-trained ResNet model (please use ```--feat_extractor``` flag to customize) and save the features into a vector database that is stored under ```data/``` directory.

```    
$ python3 build_vector_database.py --feat_extractor resnet34
```

### Query search
Finally, running ```query_search.py``` selects random object images (the amount of the selected images can be customized using ```--n``` flag) and search the most similar object images (the amount of the similar images can be customized using ```--k``` flag) by querying the pre-built vector database. 
```    
$ python3 query_search.py --feat_extractor resnet34 --n 10 --k 12
```

Some examples of the resulting search queries are as follows;

![img.png](results/results_resnet34/query_001.jpg)
---
![img.png](results/results_resnet34/query_002.jpg)
---
![img.png](results/results_resnet34/query_003.jpg)


---
---
---
---
---
### OLD README BELOW
### You can download data from [here](https://cdn.dsmcdn.com/ty1075/ds/cases/DScase_computervision.zip)

Finding the similarity ratio of 2 products to each other, forms the basis of the projects like shop the look, recommendation systems, image search and many others. One of the methods that can be used to measure the similarity of products is 'image similarity'.

The attached json file contains approximately 20 thousand images and the bounding boxes of the fashion products in them. The task we kindly request from you is to list the most similar fashion products in this json to an incoming visual input.

You are expected to randomly pick 10 different cropped images from the json file and find 12 most similar images to the selected one.

#### Below is the expected output format:

Input:

![img.png](assets/img.png)

Output:

![img.png](assets/img2.png)
![img.png](assets/img3.png)

**PS:** Please include **"case is done"** in your last commit.

## Follow Us!

[Trendyol Tech Medium](https://medium.com/trendyol-tech)

[Trendyol Open Source GitHub](https://github.com/Trendyol)

[Trendyol Tech Youtube](https://www.youtube.com/channel/UCUBiayLMggBAsiYvGLzQJ5w)

[Trendyol Tech Kommunity](https://kommunity.com/@trendyol)

![image.png](./assets/image.png)

