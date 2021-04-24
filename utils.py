import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import copy 
import tensorflow as tf
from IPython.display import clear_output
import skimage.io as io
import rectification as rec
import os
import pathlib
import traceback
import os
from time import time
from datetime import datetime
import keras


import json
from keras.preprocessing.image import save_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import autokeras as ak


class RuntimeLogger:
    def __init__(self, log_dir: str, model: str, epoch: int):
        self.runtime_filepath = os.path.join(log_dir, str(model)+".json")
        self.runtime_dict = {}
        self.runtime_dict.update({'model_name': model})
        self.runtime_dict.update({'epoch': epoch})
        self.global_start_time = None
        
    def start(self):
        self.global_start_time = time()
        
    def log_task_end(self, task_name: str, task_start_time: float):
        task_runtime = time() - task_start_time
        self.runtime_dict.update({task_name: task_runtime})
        
    def add_metric(self, name: str, value: float):
        self.runtime_dict.update({name: float(value)})
        
    def log_experiment_end(self):
        self.log_task_end('global_runtime', self.global_start_time)
        json.dump(self.runtime_dict, 
                  open(self.runtime_filepath, 'w'), indent=4)


def unnormalizeAnns(anns, x, y):
    """[summary]

    Args:
        anns ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    for i in range(0, len(anns['segmentation'][0])):
        if anns['segmentation'][0][i] < 1 :
            if i % 2 == 0:
                anns['segmentation'][0][i] = anns['segmentation'][0][i] * x
            else:
                anns['segmentation'][0][i] = anns['segmentation'][0][i] * y
    return anns


def filterDataset(folder, mode='train'):
    """[summary]

    Args:
        folder ([type]): [description]
        mode (str, optional): [description]. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    annFile = '{}/annotations/instances_{}.json'.format(folder, mode)
    coco = COCO(annFile)

    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)

    random.seed(123)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco


def filterDatasetDeepSolarEye(folder):
    """[summary]

    Args:
        folder ([type]): [description]
        mode (str, optional): [description]. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    unique_images = []
    for file in onlyfiles:
        file_obj = {'file_name' : "//" + file}
        unique_images.append(file_obj)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, None

def getClassName(classID, cats):
    """[summary]

    Args:
        classID ([type]): [description]
        cats ([type]): [description]

    Returns:
        [type]: [description]
    """
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    """[summary]

    Args:
        imageObj ([type]): [description]
        img_folder ([type]): [description]
        input_image_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'].split('/')[2])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img


def getBinaryMask(imageObj, coco, catIds, input_image_size, x, y):
    """[summary]

    Args:
        imageObj ([type]): [description]
        coco ([type]): [description]
        catIds ([type]): [description]
        input_image_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    
    #print(imageObj)
    
    coco.loadImgs(imageObj['id'])[0]['width'] = x
    coco.loadImgs(imageObj['id'])[0]['height'] = y
    for a in range(len(anns)):
        ann = copy.deepcopy(anns[a])
        new_mask = cv2.resize(coco.annToMask(unnormalizeAnns(ann,x,y)), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, coco, folder, 
                      input_image_size=(128,128), batch_size=4, mode='train'):
    """[summary]

    Args:
        images ([type]): [description]
        coco ([type]): [description]
        folder ([type]): [description]
        input_image_size (tuple, optional): [description]. Defaults to (128,128).
        batch_size (int, optional): [description]. Defaults to 4.
        mode (str, optional): [description]. Defaults to 'train'.

    Yields:
        [type]: [description]
    """
    
    img_folder = '{}/'.format(folder)
    dataset_size = len(images)
    catIds = coco.getCatIds()
    x = input_image_size[0]
    y = input_image_size[1]

    random.shuffle(images)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, x, y, 3)).astype('float')
        mask = np.zeros((batch_size, x, y, 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            train_mask = []
            #print(i)
            
            imageObj = images[i]
            #print(imageObj)
            
            #print('images loaded')
            try :
                imageObj['width'] = x
                imageObj['height'] = y
            except :
                #print(imageObj)
                print("not able to set the images size")
            
            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)
            
            train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size, x, y)
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        #print("end loop")
        c+=batch_size
        #print(c)
        if(c + batch_size >= dataset_size):
            c=0
            random.shuffle(images)
        yield img, mask


def display(display_list):
    """[summary]

    Args:
        display_list ([type]): [description]
    """
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def recification_normalisation(img, mask):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(mask,200,255,cv2.THRESH_BINARY_INV)
    
    
    contours, hierarchy = cv2.findContours(thresh,  
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    threshold = 0.1
    
    total_area = 0
    bigest_cnt = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        total_area += area
    total_area

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4:
            if area > total_area * threshold:
                bigest_cnt.append(cnt)
    
    rec_images = []
    for cnt in bigest_cnt:
                
        hull = cv2.convexHull(cnt)

        black = np.zeros_like(img1)
        black2 = black.copy()

        cv2.drawContours(black2, [hull], -5, (255, 255, 255), -1)
        cv2.drawContours(black2, [hull], -5, (255, 255, 255), 2)

        masked = cv2.bitwise_and(img, img, mask = black2) 

        panel_crop = cv2.resize(masked,(192,192))
        edgelets1 = rec.compute_edgelets(panel_crop)

        vp1 = rec.ransac_vanishing_point(edgelets1, num_ransac_iter=2000, 
                                 threshold_inlier=5)
        vp1 = rec.reestimate_model(vp1, edgelets1, threshold_reestimate=5)

        edgelets2 = rec.remove_inliers(vp1, edgelets1, 10)
        vp2 = rec.ransac_vanishing_point(edgelets2, num_ransac_iter=2000,
                                 threshold_inlier=5)
        vp2 = rec.reestimate_model(vp2, edgelets2, threshold_reestimate=5)

        warped_img = rec.compute_homography_and_warp(panel_crop, vp1, vp2)

        rec_img = cv2.resize(crop(warped_img), (192,192))
        rec_images.append(rec_img)
                
    return rec_images


def predict_and_store(model, mode, dataset, image_gen, data_size, folder, x, y):
    input_image_size = (x,y)

    folder_out = 'data/{}/{}'.format(mode, dataset)
    folder_out_mask = 'data/{}/{}_mask'.format(mode, dataset)
    folder_out_rec = 'data/{}/{}_rec'.format(mode, dataset)
    
    pathlib.Path(folder_out).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(folder_out_mask).mkdir(parents=True, exist_ok=True)
    pathlib.Path(folder_out_rec).mkdir(parents=True, exist_ok=True)
    
    img = np.zeros((1, x, y, 3)).astype('float')
    for i in range(0,data_size):
        filename = image_gen[i]['file_name'].split('/')[2]
        outfile = folder_out + "/" + filename
        outfile_mask = folder_out_mask + "/" + filename
        try:
            image = getImage(image_gen[i], folder, input_image_size)
            img[0] = image

            pred_mask = model.predict(img)

            save_img(outfile, image)
            save_img(outfile_mask, pred_mask[0]) 

        except:
            print("not possible to predict the image")
            traceback.print_exc()
            continue
        
        image = plt.imread(outfile)
        mask = plt.imread(outfile_mask)
        

        try:
            rec_images = recification_normalisation(image, mask)
            i = 0
            for rec_img in rec_images:
                outfile_rec = folder_out_rec + "/" + str(i) + filename
                save_img(outfile_rec, rec_img)
                i += 1
        except:
            print("recification for image not possible")

def calc_predictions(model, dataset=None, num=2):
    dice = []
    jaccard = []
    if dataset:
        for i in range(0, num):
            image, mask = next(dataset)
            pred_mask = model.predict(image)
            dice.append(dice_coef(sample_mask[0], pred_mask[0]))
            jaccard.append(jaccard_distance(sample_mask[0], pred_mask[0]))
            #display([image[0], mask[0], pred_mask[0]])
    print("Dice", np.mean(dice))
    print("Jaccard", np.mean(jaccard))
    
    return dice, jaccard


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)  

def train_keras(model_name, folder, nr_images_batch, nr_batch, nr_epoch, max_trials=10):
    
    image_width = 192
    image_height = 192
    ratio = 1
    channels = 3
    

        
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    allfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    for i in range(0, nr_batch):
        print("--- Batch {} --- ".format(str(i)))
            # Initialize the image regressor.
        reg = ak.ImageRegressor(
            project_name=model_name,
            metrics=[tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.RootMeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError()],
            overwrite=False,
            max_trials=max_trials)
    
        onlyfiles = allfiles[i*nr_images_batch:(i+1)*nr_images_batch]
        print("select from to", i*nr_images_batch, (i+1)*nr_images_batch)
    
        train_files = []
        y_train = np.ndarray(len(onlyfiles))
   
    
        for _file in onlyfiles:
            train_files.append(_file)
            y_train[i] = (np.float64(_file.split('_')[11]))    
        print("Files in train_files: %d" % len(train_files))
    
        image_width = int(image_width / ratio)
        image_height = int(image_height / ratio)

        dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                         dtype=np.float32)

    
        j=0
        for _file in train_files:
            img = load_img(folder + "/" + _file)  # this is a PIL image
            #img.thumbnail((image_width, image_height))
            # Convert to Numpy Array
            x = img_to_array(img)  
            #x = x.reshape((3, 48, 48))
            # Normalize
            #x = (x - 128.0) / 128.0
            dataset[i] = x
            j += 1
            if j % 25 == 0:
                print("%d images to array" % j)
        print("All images to array!")

        #Splitting 
        X_train = dataset
        #X_train, X_val, y_train, y_test = train_test_split(dataset, y_train, test_size=0.0, random_state=33)
        #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=33)
        #print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))

        # Feed the image regressor with training data.
        if i + 1 == nr_batch:
            print("Last run, eval the model")
            model = reg.evaluate(dataset, y_train)
            return model
        else:
            reg.fit(X_train, 
                y_train, 
                batch_size=25,
                validation_split=0.2,
                callbacks=[tensorboard_callback],
                epochs=nr_epoch)
        