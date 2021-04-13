import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import copy 
import tensorflow as tf
from IPython.display import clear_output
import skimage.io as io


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
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco


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

          