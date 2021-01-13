from pixelstocircles import *
path ='/home/helen/Documents/HelenExperimentalCode/Image_generation_MULTEM_largetry/predictions/Brookespredictionseverythingtogetherfolder/vacancy_mask/Cycled_004_Hour_00_Minute_00_Second_41_Frame_0003.png'
img_dir= '/home/helen/Documents/HelenExperimentalCode/experimentaldataextraction/linearsinglelinedefects/layers/'


def connected_component_label(img):
    '''
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    '''
    # Applying cv2.connectedComponents() 
    connectivity=1
    num_labels, labels = cv2.connectedComponents(img, connectivity)
    print(num_labels)
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    cv2.imwrite( img_dir+'label_hue.png',labeled_img)
    
 # Visualisation Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()
    
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()
    

    #trying component separation
    '''
    gray = img
    t, thresh = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)

    connectivity = 8 
    num_labels, label= cv2.connectedComponents(thresh, connectivity)
    label = label + 1

    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

    h, w = thresh.shape[:2]
    for index in range(1, num_labels):
        mask = np.zeros((h, w), np.uint8)
        mask[label == index + 1] = 1
        obj = thresh * mask[:, :, np.newaxis]
        obj = obj[..., : : -1]

        #plt.subplot(1, 147, index)
        plt.imshow(obj)
        plt.xticks([]), plt.yticks([])
        plt.title(index)
        plt.show()
   # plt.savefig('layers_separated.png')
    '''
   
connected_component_label(img)

