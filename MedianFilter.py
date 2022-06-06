import numpy as np
import cv2

#########################
#   Harun Serkan Metin  #
#   201101034           #
#########################

FILTER_SIZE=5 #changable
SPACE=FILTER_SIZE//2


def median_filter(data, filter_size, W):

    data = cv2.copyMakeBorder(data,SPACE,SPACE,SPACE,SPACE, cv2.BORDER_REPLICATE) #replicate borders
    """
    3. parameter "W"
    if it is 0 this means not weighted
    if it is 1 this means gonna be weighted
    """
    temp = np.array(())
    indexer = filter_size // 2
    data_final = np.zeros((len(data),len(data[0])))
    
    for i in range(len(data)-filter_size+1):
        for j in range(len(data[0])-filter_size+1):

            temp = data[i:(i+FILTER_SIZE),j:(j+FILTER_SIZE)]
            temp=temp.flatten()

            if W==1:
                med=temp[len(temp) //2]
                temp=np.append(temp,[[med],[med]])
            
            temp=np.sort(temp)
            data_final[i+indexer][j+indexer] = temp[len(temp) // 2]
            temp = np.array(())

    x=len(data_final)
    y=len(data_final[0])

    data_final = data_final[SPACE:(x-SPACE), SPACE:(y-SPACE)]      #cutting replicated borders
    
    data_final = np.reshape(data_final, (x-filter_size+1, y-filter_size+1))
    return data_final.astype(np.uint8)
##########################################################################
#1. Question

NoisyImage=cv2.imread("noisyImage.jpg",0)
removed_noise = median_filter(NoisyImage, FILTER_SIZE,0) 

openCvMedian = cv2.medianBlur(NoisyImage, FILTER_SIZE)
ex=abs(openCvMedian-removed_noise)

print("\nNot Matched Pixel Count: ",np.count_nonzero(ex))

cv2.imshow("My Median",removed_noise)
cv2.imshow("Extarct",ex)
cv2.imshow("Noisy Image",NoisyImage)

cv2.imwrite("NoisyImageFixedWithMedian.jpg",removed_noise)

##################################################################################
#2. Question

org=cv2.imread("original.jpg",0)

gaussianImage=cv2.GaussianBlur(NoisyImage,(7,7),0)
boxImage=cv2.blur(NoisyImage, (5,5))


print(
    "\n1. Mine median filter PSNR: ",cv2.PSNR(org,removed_noise)
    ,"\n2. OpenCV’s box filter PSNR: ",cv2.PSNR(org,boxImage)
    ,"\n3. OpenCV’s Gaussian filter PSNR: ",cv2.PSNR(org,gaussianImage)
    ,"\n4. OpenCV’s median filter PSNR: ",cv2.PSNR(org,openCvMedian))


##################################################################################
#3. Question

weighted=median_filter(NoisyImage,FILTER_SIZE,1) #3. parameter is 1 this means it is gonna be weighteed
cv2.imshow("Weighted",weighted)

print("5. Mine median filter center weighted PSNR: ",cv2.PSNR(org,weighted))
print()
cv2.imwrite("NoisyImageFixedWithCenteredMedian.jpg",weighted)
cv2.waitKey(0)