import cv2, numpy as np

def assess_image(path):
    img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if img is None:return {'exists':False,'blur':None,'brightness':None,'contrast':None}
    return {'exists':True,'blur':float(cv2.Laplacian(img,cv2.CV_64F).var()),'brightness':float(img.mean()),'contrast':float(img.std())}
