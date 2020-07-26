from .image_classification import Image_Classification
from .feature_extraction import Feature_Extraction
from .hybrid_image_classification import Hybrid_Image_Classification
# from .compare_classifiers import Compare_Image_Classifiers
from .gan import GAN
from ..settings import *

## Code Last Updated/Checked: 08/01/2020

def load_pipeline(target_path):
    infile=open(target_path,'rb')
    pipeline=pickle.load(infile)
    infile.close()
    return pipeline
