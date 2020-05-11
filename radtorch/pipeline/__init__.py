from .image_classification import Image_Classification
from .feature_extraction import Feature_Extraction
from .compare_classifiers import Compare_Image_Classifiers




def load_pipeline(target_path):
    infile=open(target_path,'rb')
    pipeline=pickle.load(infile)
    infile.close()
    return pipeline
