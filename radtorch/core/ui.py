# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

ui_framework = '''
from radtorch.settings import *
from radtorch import pipeline, core, utils
from radtorch.utils import *
import radtorch, sys, io
import streamlit as st

def load_saved_pipeline(target_path):
    infile=open(target_path,'rb')
    pipeline=pickle.load(infile)
    infile.close()
    return pipeline

class Image_Classification_UI():
    def __init__(self, pipeline, title='Image Classification', **kwargs):
        self.title=title
        if isinstance(pipeline, str):
            self.pipeline = load_saved_pipeline(pipeline)
        else:
            self.pipeline = pipeline
        st.markdown('<h1 style="display:inline">'+self.title+'  '+'</h1><span style="display:inline"><small>   Created with RADTorch and Streamlit</small></span>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        self.target_image = st.file_uploader(label='Please select target image:', encoding='auto', )
        self.run = st.button('Show Prediction')
        if self.run:
            self.run_prediction()

    def run_prediction(self):
        if self.target_image is not None:
            with open("temp.rt", "wb") as f:
                f.write(self.target_image.read())
        with st.spinner('Running Image Analysis...'):
          predictions = self.pipeline.classifier.predict(input_image_path='temp.rt',all_predictions=True)
        st.image(Image.open('temp.rt'))
        st.write(predictions)
        st.success('Done!')



if __name__ == "__main__":
  opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
  args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

  if args[0] == 'image_classification':
    output = Image_Classification_UI(pipeline=args[1])
'''
