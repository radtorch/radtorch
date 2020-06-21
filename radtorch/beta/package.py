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


from radtorch.settings import *
from radtorch import pipeline, core, utils
from radtorch.utils import *


import streamlit as st

class Image_Classification_UI():
    def __init__(self, pipeline, title='Image Classification', **kwargs):
        self.title=title
        self.pipeline=pipeline
        st.markdown('<h1 style="display:inline">'+self.title+' '+'</h1><span style="display:inline"><small>   Created with RADTorch and Streamlit</small></span>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        self.target_image = st.file_uploader(label='Please select target image:', encoding='auto', )
        predictions = self.pipeline.classifier.predict()







Image_Classification_UI(pipeline=None)
