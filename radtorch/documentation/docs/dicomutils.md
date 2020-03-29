# DICOM Module <small> radtorch.dicomutils </small>

!!! bug " Documentation Outdated. Please check again later for update."



Tools and Functions for DICOM images handling and extraction of pixel information.

      from radtorch import dicomutils

## window_dicom

      dicomutils.window_dicom(filepath, level, width)

!!! quote ""
      Converts DICOM image to numpy array with certain width and level.

      **Arguments**

      - filepath: _(str)_ input DICOM image path.

      - level: _(int)_ target window level.

      - width: _(int)_ target window width.

      **Output**

      - Output: _(array)_ windowed image as numpy array.


## dicom_to_narray

      dicomutils.dicom_to_narray(filepath, mode='RAW', wl=None)

!!! quote ""
      Converts DICOM image to a numpy array with target changes as below.

      **Arguments**

      - filepath: _(str)_ input DICOM image path.

      - mode: _(str)_ output mode. (default='RAW')
          - Options:

              - 'RAW' = Raw pixels,

              - 'HU' = Image converted to Hounsefield Units.

              - 'WIN' = 'window' image windowed to certain W and L,

              - 'MWIN' = 'multi-window' converts image to 3 windowed images of different W and L (specified in wl argument) stacked together].

      - wl: _(list)_ list of lists of combinations of window level and widths to be used with WIN and MWIN. (default=None)
          In the form of : [[Level,Width], [Level,Width],...].
          Only 3 combinations are allowed for MWIN (for now).

      **Output**

      - Output: _(array)_ array of same shape as input DICOM image with 1 channel. In case of MWIN mode, output has same size by 3 channels.


## dicom_to_pil

      dicomutils.dicom_to_pil(filepath)

!!! quote ""
      Converts DICOM image to PIL image object.

      **Arguments**

      - filepath: _(str)_ input DICOM image path.


      **Output**

      - Output: _(pillow image object)_.
