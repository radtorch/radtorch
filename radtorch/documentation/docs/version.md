# Releases/Versions

<small>


## Nightly


## Official Releases
### [v.0.1.2-beta](https://github.com/radtorch/radtorch/releases)
    - Projected Release Date: 3-15-2020
    - Expected Fixes
      - Restructure how dataset information is processed.
        Internally a dataframe input_data is created to expedite calling dataset_info.

    - Expected New Features
      - Allow user to set/change pytorch and numpy random seed.
      - Misclassified items during testing in the image classification pipeline can now be viewed.
      - Figure size of traning metrics is now changable.
      - Multilabel classification added to Image Classification Pipeline.

### [v.0.1.1-beta](https://github.com/radtorch/radtorch/releases)

    -  Release Date: 3-3-2020
    -  Feature Extraction Pipeline now uses batches not single images into GPU = improved speed.
    -  Inference done during confusion matrix and roc creation for Image Classification Pipeline now uses batches not single images into GPU = improved speed.
    -  Fix error with Image Classification Pipeline when using external test dataset.
    -  Allow omitting creating an internal test subset for the Image Classification Pipeline by setting test_percent = 0.
    -  Change line color in roc to default blue.
    -  Allow export and import of pipeline structures.
    -  Add shuffle parameter to Feature Extraction Pipeline.
    -  Restructure exported csv file with feature extraction pipeline.
    -  Updated documentation.


### [v.0.1.0-beta](https://github.com/radtorch/radtorch/releases)
    - Release Date: 3-1-2020
    - First BETA release.

</small>
