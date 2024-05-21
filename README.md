# Final_Project_PIM

The first part of the project focuses on segmenting a lengthy medical image that contains multiple anatomical structures, specifically the liver, tumor, vessels, and abdominal aorta. The provided segmentation image is a concatenated sequence where each anatomical structure occupies a distinct portion of the image. The main objective is to separate this long segmentation image into four individual segments, each corresponding to one of the anatomical structures, and then create a new consolidated image. This new image will facilitate easier analysis and visualization of each anatomical structure separately, aiding in medical research and diagnostics.

The second part includes all the code necessary to perform a 3d rigid coregistration between 2 images, implemented with quaternions and including code to load the DICOM files and visualizing the results all the way through
