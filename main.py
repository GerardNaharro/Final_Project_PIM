import cv2
from matplotlib import pyplot as plt, animation
import matplotlib
import numpy as np
import pydicom
import glob
import scipy
import os
from skimage import measure

def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    # EN VERDAD ES SAGITAL
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    # EN VERDAD ES AXIAL
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)

def MIP_axial_plane(img_dcm: np.ndarray) -> np.ndarray:
    # EN VERDAD ES CORONAL
    """ Compute the maximum intensity projection on the axial orientation. """
    return np.max(img_dcm, axis=0)

def rotate_on_coronal_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    # EN VERDAD ES SAGITAL
    """Rotate the image on the coronal plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(0, 2), reshape=False)


def rotate_on_sagittal_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    # EN VERDAD ES AXIAL
    """Rotate the image on the sagittal plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(0, 1), reshape=False)

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    # EN VERDAD ES CORONAL
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    ind = np.where(mask == 1)
    return np.array([np.mean(ind[0]), np.mean(ind[1]), np.mean(ind[2])])

def apply_segmentation_mask(img: np.ndarray,mask: np.ndarray,):
    """ Apply the segmentation mask with alpha fusion. """

    cmap = plt.get_cmap("bone")
    img_cmapped = cmap(img)
    cmap = plt.get_cmap("copper")
    mask_cmapped = cmap(mask)
    mask_cmapped = mask_cmapped * mask[..., np.newaxis]


    return img_cmapped, mask_cmapped


def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    # Your code here: See `matplotlib.colormaps[...]`.
    # ...
    cm = plt.colormaps[cmap_name]
    return cm(img)


def min_max_normalization(img):
    min_val, max_val = np.min(img), np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


if __name__ == "__main__":
    segmentation_path = "/Users/gerardnaharrolopez/Desktop/datos_practica_PIM/manifest-1713863870274/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/300.000000-Segmentation-24189/1-1.dcm"
    # Looking at 3d slicer we can know that the corresponding CT image of the segmentation image
    # is the acquisition number 1 image
    ct_path = "/Users/gerardnaharrolopez/Desktop/datos_practica_PIM/manifest-1713863870274/HCC-TACE-Seg/HCC_010/02-21-1998-NA-AP LIVER-61733/4.000000-Recon 2 LIVER 3 PHASE AP-37279"
    # load the DICOM files
    raw_ct_files = []
    # Use glob to get all the files in the folder
    files = glob.glob(ct_path + '/*')
    for fname in files:
        print("loading: {}".format(fname))
        raw_ct_files.append(pydicom.dcmread(fname))
    print("CT slices loaded! \n")

    dcm_segmentation = pydicom.read_file(segmentation_path)
    img_segmentation = dcm_segmentation.pixel_array

    #print(img_segmentation)
    print(img_segmentation.shape)


    print(dcm_segmentation)
    [print(f'{dicom_tag}: {dicom_value}') for dicom_tag, dicom_value in dcm_segmentation.items()]


    print("--------------")
    print("Starting the cleaning of raw ct slices to get only the corresponding acquisition of the segmentation image...")
    ct_files = []
    for sl in raw_ct_files:
        acq_num = sl.AcquisitionNumber
        #print("Acquisition Number:", acq_num)
        if acq_num == 1:
            ct_files.append(sl)

    print("Cleaning done!")
    print("-------------")

    print("Actual ct slices:")
    for sl in ct_files:
        acq_num = sl.AcquisitionNumber
        print("Acquisition Number:", acq_num)

    # Sort according to ImagePositionPatient header, based on the 3rd component (which is equal to "SliceLocation")
    print("Sorting slices...")
    ct_files = sorted(ct_files, key=lambda ds: float(ds.ImagePositionPatient[2]))
    print("Slices sorted!")

    #print(ct_files)

    # pixel aspects, assuming all slices are the same
    ps = ct_files[0].PixelSpacing
    ss = ct_files[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # create 3D array
    img_shape = list(ct_files[0].pixel_array.shape)
    img_shape.append(len(ct_files))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(ct_files):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    print(img3d.shape)
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, img_shape[2] // 2])
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, img_shape[1] // 2, :])
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[img_shape[0] // 2, :, :].T)
    a3.set_aspect(cor_aspect)

    plt.show()

    print(img3d.shape)
    print(img_segmentation.shape)

    print("transpose img3d")

    img3d_transposed = np.transpose(img3d, (2, 0, 1))

    print(img3d_transposed.shape)
    print(img_segmentation.shape)



    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(MIP_axial_plane(img_segmentation))
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(MIP_sagittal_plane(img_segmentation).T)
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(MIP_coronal_plane(img_segmentation))
    a3.set_aspect(cor_aspect)

    plt.show()


    #Create an animation (e.g. gif file) with a rotating Maximum Intensity Projection on the coronal-sagittal planes.
    print("Creating animations...")
    pixel_len_mm = [2.5, 0.7, 0.7]  # Pixel length in mm [z, y, x], same values that we have in ps and ss
    alpha_value = 0.25 # Set the alpha value for drawing the segmentation mask over the projections
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    img_min = np.amin(img3d)
    img_max = np.amax(img3d)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)


    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):

        rotated_img = rotate_on_sagittal_plane(img3d, alpha)
        projection = MIP_coronal_plane(rotated_img)

        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=sag_aspect)
        plt.savefig(f'results/MIP/ProjectionSagittal_{idx}.png')  # Save animation
        print(f'Sagittal projection {idx} created!')
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=sag_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/AnimationSagittal.gif')  # Save animation
    print("Sagittal animation created!")

    
    #   Create projections
    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        #print(alpha)
        rotated_img = rotate_on_sagittal_plane(img3d, alpha)
        projection = MIP_axial_plane(rotated_img)
        projection = np.flip(projection, 1)
        plt.imshow(projection.T, cmap=cm, vmin=img_min, vmax=img_max, aspect=cor_aspect)
        plt.savefig(f'results/MIP/ProjectionCoronal_{idx}.png')  # Save animation
        print(f'Coronal projection {idx} created!')
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img.T, animated=True, cmap=cm, vmin=img_min, vmax=img_max,
                    aspect=cor_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                         interval=250, blit=True)
    anim.save('results/MIP/AnimationCoronal.gif')  # Save animation
    print("Coronal animation created!")
    

    cm = matplotlib.colormaps['copper']
    img_min = np.amin(img_segmentation)
    img_max = np.amax(img_segmentation)
    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated_img = rotate_on_axial_plane(img_segmentation, alpha)
        projection = MIP_sagittal_plane(rotated_img)

        plt.imshow(projection.T, cmap=cm, vmin=img_min, vmax=img_max, aspect=sag_aspect)
        plt.savefig(f'results/MIP/ProjectionSegmentationSagittal_{idx}.png')  # Save animation
        print(f'Sagittal Segmentation projection {idx} created!')
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img.T, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=sag_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/AnimationSegmentationSagittal.gif')  # Save animation
    print("Sagittal Segmentation animation created!")

    #   Create projections
    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        # print(alpha)
        rotated_img = rotate_on_axial_plane(img_segmentation, alpha)
        projection = MIP_coronal_plane(rotated_img)
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=cor_aspect)
        plt.savefig(f'results/MIP/ProjectionSegmentationCoronal_{idx}.png')  # Save animation
        print(f'Coronal Segmentation projection {idx} created!')
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max,
                    aspect=cor_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/AnimationSegmentationCoronal.gif')  # Save animation
    print("Coronal Segmentation animation created!")

    all_segments = np.zeros((77,512,512))
    for i in range(4):
        all_segments = np.where(img_segmentation[i * 77:(i+1) * 77,:,:] > 0, i + 1, all_segments)


    img_min = np.amin(img3d)
    img_max = np.amax(img3d)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    n = 16
    projections = []
    img3d = min_max_normalization(img3d)
    img3d[img3d < 0.3] = 0

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):

        rotated_img = rotate_on_sagittal_plane(img3d, alpha)
        rotated_seg = rotate_on_axial_plane(all_segments, alpha)
        projection = MIP_coronal_plane(rotated_img)
        projection_seg = MIP_sagittal_plane(rotated_seg)

        img_cmapped, mask_cmapped = apply_segmentation_mask(projection,projection_seg)
        mask_cmapped = np.transpose(mask_cmapped, (1, 0, 2))
        projection = ((img_cmapped * (1 - 0.25)) + (mask_cmapped * 0.25))
        plt.clf()
        plt.imshow(projection, aspect=sag_aspect)
        plt.savefig(f'results/MIP/ProjectionSagittal_SEGMENTATED_{idx}.png')  # Save animation
        print(f'Sagittal projection {idx} created!')
        projections.append(projection)  # Save for later animation
        plt.clf()
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, vmin=np.amin(img3d), vmax=np.amax(img3d), aspect=sag_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/AnimationSagittal_SEGMENTATED.gif')  # Save animation
    print("Sagittal animation created!")

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated_img = rotate_on_sagittal_plane(img3d, alpha)
        rotated_seg = rotate_on_axial_plane(all_segments, alpha)
        projection = MIP_axial_plane(rotated_img)
        projection_seg = MIP_coronal_plane(rotated_seg)

        img_cmapped, mask_cmapped = apply_segmentation_mask(projection.T, projection_seg)
        projection = ((img_cmapped * (1 - 0.25)) + (mask_cmapped * 0.25))
        plt.clf()
        projection = np.flip(projection, 0)
        plt.imshow(projection, aspect=cor_aspect)
        plt.savefig(f'results/MIP/ProjectionCoronal_SEGMENTATED_{idx}.png')  # Save animation
        print(f'Coronal projection {idx} created!')
        projections.append(projection)  # Save for later animation
        plt.clf()
        # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, vmin=np.amin(img3d), vmax=np.amax(img3d), aspect=cor_aspect)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/AnimationCoronal_SEGMENTATED.gif')  # Save animation
    print("Coronal animation created!")



