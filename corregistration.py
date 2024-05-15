import math

import scipy
from scipy.optimize import minimize
from matplotlib import pyplot as plt, animation
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
import pydicom
import glob
import cv2
import quaternion
from scipy.optimize import least_squares
from scipy import ndimage
import os

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
def min_max_normalization(img):
    min_val, max_val = np.min(img), np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    # EN VERDAD ES AXIAL
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)

def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MSE between two images. """
    # Your code here:
    #   ...
    return np.mean((img_input - img_reference) ** 2)

def rescale_image(input_img, reference_img):
    """
    Reescala la imagen de input para que tenga el mismo factor de escala que la imagen de referencia.

    Args:
        input_img (numpy.ndarray): Imagen de entrada.
        reference_img: Imagen de referencia con información de escala en el header.

    Returns:
        numpy.ndarray: Imagen de input reescalada.
    """
    intercept_ref = reference_img.RescaleIntercept
    slope_ref = reference_img.RescaleSlope

    rescaled_img = input_img * slope_ref + intercept_ref

    return rescaled_img

def resize_images_to_same_scale(pixel_spacing1, image2, pixel_spacing2):

    # Calcula el factor de escala para redimensionar la segunda imagen
    scale_factor_x = pixel_spacing1[1] / pixel_spacing2[1]
    scale_factor_y = pixel_spacing1[0] / pixel_spacing2[0]

    # Redimensiona la segunda imagen utilizando cv2.resize
    resized_image2 = cv2.resize(image2, (int(image2.shape[0] * scale_factor_x), int(image2.shape[1] * scale_factor_y)))

    return resized_image2

def apply_segmentation_mask(img: np.ndarray,mask: np.ndarray,):
    """ Apply the segmentation mask with alpha fusion. """

    cmap = plt.get_cmap("bone")
    img_cmapped = cmap(img)
    cmap = plt.get_cmap("autumn")
    mask_cmapped = cmap(mask)
    mask_cmapped = mask_cmapped * mask[..., np.newaxis]


    return img_cmapped, mask_cmapped

def animation_alpha_fusion(img1, img2, rotation, proj, name):

    fig, ax = plt.subplots()
    n = 16
    projections = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated_img = rotation(img1, alpha)
        rotated_seg = rotation(img2, alpha)
        projection = proj(rotated_img)
        projection_seg = proj(rotated_seg)

        img_cmapped, mask_cmapped = apply_segmentation_mask(projection, projection_seg)

        projection = ((img_cmapped * (1 - 0.25)) + (mask_cmapped * 0.25))
        plt.clf()
        plt.imshow(projection)
        plt.savefig('results/MIP/' + name + f'_{idx}.png')  # Save animation
        print(f'projection {idx} created!')
        projections.append(projection)  # Save for later animation
        plt.clf()
        # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True)]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=250, blit=True)
    anim.save('results/MIP/Animation_' + name + '_.gif')  # Save animation
    print("Animation created!")


# Rigid corregistration and quaternions functions

def translation(
        points: np.ndarray,
        translation_vector: np.ndarray
        ) -> np.ndarray:
    """ Perform translation of points """

    result = points + translation_vector
    return result


def axial_rotation(
        points: np.ndarray,
        angle_in_rads: float,
        axis_of_rotation: np.ndarray) ->np.ndarray:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """

    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    axis_of_rotation = axis_of_rotation * sin


    Quaternion_q = quaternion.as_quat_array(np.insert(axis_of_rotation, 0, cos))


    # Calculate the rotation
    points_rot_quaternion = Quaternion_q * points * quaternion.quaternion.conjugate(Quaternion_q)
    points_rot = np.array(quaternion.as_vector_part(points_rot_quaternion))

    # Round the values to the near value
    points_rot = np.round(points_rot).astype(int)


    return points_rot
def translation_then_axialrotation(img:np.ndarray, parameters: tuple[float, ...])-> np.ndarray:
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """

    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

    #img_trans = np.full(img.shape,-1000)
    img_trans = np.zeros(img.shape)
    #Get the tuples of indices

    points_img = np.argwhere(np.ones(img.shape))
    # Translate the points
    points_traslation = translation(points_img, np.array([t1,t2,t3]))
    # Transform the points to quaternions
    points_as_quaternions = PointsToquartenions(points_traslation)
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    axis_rotation = np.array([v1,v2,v3])
    # Rotate the points
    points_rot = axial_rotation(points_as_quaternions,angle_in_rads,axis_rotation)

    # Filter points

    del_indexes = my_filtering_function(points_rot, img.shape)
    points_rot = np.delete(points_rot, del_indexes, axis=0)
    points_img = np.delete(points_img, del_indexes, axis=0)

    if np.any(points_rot):
        img_trans[points_rot[:,0],points_rot[:,1],points_rot[:,2]] = img[points_img[:,0], points_img[:, 1], points_img[:,2]]

    return img_trans

def return_to_patient_space(img:np.ndarray, parameters: tuple[float, ...])-> np.ndarray:
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """

    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

    #img_trans = np.full(img.shape,-1000)
    img_trans = np.zeros(img.shape)
    #Get the tuples of indices

    points_img = np.argwhere(np.ones(img.shape))

    # Rotate the points
    # Transform the points to quaternions
    points_as_quaternions = PointsToquartenions(points_img)
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    axis_rotation = np.array([v1, v2, v3])
    points_rot = axial_rotation(points_as_quaternions, angle_in_rads, axis_rotation)

    # Translate the points
    points_translation = translation(points_rot, np.array([t1,t2,t3]))

    # Convert indices to integers
    points_translation = points_translation.astype(int)
    points_img = points_img.astype(int)

    # Filter points

    del_indexes = my_filtering_function(points_translation, img.shape)
    points_translation = np.delete(points_translation, del_indexes, axis=0)
    points_img = np.delete(points_img, del_indexes, axis=0)


    if np.any(points_translation):
        img_trans[points_translation[:,0],points_translation[:,1],points_translation[:,2]] = img[points_img[:,0], points_img[:, 1], points_img[:,2]]

    return img_trans


def my_filtering_function(points: np.ndarray,unwanted_value: tuple[int,...])->np.ndarray:
    cond1 = (points[:,0] >= unwanted_value[0]) | (points[:,0] < 0)
    cond2 = (points[:, 1] >= unwanted_value[1]) | (points[:, 1] < 0)
    cond3 = (points[:, 2] >= unwanted_value[2]) | (points[:, 2] < 0)

    conditions = cond1 | cond2 | cond3

    del_indexes = np.where(conditions)[0]
    return del_indexes

# Normalizar imagen
def coregister(ref_img: np.ndarray, inp_img: np.ndarray):
    """ Coregister two sets of images using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]

    def function_to_minimize(parameters):
        """ Transform input image, then compare with reference image."""
        t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
        print(f'  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).')
        print(f'  >> Rotation: {angle_in_rads:0.02f} rads around axis ({v1:0.02f}, {v2:0.02f}, {v3:0.02f}).')

        inp_transf = translation_then_axialrotation(inp_img,parameters)

        #inp_transf = screw_full(inp_img,parameters)

        # Transformar imagen con parametros
        error = mean_squared_error(ref_img, inp_transf)
        print("MSE = " + str(error))
        return error

    # Apply least squares optimization
    result = minimize(function_to_minimize,
                       initial_parameters,
                       method='Powell',
                       options={'disp':True})
    return result


def PointsToquartenions (points:np.ndarray)-> np.ndarray:

    quaternion_numpy = quaternion.as_quat_array(np.insert(points,0,0,axis=1))

    return quaternion_numpy

def get_thalamus_mask(img_atlas: np.ndarray) -> np.ndarray:

    mask = np.zeros_like(img_atlas)
    mask[img_atlas == 81] = 1
    mask[img_atlas == 82] = 1
    for i in range(121,151):
        mask[img_atlas == i] = 1
    return mask


def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    ind = np.where(mask == 1)
    return np.array([np.mean(ind[0]), np.mean(ind[1]), np.mean(ind[2])])

def visualize_axial_slice(img: np.ndarray,mask: np.ndarray,mask_centroid: np.ndarray,):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """

    img_slice = img[int(mask_centroid[0]), :, :]
    mask_slice = mask[int(mask_centroid[0]), :, :]

    img_cmapped = plt.colormaps["bone"](img_slice)
    mask_cmapped = plt.colormaps["prism"](mask_slice)
    mask_cmapped = mask_cmapped * mask_slice[..., np.newaxis]

    alpha = 0.25
    plt.imshow(img_cmapped * (1 - alpha) + mask_cmapped * alpha)
    plt.title(f'Segmentation with alpha {alpha}')
    plt.show()

def plot_orthogonal_slices_alpha_fusion(img, mask, mask_centroid, name):
    alpha = 0.25
    # plot 3 orthogonal slices
    plt.clf()
    a1 = plt.subplot(2, 2, 1)
    img_slice = img[int(mask_centroid[0]), :, :]
    mask_slice = mask[int(mask_centroid[0]), :, :]

    img_cmapped = plt.colormaps["bone"](img_slice)
    mask_cmapped = plt.colormaps["autumn"](mask_slice)
    mask_cmapped = mask_cmapped * mask_slice[..., np.newaxis]

    plt.imshow(img_cmapped * (1 - alpha) + mask_cmapped * alpha)
    # a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    img_slice = img[:, :, int(mask_centroid[2])]
    mask_slice = mask[:, :, int(mask_centroid[2])]

    img_cmapped = plt.colormaps["bone"](img_slice)
    mask_cmapped = plt.colormaps["autumn"](mask_slice)
    mask_cmapped = mask_cmapped * mask_slice[..., np.newaxis]

    plt.imshow(img_cmapped * (1 - alpha) + mask_cmapped * alpha)
    # a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    img_slice = img[:, int(mask_centroid[1]), :]
    mask_slice = mask[:, int(mask_centroid[1]), :]

    img_cmapped = plt.colormaps["bone"](img_slice)
    mask_cmapped = plt.colormaps["autumn"](mask_slice)
    mask_cmapped = mask_cmapped * mask_slice[..., np.newaxis]

    plt.imshow(img_cmapped * (1 - alpha) + mask_cmapped * alpha)
    # a3.set_aspect(cor_aspect)
    plt.title(name)
    plt.show()
    plt.clf()



if __name__ == '__main__':
    reference_path = "part_2/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
    reference = pydicom.read_file(reference_path)
    img_reference = reference.pixel_array # phantom

    input_path = "part_2/RM_Brain_3D-SPGR"
    # load the DICOM files
    input_slices = []
    # Use glob to get all the files in the folder
    files = glob.glob(input_path + '/*')
    for fname in files:
        print("loading: {}".format(fname))
        input_slices.append(pydicom.dcmread(fname))
    print("CT slices loaded! \n")

    # Sort according to ImagePositionPatient header, based on the 3rd component (which is equal to "SliceLocation")
    print("Sorting slices...")
    input_slices = sorted(input_slices, key=lambda ds: float(ds.ImagePositionPatient[2]))
    print("Slices sorted!")

    # pixel aspects, assuming all slices are the same
    ps = input_slices[0].PixelSpacing
    ss = input_slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]



    # create 3D array
    img_shape = list(input_slices[0].pixel_array.shape)
    img_shape.append(len(input_slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(input_slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d


    print("shape input img3d: " + str(img3d.shape))
    print("shape reference img: " + str(img_reference.shape))

    #print(reference)
    #print(input_slices[0])
    #[print(f'{dicom_tag}: {dicom_value}') for dicom_tag, dicom_value in input_slices[0].items()]
    #[print(f'{dicom_tag}: {dicom_value}') for dicom_tag, dicom_value in reference.items()]

    # by looking at Image Orientation we can see x and y axes are not in the same orientation
    img3d = np.flip(img3d, axis=0)
    img3d = np.flip(img3d, axis=1)



    pixel_spacing1 = [0.5078, 0.5078]  # Espaciado de píxeles de la input image
    pixel_spacing2 = [1, 1]  # Espaciado de píxeles de la reference image (phantom)

    img3d_resized = resize_images_to_same_scale(pixel_spacing1, img3d, pixel_spacing2)
    img3d_resized = np.transpose(img3d_resized, (2, 0, 1))


    plt.clf()
    #print("shape 1 " + str(img3d_resized.shape))
    img3d_resized = img3d_resized[9:-10, 15:-15, 33:-33]
    #print("shape 2 " + str(img3d_resized.shape))
    img3d = np.transpose(img3d, (2, 0, 1))
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[img3d.shape[0] // 2, :, :])
    #a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, :, img3d.shape[2] // 2])

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[:, img3d.shape[1] // 2, :])
    plt.title("IMG 3D (INPUT IMAGE) NO REESCLALED")
    plt.show()

    plt.clf()
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img_reference[img_reference.shape[0] // 2, :, :])
    #a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img_reference[:, :, img_reference.shape[2] // 2])
    #a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img_reference[:, img_reference.shape[1] // 2, :])
    #a3.set_aspect(cor_aspect)

    plt.title("REFERENCE IMAGE")
    plt.show()


    plt.clf()
    # plot 3 orthogonal slices
    a1 = plt.subplot(2, 2, 1)

    plt.imshow(img3d_resized[img3d_resized.shape[0] // 2, :, :])
    #a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)

    plt.imshow(img3d_resized[:, :, img3d_resized.shape[2] // 2])
    #a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d_resized[:, img3d_resized.shape[1] // 2, :])
    #a3.set_aspect(cor_aspect)
    plt.title("IMG 3D (INPUT IMAGE) RESCALED")
    plt.show()
    plt.clf()

    #input_pruebas = input_pruebas[:, :, 33:-33]
    print(img3d_resized.shape)
    print(img_reference.shape)
    input_pruebas = min_max_normalization(img3d_resized)
    phantom_pruebas = min_max_normalization(img_reference)


    params = coregister(phantom_pruebas, input_pruebas)
    params = params.x
    img_input_corregistered = translation_then_axialrotation(input_pruebas,params)
    print(img_input_corregistered.shape)

    # plot 3 orthogonal slices
    plt.clf()
    a1 = plt.subplot(2, 2, 1)

    plt.imshow(img_input_corregistered[img_input_corregistered.shape[0] // 2, :, :])
    # a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img_input_corregistered[:, :, img_input_corregistered.shape[2] // 2])
    # a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img_input_corregistered[:, img_input_corregistered.shape[1] // 2, :])
    # a3.set_aspect(cor_aspect)
    plt.title("IMG 3D (INPUT IMAGE) Coregistered")
    plt.show()
    plt.clf()

    #Animation to check right corregistration besides the subplots previously done
    name = "Corregistration_Comparison"
    animation_alpha_fusion(img_input_corregistered, phantom_pruebas, rotate_on_axial_plane,MIP_coronal_plane, name)

    # Visualize thalamus on the patient but in the normalized space
    dcm_atlas = pydicom.dcmread("/Users/gerardnaharrolopez/PycharmProjects/Project_PIM/part_2/AAL3_1mm.dcm")
    img_atlas = dcm_atlas.pixel_array
    thalamus = get_thalamus_mask(img_atlas)
    print(img_input_corregistered.shape)
    print(img_atlas.shape)

    #crop to atlas size
    img_imput_corregistered_cropped = img_input_corregistered[6:-6, 6:-6, 6:-6]
    name = "Thalamus_Mask_In_Normalized_Space"
    animation_alpha_fusion(img_imput_corregistered_cropped, thalamus, rotate_on_axial_plane, MIP_coronal_plane, name)

    mask_centroid = find_centroid(thalamus)
    name = "Orthogonal Slices In Normalized Space"
    plot_orthogonal_slices_alpha_fusion(img_imput_corregistered_cropped, thalamus, mask_centroid, name)

    # Revert the transformations to go back to the input space with thalamus applied
    inv_params = tuple([-x for x in params])
    img_input_uncorregistered = return_to_patient_space(img_imput_corregistered_cropped,inv_params)
    thalamus_patient_space = return_to_patient_space(thalamus, inv_params)

    name = "Thalamus_Mask_In_Patient_Space"
    animation_alpha_fusion(img_input_uncorregistered, thalamus_patient_space, rotate_on_axial_plane, MIP_coronal_plane, name)
    mask_centroid = find_centroid(thalamus_patient_space)
    name = "Orthogonal Slices In Patient Space"
    plot_orthogonal_slices_alpha_fusion(img_input_uncorregistered, thalamus_patient_space, mask_centroid, name)








