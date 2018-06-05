#
import os
import numpy as np
import PySpin

import cv2
import datetime
from math import *

FRAME_WIDTH=720
FRAME_HEIGHT=540
k = 2
base = 16
bias_base = 10
bias = [log(bias_base+i,3) for i in range(1,540000,18000)]
exposure_range = 11
#exposures=[(base+bias*j)*k**i for i in range(7)] #exposure list comprehension

IMG_FOLDER="photo/{}".format(datetime.datetime.now().strftime("%d %b %Y %H_%M_%S"))

def configure_settings(cam):
    # Configure some settings to ensure no compensations is applied by default
    cam.GammaEnable.SetValue(False) #disable gamma correction  
    cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off) #Turn off auto balance white 
    cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
    cam.BalanceRatio.SetValue(1.1) #
    cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
    cam.BalanceRatio.SetValue(3.0) #
    #cam.AutoExposureEVCompensation.SetValue(False) #disable EV compensation
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.GainAuto.SetValue(False)
    #cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8)
    cam.AutoExposureTargetGreyValueAuto.SetValue(PySpin.AutoExposureTargetGreyValueAuto_Off) #disable Target Grey Value
    
    cam.AcquisitionFrameRateEnable.SetValue(True) #allow framerate change 
    cam.AcquisitionFrameRate.SetValue(cam.AcquisitionFrameRate.GetMax()) #change framerate
    cam.AcquisitionFrameRateEnable.SetValue(False) #disable framerate change
    return True



def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print("\n*** DEVICE INFORMATION ***\n")

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode("DeviceInformation"))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print("%s: %s" % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else "Node not readable"))

        else:
            print("Device control information not available.")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result


def acquire_images(cam, nodemap):
    """
    This function acquires 10 images from a device, stores them in a list, and returns the list.
    please see the Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    #print("*** IMAGE ACQUISITION ***\n")
    try:
        result = True

        # Set acquisition mode to continuous
        

        #  Begin acquiring images
        cam.BeginAcquisition()

        #print("Acquiring images...")

        # Retrieve, convert, and save images
        

        try:
            #  Retrieve next received image
            image_result = cam.GetNextImage()

            #  Ensure image completion
            if image_result.IsIncomplete():
                print("Image incomplete with image status %d..." % image_result.GetImageStatus())

            else:
                #  Print image information; height and width recorded in pixels
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                
                #  Convert image to mono 8 and append to list

                image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                
                

                #  Release image
                image_result.Release()
                #print("")

        except PySpin.SpinnakerException as ex:
            print("Error: %s" % ex)
            result = False

        # End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result, image_converted,width,height


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run example on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        # Retrieve TL device nodemap and print device information
        #nodemap_tldevice = cam.GetTLDeviceNodeMap()

        #result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        configure_settings(cam)
        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        index=0
        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print("Unable to disable automatic exposure. Aborting...")
            return False
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print("Acquisition mode set to continuous...")

        # Set maximum width
        #
        # *** NOTES ***
        # Other nodes, such as those corresponding to image width and height,
        # might have an increment other than 1. In these cases, it can be
        # important to check that the desired value is a multiple of the
        # increment.
        #
        # This is often the case for width and height nodes. However, because
        # these nodes are being set to their maximums, there is no real reason
        # to check against the increment.
        if cam.Width.GetAccessMode() == PySpin.RW and cam.Width.GetInc() != 0 and cam.Width.GetMax != 0:
            cam.Width.SetValue(FRAME_WIDTH)
            print("Width set to %i..." % cam.Width.GetValue())

        else:
            print("Width not available...")
            result = False

        # Set maximum height
        #
        # *** NOTES ***
        # A maximum is retrieved with the method GetMax(). A node's minimum and
        # maximum should always be a multiple of its increment.
        if cam.Height.GetAccessMode() == PySpin.RW and cam.Height.GetInc() != 0 and cam.Height.GetMax != 0:
            cam.Height.SetValue(FRAME_HEIGHT)
            print("Height set to %i..." % cam.Height.GetValue())

        else:
            print("Height not available...")
            result = False

        print("Automatic exposure disabled...")
        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        for j in range(len(bias)): #loop for bias
            exposures=[(base+bias[j])*k**i for i in range(exposure_range)] #exposure list comprehension
            index = 0 #reset index
            while index<len(exposures):
                exposure=exposures[index]
                
                configure_exposure(cam, exposure)
                # Acquire images
                err, img,width,height = acquire_images(cam, nodemap)
                if err < 0:
                    return err

                
                img = img.GetData().reshape(height,width,3)
                
                
                index+=1
                imgdir="{}.jpg".format(exposure)
                imgdir=os.path.join(IMG_FOLDER,imgdir)
                cv2.imwrite(imgdir,img)
                

            
       
        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    
    

    



def configure_exposure(cam,exposure):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.

     :param cam: Camera to configure exposure for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    #print("*** CONFIGURING EXPOSURE ***\n")

    try:
        result = True

        # Turn off automatic exposure mode
        #
        # *** NOTES ***
        # Automatic exposure prevents the manual configuration of exposure
        # times and needs to be turned off for this example. Enumerations
        # representing entry nodes have been added to QuickSpin. This allows
        # for the much easier setting of enumeration nodes to new values.
        #
        # The naming convention of QuickSpin enums is the name of the
        # enumeration node followed by an underscore and the symbolic of
        # the entry node. Selecting "Off" on the "ExposureAuto" node is
        # thus named "ExposureAuto_Off".
        #
        # *** LATER ***
        # Exposure time can be set automatically or manually as needed. This
        # example turns automatic exposure off to set it manually and back
        # on to return the camera to its default state.

        

        # Set exposure time manually; exposure time recorded in microseconds
        #
        # *** NOTES ***
        # Notice that the node is checked for availability and writability
        # prior to the setting of the node. In QuickSpin, availability and
        # writability are ensured by checking the access mode.
        #
        # Further, it is ensured that the desired exposure time does not exceed
        # the maximum. Exposure time is counted in microseconds - this can be
        # found out either by retrieving the unit with the GetUnit() method or
        # by checking SpinView.

        if cam.ExposureTime.GetAccessMode() != PySpin.RW:
            print("Unable to set exposure time. Aborting...")
            return False

        # Ensure desired exposure time does not exceed the maximum
        exposure_time_to_set = exposure
        exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure_time_to_set)

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("Number of cameras detected:", num_cameras)
    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system
        system.ReleaseInstance()

        print("Not enough cameras!")
        input("Done! Press Enter to exit...")
        return False

    cam = cam_list.GetByIndex(0)
    run_single_camera(cam)


    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release instance
    system.ReleaseInstance()

    

if __name__ == "__main__":
    if os.path.isdir(IMG_FOLDER) is False:
        os.mkdir(IMG_FOLDER)
    main()
