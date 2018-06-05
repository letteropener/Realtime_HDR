# coding=utf-8
# =============================================================================
# Copyright © 2017 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
#  SaveToAvi.py shows how to create an AVI video from a vector of
#  images. It relies on information provided in the Enumeration, Acquisition,
#  and NodeMapInfo examples.
#
#  This example introduces the AVIRecorder class, which is used to quickly and
#  easily create various types of AVI videos. It demonstrates the creation of
#  three types: uncompressed, MJPG, and H264.

import PySpin
import cv2
import numpy as np
#from run_CCRF_RGB import parallel_CCRF_B
#from run_CCRF_RGB import parallel_CCRF_G
#from run_CCRF_RGB import parallel_CCRF_R
#from numba import vectorize


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

class HDRType:
    CCRF = 1
    Weighted_Avg = 2

CHOSEN_HDR = HDRType.CCRF
CHOSEN_TRIGGER = TriggerType.SOFTWARE

FRAME_WIDTH=400
FRAME_HEIGHT=300
BASE_EXPO=2000
exposures=[2000,4000,8000,16000]
MAX_FRAME_RATE = 118.38971

#cv2.namedWindow('frame')
#cv2.namedWindow("HDR")

#@vectorize([uint8(uint8,uint8)],target='parallel')
#def HDR_calc(f2q,fq):
#    calc_B=parallel_CCRF_B(f2q[:,:,0],fq[:,:,0])
#    calc_G=parallel_CCRF_G(f2q[:,:,1],fq[:,:,1])
#    calc_R=parallel_CCRF_R(f2q[:,:,2],fq[:,:,2])
#    calc = np.dstack((calc_B,calc_G,calc_R))
#    return calc

def configure_settings(cam):
    # Configure some settings to ensure no compensations is applied by default
    cam.GammaEnable.SetValue(False) #disable gamma correction
    cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off) #Turn off auto balance white 
    cam.BalanceRatio.SetValue(1) #Set balance ratio to 1 so it is the original image
    cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8)
    cam.GainAuto.SetValue(False)
    #cam.AutoExposureEVCompensation.SetValue(False) #disable EV compensation
    cam.AutoExposureTargetGreyValueAuto.SetValue(PySpin.AutoExposureTargetGreyValueAuto_Off) #disable Target Grey Value
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

    cam.AcquisitionFrameRateEnable.SetValue(True) #allow framerate change 
    cam.AcquisitionFrameRate.SetValue(cam.AcquisitionFrameRate.GetMax()) #change framerate
    cam.AcquisitionFrameRateEnable.SetValue(False) #disable framerate change
    print(cam.AcquisitionFrameRate.GetMax())
def configure_trigger(cam):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    ensured to be off in order to select the trigger source. Trigger mode is
    then enabled, which has the camera capture only a single image upon the
    execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    print("*** CONFIGURING TRIGGER ***\n")

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print("Software trigger chosen...")
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print("Hardware trigger chose...")

    try:
        result = True

        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        
        print("Trigger mode disabled...")

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
		# mode is off.
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print("Unable to get trigger source (node retrieval). Aborting...")
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        
        print("Trigger mode turned back on...")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result

def grab_next_image_by_trigger(cam):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        #result = True
        # Use trigger to capture image
        # The software trigger only feigns being executed by the Enter key;
        # what might not be immediately apparent is that there is not a
        # continuous stream of images being captured; in other examples that
        # acquire images, the camera captures a continuous stream of images.
        # When an image is retrieved, it is plucked from the stream.

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            # Get user input
            #input("Press the Enter key to initiate software trigger.")

            # Execute software trigger
            if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                print("Unable to execute trigger. Aborting...")
                #return False

            cam.TriggerSoftware.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            print("Use the hardware to trigger image acquisition.")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        #return False

    #return result

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

        #  Begin acquiring images
        #cam.BeginAcquisition()

        #print("Acquiring images...")

        # Retrieve, convert, and save images
        

        try:
            #  Retrieve the next image from the trigger
            grab_next_image_by_trigger(cam)

            #  Retrieve next received image
            image_result = cam.GetNextImage()

            #  Ensure image completion
            if image_result.IsIncomplete():
                print("Image incomplete with image status %d..." % image_result.GetImageStatus())

            else:
                #  Print image information; height and width recorded in pixels
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                
                #  Convert image to rgb 8 and append to list

                image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                
                

                #  Release image
                image_result.Release()
                #print("")

        except PySpin.SpinnakerException as ex:
            print("Error: %s" % ex)
            result = False

        # End acquisition
        #cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result, image_converted,width,height

def reset_trigger(cam):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print("Trigger mode disabled...")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result

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
        # Initialize camera
        cam.Init()

        configure_settings(cam)

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()
        exposures=[4000,8000,16000,32000]
        index=0

        #configure the trigger
        if configure_trigger(cam) is False:
            return False

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
        
        print("Automatic exposure disabled...")
        #node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))

        # if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
        #     print("Unable to retrieve frame rate. Aborting...")
        #     return False

        # framerate_to_set = node_acquisition_framerate.GetValue()

        # print("Frame rate to be set to %d..." % framerate_to_set)
        canvas=np.zeros((FRAME_HEIGHT*2,FRAME_WIDTH*2,3), np.uint8)
        #HDR_FRAME=np.zeros((FRAME_HEIGHT,FRAME_WIDTH,3), np.uint8)
        cam.BeginAcquisition()
        err, img,width,height = acquire_images(cam, nodemap)
        if err < 0:
            return err
        half_frame_height = int(FRAME_HEIGHT/2)
        half_frame_width = int(FRAME_WIDTH/2)
        half_height = int(height/2)
        half_width = int(width/2)

        bot = half_height-half_frame_height
        top = half_height+half_frame_height
        left = half_width-half_frame_width
        right = half_width+half_frame_width

        while True:
            #exposure=exposures[index]
            
            configure_exposure(cam, exposures[index])
            # Acquire images
            err, img,width,height = acquire_images(cam, nodemap)

            img = img.GetData().reshape(height,width,3)
            
            img = img[bot:top,left:right]
            #smallimg=cv2.resize(img,(int(FRAME_WIDTH/2),int(FRAME_HEIGHT/2)))
            if index==0:
                #top left
                topleft=img
                canvas[0:FRAME_HEIGHT,0:FRAME_WIDTH]=img
            elif index==1:
                #top right
                topright=img
                canvas[0:FRAME_HEIGHT,FRAME_WIDTH:FRAME_WIDTH*2]=img
                #calc1 = HDR_calc(topright,topleft)     
            elif index==2:
                #bot left
                botleft=img
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,0:FRAME_WIDTH]=img
                #calc2 = HDR_calc(botleft,topright)
            else:
                #bot right
                botright=img
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH:FRAME_WIDTH*2]=img
                #calc3 = HDR_calc(botright,botleft)
                #calc4 = HDR_calc(calc2,calc1) 
                #calc5 = HDR_calc(calc3,calc2)
                #HDR_FRAME = HDR_calc(calc5,calc4)
                
                #cv2.imshow("HDR",HDR_FRAME/255)
                #if cv2.waitKey(5) & 0xff==ord('q'):
                #    break

            index+=1
            if index>=len(exposures):
                index=0

            cv2.imshow("frame",canvas)
            
            if cv2.waitKey(1) &0xff ==ord('q'):
                #stop the feed the 'q'
                break
        cv2.destroyAllWindows()
        # Deinitialize camera
        dm=reset_trigger(cam)
        cam.EndAcquisition()
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
        #exposure_time_to_set = exposure
        #exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure)

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
    #if CHOSEN_HDR == HDRType.CCRF:
    #    HDR_dict = {"B": parallel_CCRF_B}
    main()
