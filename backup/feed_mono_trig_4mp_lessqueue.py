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
from run_CCRF_mono import parallel_CCRF,ldr_tonemap_L_image
from multiprocessing import Process,Queue
import sys
import queue
import time
import multiprocessing
import os

class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

CHOSEN_TRIGGER = TriggerType.SOFTWARE

FRAME_WIDTH=400
FRAME_HEIGHT=300
BASE_EXPO=2000

def base_exposure(cam,k):#if k =2
    return 1/(cam.AcquisitionFrameRate.GetMax()/8)/sum([(k**i) for i in range(0,4)])*1000000

def configure_settings(cam):
    try:
        # Configure some settings to ensure no compensations is applied by default
        cam.GammaEnable.SetValue(False) #disable gamma correction
        #cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off) #Turn off auto balance white 
        #cam.BalanceRatio.SetValue(1) #Set balance ratio to 1 so it is the original image
        #cam.AutoExposureEVCompensation.SetValue(False) #disable EV compensation
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.GainAuto.SetValue(False)
        cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8)
        cam.AutoExposureTargetGreyValueAuto.SetValue(PySpin.AutoExposureTargetGreyValueAuto_Off) #disable Target Grey Value
        
        cam.AcquisitionFrameRateEnable.SetValue(True) #allow framerate change 
        cam.AcquisitionFrameRate.SetValue(cam.AcquisitionFrameRate.GetMax()) #change framerate
        cam.AcquisitionFrameRateEnable.SetValue(False) #disable framerate change
        print("Maximum achievable framerate is : ", cam.AcquisitionFrameRate.GetMax())
        return True
    except:
        return False

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

                image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                
                

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

def run_single_camera(cam,lv1_to_proc,lv3_from_proc):
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
        exposures=[(2**i)*int(base_exposure(cam,2)) for i in range(0,4)]
        print(exposures)
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
        canvas=np.zeros((FRAME_HEIGHT*2,FRAME_WIDTH*2,1), np.uint8)
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
        HDR_FRAME = np.zeros((FRAME_HEIGHT,FRAME_WIDTH))
        #cv2.namedWindow("HDR", cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty("HDR",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        while True:
            #exposure=exposures[index]
            
            configure_exposure(cam, exposures[index])
            # Acquire images
            err, img,width,height = acquire_images(cam, nodemap)

            
            img = img.GetData().reshape(height,width,1)
            
            img = img[bot:top,left:right]
            lv1_to_proc.put((img,index))

            '''
            if index==0:
                #top left
                topleft=img
                canvas[0:FRAME_HEIGHT,0:FRAME_WIDTH]=img
            elif index==1:
                #top right
                topright=img
                canvas[0:FRAME_HEIGHT,FRAME_WIDTH:FRAME_WIDTH*2]=img
                #calc1 = parallel_CCRF(topright,topleft)
            elif index==2:
                #bot left
                botleft=img
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,0:FRAME_WIDTH]=img
                #calc2 = parallel_CCRF(botleft,topright)
            else:
                #bot right
                botright=img
                canvas[FRAME_HEIGHT:FRAME_HEIGHT*2,FRAME_WIDTH:FRAME_WIDTH*2]=img
                #calc3 = parallel_CCRF(botright,botleft)#frame 3,4
                #calc4 = parallel_CCRF(calc2,calc1)
                #calc5 = parallel_CCRF(calc3,calc2)
                #print(HDR_FRAME)
                #HDR_FRAME = ldr_tonemap_L_image(HDR_FRAME/255,5,50)
                cv2.imshow("HDR",HDR_FRAME/255)
                if cv2.waitKey(5) & 0xff==ord('q'):
                    break
                HDR_FRAME = lv3_from_proc.get()
            '''
            index+=1
            if index>=len(exposures):
                index=0
            if lv3_from_proc.empty() is False:
                HDR_FRAME = lv3_from_proc.get()
            cv2.imshow("HDR",HDR_FRAME)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
            
            #cv2.imshow("frame",canvas)
            #if cv2.waitKey(1) &0xff ==ord('q'):
                #stop the feed the 'q'
            #    break
        cv2.destroyAllWindows()
        # Deinitialize camera
        cam.EndAcquisition()
        reset_trigger(cam)
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

def lv1_HDR_proc(lv1_in_queue,lv2_in_queue):
    frame1 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    frame2 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    frame3 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    frame4 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    info('lv1')
    while True:
        try:
            v=lv1_in_queue.get() #get
            if v is None:
                break
            if v[1] == 1:  #v[1] == f2q
                lv2_in_queue.put((parallel_CCRF(v[0],frame1),0)) # frame1 + frame2 --> calc1
                frame2 = v[0]
            elif v[1] ==2: #v[1] == f4q
                lv2_in_queue.put((parallel_CCRF(v[0],frame2),1)) # frame2 + frame3 --> calc2
                #calc4 = parallel_CCRF(calc2[0],calc1[0])     # calc1 + calc2 --> calc4
                frame3 = v[0]
            elif v[1] ==3: #v[1] == f8q
                lv2_in_queue.put((parallel_CCRF(v[0],frame3),2)) # frame3 + frame4 --> calc3
                #calc5 = parallel_CCRF(calc3[0],calc2[0])     # calc2 + calc3 --> calc5
                #result_HDR = parallel_CCRF(calc5,calc4)
                #lv1_out_queue.put(result_HDR) #put
                frame4 = v[0]
            elif v[1] == 0: #v[1] ==fq
                frame1 = v[0]
            #print(result)
            #process v
        except queue.Empty:
            continue
        except Exception as e:
            raise e

def lv2_HDR_proc(lv2_in_queue,lv3_in_queue):
    calc1 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    calc2 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    calc3 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    info('lv2')
    while True:
        try:
            v=lv2_in_queue.get() #get
            if v is None:
                break
            if v[1] == 1:
                lv3_in_queue.put((parallel_CCRF(v[0],calc1),0)) # calc2 + calc1 --> calc4
                calc2 = v[0]
            elif v[1] == 2:
                lv3_in_queue.put((parallel_CCRF(v[0],calc2),1)) # calc3 + calc2 --> calc5
                calc3 = v[0]
            elif v[1] == 0:
                calc1 = v[0]
        except queue.Empty:
            continue
        except Exception as e:
            raise e
def lv3_HDR_proc(lv3_in_queue,lv3_out_queue):
    calc4 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    calc5 =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    #result_HDR =np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    HDR_FRAME = np.zeros((FRAME_HEIGHT,FRAME_WIDTH),np.uint8)
    #cv2.namedWindow("HDR", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("HDR",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    init_time = time.time()
    info('lv3')
    while True:
        lv3_out_queue.put(HDR_FRAME) 
        try:
            v=lv3_in_queue.get() #get
            if v is None:
                break
            if v[1] == 1:
                #lv3_out_queue.put(parallel_CCRF(v[0],calc4)) #calc5 + calc4 -->HDR_frame
                HDR_FRAME = parallel_CCRF(v[0],calc4)
                print("Time taken to spit one HDR frame = ",time.time()-init_time)
                init_time = time.time()
                calc5 = v[0]
            elif v[1] == 0: # got calc4 frame
                calc4 = v[0]
        except queue.Empty:
            continue
        except Exception as e:
            raise e
    #cv2.destroyAllWindows()
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == "__main__":

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
        sys.exit(0)
    
    lv1_to_proc = Queue()
    lv2_to_proc = Queue()
    lv3_to_proc = Queue()
    lv3_from_proc=Queue()
    p1 = Process(target=lv1_HDR_proc, args=(lv1_to_proc,lv2_to_proc))
    p2 = Process(target=lv2_HDR_proc, args=(lv2_to_proc,lv3_to_proc))
    p3 = Process(target=lv3_HDR_proc, args=(lv3_to_proc,lv3_from_proc))
    p1.start()
    p2.start()
    p3.start()
    print("Number of cpus =",multiprocessing.cpu_count())
    cam = cam_list.GetByIndex(0)
    cv2.namedWindow("HDR", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("HDR",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    run_single_camera(cam,lv1_to_proc,lv3_from_proc)
    #-----------------------------------------------

    lv1_to_proc.put(None)
    lv2_to_proc.put(None)
    lv3_to_proc.put(None)
    while lv3_from_proc.empty() is False:
        #print("in loop")
        try:
            x= lv3_from_proc.get()
        except:
            continue
    '''
    lv2_to_proc.put(None)
    while lv3_from_proc.empty() is False:
        print("in loop")
        try:
            print(lv3_from_proc.get())
        except:
            continue
    lv3_to_proc.put(None)
    while lv3_from_proc.empty() is False:
        print("in loop")
        try:
            print(lv3_from_proc.get())
        except:
            continue
    '''
    p1.join()
    p2.join()
    p3.join()
    #-----------------------------------------
    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release instance
    system.ReleaseInstance()

