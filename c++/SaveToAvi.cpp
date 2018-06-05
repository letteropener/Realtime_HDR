//=============================================================================
// Copyright © 2017 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================*/

/**
 *	@example SaveToAvi.cpp
 *
 *	@brief SaveToAvi.cpp shows how to create an AVI video from a vector of 
 *	images. It relies on information provided in the Enumeration, Acquisition,
 *	and NodeMapInfo examples.  
 *
 *	This example introduces the AVIRecorder class, which is used to quickly and
 *	easily create various types of AVI videos. It demonstrates the creation of
 *	three types: uncompressed, MJPG, and H264. 
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include "AVIRecorder.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

// Use the following enum and global constant to select the type of AVI video
// file to be created and saved.
enum aviType
{
	UNCOMPRESSED,
	MJPG,
	H264
};

const aviType chosenAviType = UNCOMPRESSED;
Mat ConvertToCVmat(ImagePtr pImage, unsigned int& rowsize, unsigned int&colsize )
{
	int result = 0;
	ImagePtr convertedImage = pImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);

	unsigned int XPadding = convertedImage->GetXPadding();
	unsigned int YPadding = convertedImage->GetYPadding();
	rowsize = convertedImage->GetWidth();
	colsize = convertedImage->GetHeight();

	//image data contains padding. When allocating Mat container size, you need to account for the X,Y image data padding. 
	Mat cvimg = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC3, convertedImage->GetData(), convertedImage->GetStride());
	//namedWindow("current Image", CV_WINDOW_AUTOSIZE);
	//imshow("current Image", cvimg);
	//resizeWindow("current Image", rowsize / 2, colsize / 2);
	//otherwise the image will not display...
	
	return cvimg;
}

int ConfigureExposure(CameraPtr pCam, float exposure)
{
	int result = 0;

	//cout << endl << endl << "*** CONFIGURING EXPOSURE ***" << endl << endl;

	try
	{
		//
		// Turn off automatic exposure mode
		//
		// *** NOTES ***
		// Automatic exposure prevents the manual configuration of exposure 
		// times and needs to be turned off for this example. Enumerations
		// representing entry nodes have been added to QuickSpin. This allows
		// for the much easier setting of enumeration nodes to new values.
		// 
		// In C++, the naming convention of QuickSpin enums is the name of the
		// enumeration node followed by an underscore and the symbolic of
		// the entry node. Selecting "Off" on the "ExposureAuto" node is
		// thus named "ExposureAuto_Off".
		//
		// *** LATER ***
		// Exposure time can be set automatically or manually as needed. This
		// example turns automatic exposure off to set it manually and back
		// on to return the camera to its default state.
		//
		//if (pCam->ExposureAuto == NULL || pCam->ExposureAuto.GetAccessMode() != RW)
		//{
		//	cout << "Unable to disable automatic exposure. Aborting..." << endl << endl;
		//	return -1;
		//}

		//pCam->ExposureAuto.SetValue(ExposureAuto_Off);

		//cout << "Automatic exposure disabled..." << endl;

		//
		// Set exposure time manually; exposure time recorded in microseconds
		//
		// *** NOTES ***
		// Notice that the node is checked for availability and writability
		// prior to the setting of the node. In QuickSpin, availability is
		// ensured by checking for null while writability is ensured by checking
		// the access mode. 
		//
		// Further, it is ensured that the desired exposure time does not exceed 
		// the maximum. Exposure time is counted in microseconds - this can be 
		// found out either by retrieving the unit with the GetUnit() method or 
		// by checking SpinView.
		// 
		if (pCam->ExposureTime == NULL || pCam->ExposureTime.GetAccessMode() != RW)
		{
			cout << "Unable to set exposure time. Aborting..." << endl << endl;
			return -1;
		}

		// Ensure desired exposure time does not exceed the maximum
		const double exposureTimeMax = pCam->ExposureTime.GetMax();
		double exposureTimeToSet = exposure;

		if (exposureTimeToSet > exposureTimeMax)
		{
			exposureTimeToSet = exposureTimeMax;
		}

		pCam->ExposureTime.SetValue(exposureTimeToSet);

		//cout << "Shutter time set to " << exposureTimeToSet << " us..." << endl << endl;
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}


// This function prepares, saves, and cleans up an AVI video from a vector of 
// images.
int SaveVectorToAVI(INodeMap & nodeMap, INodeMap & nodeMapTLDevice, vector<ImagePtr> & images)
{
	int result = 0;

	cout << endl << endl << "*** CREATING VIDEO ***" << endl << endl;

	try
	{
		// Retrieve device serial number for filename
		string deviceSerialNumber = "";

		CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
		if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
		{
			deviceSerialNumber = ptrStringSerial->GetValue();

			cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
		}

		//
		// Get the current frame rate; acquisition frame rate recorded in hertz
		//
		// *** NOTES ***
		// The video frame rate can be set to anything; however, in order to 
		// have videos play in real-time, the acquisition frame rate can be
		// retrieved from the camera.
		//
		CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
		if (!IsAvailable(ptrAcquisitionFrameRate) || !IsReadable(ptrAcquisitionFrameRate))
		{
			cout << "Unable to retrieve frame rate. Aborting..." << endl << endl;
			return -1;
		}

		float frameRateToSet = static_cast<float>(ptrAcquisitionFrameRate->GetValue());

		cout << "Frame rate to be set to " << ptrAcquisitionFrameRate->GetValue() << "...";

		//
		// Create a unique filename
		//
		// *** NOTES ***
		// This example creates filenames according to the type of AVI video 
		// being created. Notice that '.avi' does not need to be appended to the 
		// name of the file. This is because the AVI recorder object takes care
		// of the file extension automatically.
		//
		string aviFilename;

		switch (chosenAviType)
		{
		case UNCOMPRESSED:
			aviFilename = "SaveToAvi-Uncompressed";
			if (deviceSerialNumber != "")
			{
				aviFilename = aviFilename + "-" + deviceSerialNumber.c_str();
			}

			break;

		case MJPG:
			aviFilename = "SaveToAvi-MJPG";
			if (deviceSerialNumber != "")
			{
				aviFilename = aviFilename + "-" + deviceSerialNumber.c_str();
			}

			break;

		case H264:
			aviFilename = "SaveToAvi-H264";
			if (deviceSerialNumber != "")
			{
				aviFilename = aviFilename + "-" + deviceSerialNumber.c_str();
			}
		}

		//
		// Select option and open AVI filetype
		//
		// *** NOTES ***
		// Depending on the filetype, a number of settings need to be set in
		// an object called an option. An uncompressed option only needs to
		// have the video frame rate set whereas videos with MJPG or H264
		// compressions should have more values set.
		//
		// Once the desired option object is configured, open the AVI file
		// with the option in order to create the image file.
		//
		// *** LATER ***
		// Once all images have been added, it is important to close the file -
		// this is similar to many other standard file streams.
		//
		AVIRecorder aviRecorder;

		// Set maximum AVI file size to 2GB.
		// A new AVI file is generated when 2GB
		// limit is reached. Setting maximum file
		// size to 0 indicates no limit.
		const unsigned int k_aviFileSize = 2048;

		aviRecorder.SetMaximumAVISize(k_aviFileSize);

		if (chosenAviType == UNCOMPRESSED)
		{
			AVIOption option;

			option.frameRate = frameRateToSet;

			aviRecorder.AVIOpen(aviFilename.c_str(), option);
		}
		else if (chosenAviType == MJPG)
		{
			MJPGOption option;

			option.frameRate = frameRateToSet;
			option.quality = 75;

			aviRecorder.AVIOpen(aviFilename.c_str(), option);
		}
		else if (chosenAviType == H264)
		{
			H264Option option;

			option.frameRate = frameRateToSet;
			option.bitrate = 1000000;
			option.height = static_cast<unsigned int>(images[0]->GetHeight());
			option.width = static_cast<unsigned int>(images[0]->GetWidth());

			aviRecorder.AVIOpen(aviFilename.c_str(), option);
		}

		//
		// Construct and save AVI video
		//
		// *** NOTES ***
		// Although the video file has been opened, images must be individually
		// appended in order to construct the video. 
		//
		cout << "Appending " << images.size() << " images to AVI file: " << aviFilename << ".avi... " << endl << endl;

		for (unsigned int imageCnt = 0; imageCnt < images.size(); imageCnt++)
		{
			aviRecorder.AVIAppend(images[imageCnt]);

			cout << "\tAppended image " << imageCnt << "..." << endl;
		}

		//
		// Close AVI file
		//
		// *** NOTES ***
		// Once all images have been appended, it is important to close the
		// AVI file. Notice that once an AVI file has been closed, no more
		// images can be added.
		//
		aviRecorder.AVIClose();

		cout << endl << "Video saved at " << aviFilename << ".avi" << endl << endl;
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(INodeMap & nodeMap)
{
	int result = 0;

	cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

	try
	{
		FeatureList_t features;
		CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
		if (IsAvailable(category) && IsReadable(category))
		{
			category->GetFeatures(features);

			FeatureList_t::const_iterator it;
			for (it = features.begin(); it != features.end(); ++it)
			{
				CNodePtr pfeatureNode = *it;
				cout << pfeatureNode->GetName() << " : ";
				CValuePtr pValue = (CValuePtr)pfeatureNode;
				cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
				cout << endl;
			}
		}
		else
		{
			cout << "Device control information not available." << endl;
		}
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

// This function acquires and saves 10 images from a device; please see
// Acquisition example for more in-depth comments on acquiring images.
int AcquireImages(CameraPtr pCam, INodeMap & nodeMap)
{
	int result = 0;
	ImagePtr pResultImage;
	Mat cvimg;
	unsigned int rowsize;
	unsigned int colsize;
	cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;
	pCam->BeginAcquisition();
	namedWindow( "current Image", WINDOW_AUTOSIZE );
	
	while(1){

		try
		{

			//cout << "Acquiring images..." << endl << endl;

			// Retrieve and convert images
			//const unsigned int k_numImages = 10;

			//for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
			//{
				// Retrieve the next received image
			pResultImage = pCam->GetNextImage();

			try
			{
				if (pResultImage->IsIncomplete())
				{
					cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl << endl;
				}
				else
				{
					//cout << "Grabbed imag: width = " << pResultImage->GetWidth() << ", height = " << pResultImage->GetHeight() << endl;

					//ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);
					ImagePtr convertedImage = pResultImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);
					
					//ImagePtr convertedImage = pResultImage->Convert(PixelFormat_BGR8, HQ_LINEAR);
					rowsize = convertedImage->GetWidth();
					colsize = convertedImage->GetHeight();
					cvimg = cv::Mat(colsize, rowsize, CV_8UC3, convertedImage->GetData(), convertedImage->GetStride());
					//printf("yay2");
					imshow("current Image", cvimg);
					//resizeWindow("current Image", rowsize / 2, colsize / 2);
					//printf("yay3");
					pResultImage->Release();
					//printf("yay1");
					if ((char)27==waitKey(1)&0xff){
						break;
					}
			
				}
			}
			

				// Release image
				//pResultImage->Release();
			//}

			// End acquisition
			catch (Spinnaker::Exception &e)
			{
				cout << "Error: " << e.what() << endl;
				result = -1;
			}
		}
		catch (Spinnaker::Exception &e)
		{
			cout << "Error: " << e.what() << endl;
			result = -1;
		}	
	}
	pCam->EndAcquisition();
	destroyAllWindows();
	

	return result;
}

// This function acts as the body of the example; please see NodeMapInfo example 
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
	int result = 0;
	int self_result = 0;
	int err = 0;
	float exposures[4] = {200.0, 400.0, 800.0, 1600.0};
	ImagePtr pImage, convertedImage;
	Mat cvimg;
	unsigned int rowsize;
	unsigned int colsize;
	unsigned int XPadding;
	unsigned int YPadding;

	try
	{
		// Retrieve TL device nodemap and print device information
		INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
		result = 0;

		//result = PrintDeviceInfo(nodeMapTLDevice);

		// Initialize camera
		pCam->Init();
		//----------------auto off---------------------------------
		pCam->GainAuto.SetValue(GainAuto_Off);
		pCam->AdcBitDepth.SetValue(AdcBitDepth_Bit8);
		pCam->AutoExposureTargetGreyValueAuto.SetValue(AutoExposureTargetGreyValueAuto_Off);
		pCam->AcquisitionFrameRateEnable.SetValue(true);
		pCam->AcquisitionFrameRate.SetValue(pCam->AcquisitionFrameRate.GetMax());
		pCam->AcquisitionFrameRateEnable.SetValue(false);
		//-----------------------------------------------------------

		// Retrieve GenICam nodemap
		INodeMap & nodeMap = pCam->GetNodeMap();

		// Acquire images and save into vector
		vector<ImagePtr> images;
		//------------configure exposure------------------
		if (pCam->ExposureAuto == NULL || pCam->ExposureAuto.GetAccessMode() != RW)
		{
			cout << "Unable to disable automatic exposure. Aborting..." << endl << endl;
			return -1;
		}

		pCam->ExposureAuto.SetValue(ExposureAuto_Off);

		//cout << "Automatic exposure disabled..." << endl;
		//--------acquire image-------------------------------------
		// Set acquisition mode to continuous
		CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
		if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
		{
			cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
			return -1;
		}

		CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
		if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
		{
			cout << "Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting..." << endl << endl;
			return -1;
		}

		int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

		ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
		ConfigureExposure(pCam, exposures[2]);

		cout << "Acquisition mode set to continuous..." << endl;
		//------------------------------------------------------
		//namedWindow( "4 Displays", WINDOW_AUTOSIZE );
		// Begin acquiring images
		//pCam->BeginAcquisition();
		result = AcquireImages(pCam, nodeMap);
		//convertedImage = pImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);
		//XPadding = convertedImage->GetXPadding();
		//YPadding = convertedImage->GetYPadding();
		//rowsize = convertedImage->GetWidth();
		//colsize = convertedImage->GetHeight();
		//cvimg = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC3, convertedImage->GetData(), convertedImage->GetStride());
		//pImage->Release();
		//printf("yay!\n");
		//if (err < 0)
		//{
		//	return err;
		//}
		/*while(1){
			
			//printf("yay2!\n");
			pImage = AcquireImages(pCam, nodeMap);
			//printf("yay3!\n");
			convertedImage = pImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);
			//XPadding = convertedImage->GetXPadding();
			//YPadding = convertedImage->GetYPadding();
			
			//resizeWindow("current Image", rowsize / 2, colsize / 2);
			pImage->Release();
			if ((char)27==waitKey(1)&0xff){
				break;
			}
			//pCam->EndAcquisition();
		}*/
		// Save vector of images to AVI
		//result = result | SaveVectorToAVI(nodeMap, nodeMapTLDevice, images);
		//pCam->EndAcquisition();
		//destroyAllWindows();
		// Deinitialize camera
		pCam->DeInit();
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

// Example entry point; please see Enumeration example for more in-depth 
// comments on preparing and cleaning up the system.
int main(int /*argc*/, char** /*argv*/)
{
	int result = 0;

	// Print application build information
	//cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

	// Retrieve singleton reference to system object
	SystemPtr system = System::GetInstance();

	// Retrieve list of cameras from the system
	CameraList camList = system->GetCameras();

	unsigned int numCameras = camList.GetSize();

	cout << "Number of cameras detected: " << numCameras << endl << endl;

	// Finish if there are no cameras
	if (numCameras == 0)
	{
		// Clear camera list before releasing system
		camList.Clear();

		// Release system
		system->ReleaseInstance();

		cout << "Not enough cameras!" << endl;
		cout << "Done! Press Enter to exit..." << endl;
		getchar();

		return -1;
	}

	// Run example on each camera
	for (unsigned int i = 0; i < numCameras; i++)
	{
		cout << endl << "Running example for camera " << i << "..." << endl;

		result = result | RunSingleCamera(camList.GetByIndex(i));

		cout << "Camera " << i << " example complete..." << endl << endl;
	}

	// Clear camera list before releasing system
	camList.Clear();

	// Release system
	system->ReleaseInstance();

	cout << endl << "Done! Press Enter to exit..." << endl;
	getchar();

	return result;
}
