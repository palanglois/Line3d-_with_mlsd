/*
Line3D++ - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// check libs
#include "configLIBS.h"

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>
#include <iomanip>

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"

// INFO:
// This executable reads mavmap results (image-data-*.txt) and executes the Line3D++ algorithm.
// Currently, only the PINHOLE camera model is supported!
// If distortion coefficients are stored in the sfm_data file, you need to use the _original_
// (distorted) images!

int main(int argc, char *argv[])
{


    //Finding the media directory
    string mediaDir = string(TEST_DIR);

    string defaultInputDir = mediaDir + "CastleP30";
    string defaultMavmapDir = mediaDir + "CastleP30/cameras";

    //Make the images.txt file
    ofstream imageFile((defaultInputDir + "/images.txt").c_str());
    imageFile << 30 << endl;
    for(int i=0; i< 30; i++)
        imageFile<< setfill('0') << setw(4) << i << " " << defaultInputDir
                 << "/" << setfill('0') << setw(4) << i << ".png" << endl;

    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", false, defaultInputDir, "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> mavmapArg("b", "mavmap_output", "full path to the mavmap output (image-data-*.txt)", false, defaultMavmapDir, "string");
    cmd.add(mavmapArg);

    TCLAP::ValueArg<std::string> extArg("t", "image_extension", "image extension (case sensitive), if not specified: jpg, png or bmp expected", false, "", "string");
    cmd.add(extArg);

    TCLAP::ValueArg<std::string> prefixArg("f", "image_prefix", "optional image prefix", false, "", "string");
    cmd.add(prefixArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> input_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);

    TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
    cmd.add(sigma_A_Arg);

    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
    cmd.add(sigma_P_Arg);

    TCLAP::ValueArg<float> epipolarArg("e", "min_epipolar_overlap", "minimum epipolar overlap for matching", false, L3D_DEF_EPIPOLAR_OVERLAP, "float");
    cmd.add(epipolarArg);

    TCLAP::ValueArg<int> knnArg("k", "knn_matches", "number of matches to be kept (<= 0 --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
    cmd.add(knnArg);

    TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
    cmd.add(segNumArg);

    TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
    cmd.add(visibilityArg);

    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);

    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);

    TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
    cmd.add(collinArg);

    TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
    cmd.add(cudaArg);

    TCLAP::ValueArg<bool> ceresArg("c", "use_ceres", "use CERES (for 3D line optimization)", false, L3D_DEF_USE_CERES, "bool");
    cmd.add(ceresArg);

    TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
    cmd.add(constRegDepthArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string inputFolder = inputArg.getValue().c_str();
    std::string poseFolder = mavmapArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    std::string imgExtension = extArg.getValue().c_str();
    std::string imgPrefix = prefixArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = inputFolder+"/Line3D++/";

    int maxWidth = scaleArg.getValue();
    unsigned int neighbors = std::max(neighborArg.getValue(),2);
    bool diffusion = diffusionArg.getValue();
    bool loadAndStore = loadArg.getValue();
    float collinearity = collinArg.getValue();
    bool useGPU = cudaArg.getValue();
    bool useCERES = ceresArg.getValue();
    float epipolarOverlap = fmin(fabs(epipolarArg.getValue()),0.99f);
    float sigmaA = fabs(sigma_A_Arg.getValue());
    float sigmaP = sigma_P_Arg.getValue();
    int kNN = knnArg.getValue();
    unsigned int maxNumSegments = segNumArg.getValue();
    unsigned int visibility_t = visibilityArg.getValue();
    float constRegDepth = constRegDepthArg.getValue();

    if(imgExtension.substr(0,1) != ".")
        imgExtension = "."+imgExtension;

    // since no median depth can be computed without camera-to-worldpoint links
    // sigma_p MUST be positive and in pixels!
    if(sigmaP < L3D_EPS && constRegDepth < L3D_EPS)
    {
        std::cout << "sigma_p cannot be negative (i.e. in world coordiantes) when no valid regularization depth (--const_reg_depth) is given!" << std::endl;
        std::cout << "reverting to: sigma_p = 2.5px" << std::endl;
        sigmaP = 2.5f;
    }
    
    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,loadAndStore,maxWidth,
                                              maxNumSegments,false,useGPU);

    // read camera data (sequentially)
    // read images names
    std::vector<std::string> cams_filenames;
    int num_cams;
    {
      std::ifstream file((inputFolder + "/images.txt").c_str());
      if (!file.is_open())
      {
	std::cout << "error in reading pictures file" << std::endl;
	std::cout << (inputFolder + "/images.txt").c_str() << std::endl;
	return -1;
      }
      file >> num_cams;
      cams_filenames.resize(num_cams);
      std::string toto;
      for (int i=0; i<num_cams;i++){
	file >> cams_filenames[i] >> toto;
	std::cout << poseFolder + "/" + cams_filenames[i] + "_R.txt" << std::endl;
      }
    }
    
    // read internal parameters
    std::vector<std::pair<double,double> > cams_focals;
    std::vector<Eigen::Vector2d> cams_principle;
    {
      std::ifstream file((poseFolder + "/K.txt").c_str());
      if (!file.is_open())
      {
	std::cout << "error in reading K file" << std::endl;
	std::cout << (poseFolder + "/K.txt").c_str() << std::endl;
	return -1;
      }
      float fx, fy, cx, cy, temp;
      file >> fx >> temp >> cx;
      file >> temp >> fy >> cy;
      cams_focals = std::vector<std::pair<double,double> > (cams_filenames.size(), std::pair<double,double>(fx, fy));
      cams_principle = std::vector<Eigen::Vector2d> (cams_filenames.size(), Eigen::Vector2d(cx, cy));
    }
   
    // read global rotations and translations
    std::vector<Eigen::Matrix3d> cams_rotation(cams_filenames.size());
    std::vector<Eigen::Vector3d> cams_translation(cams_filenames.size());
    {
      for(int i = 0; i < cams_filenames.size(); i++){
	std::ifstream file_R((poseFolder + "/" + cams_filenames[i] + "_R.txt").c_str());
	std::cout << poseFolder + "/" + cams_filenames[i] + "_R.txt" << std::endl;
	if (!file_R.is_open())
	{
	  std::cout << "error in reading R_global file" << std::endl;
	  return -1;
	}
	for(int j = 0; j < 3; j++){
	  file_R >> cams_rotation[i](j,0) >> cams_rotation[i](j,1) >> cams_rotation[i](j,2);
	}
        std::cout << cams_rotation[i] << std::endl;

	std::ifstream file_t((poseFolder + "/" + cams_filenames[i] + "_C.txt").c_str());
	if (!file_t.is_open())
	{
	  std::cout << "error in reading t_global file" << std::endl;
	  return -1;
	}
	for(int j = 0; j < 3; j++){
	  file_t >> cams_translation[i][j];
	}
	cams_translation[i] = -cams_rotation[i]*cams_translation[i];
      }
    }
    
    // load images (parallel)
#ifdef L3DPP_OPENMP
    #pragma omp parallel for
#endif //L3DPP_OPENMP
    for(unsigned int i=0; i<cams_rotation.size(); ++i)
    {
        // load image
        std::string img_filename = inputFolder + "/" + cams_filenames[i] + ".png";
        // std::string img_filename = inputFolder + "/" + cams_filenames[i] + ".png";
        cv::Mat image;     

	// load image
	image = cv::imread(img_filename);

	// setup intrinsics
	double px = cams_principle[i].x();
	double py = cams_principle[i].y();
	double fx = cams_focals[i].first;
	double fy = cams_focals[i].second;

	Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
	K(0,0) = fx;
	K(1,1) = fy;
	K(0,2) = px;
	K(1,2) = py;
	K(2,2) = 1.0;

	// set neighbors
	std::list<unsigned int> neighbor_list;
	size_t n_before = neighbors/2;
	for(int nID=int(i)-1; nID >= 0 && neighbor_list.size()<n_before; --nID)
	{
	    neighbor_list.push_back(nID);
	}
	for(int nID=int(i)+1; nID < int(cams_rotation.size()) && neighbor_list.size() < neighbors; ++nID)
	{
	    neighbor_list.push_back(nID);
	}

	// add to system
	Line3D->addImage(i,image,K,cams_rotation[i],
			  cams_translation[i],constRegDepth,neighbor_list);
    }

    // match images
    Line3D->matchImages(sigmaP,sigmaA,neighbors,epipolarOverlap,
                        kNN,constRegDepth);

    // compute result
    Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useCERES);

    // save end result
    std::vector<L3DPP::FinalLine3D> result;
    Line3D->get3Dlines(result);

    // save as STL
    Line3D->saveResultAsSTL(outputFolder);
    // save as OBJ
    Line3D->saveResultAsOBJ(outputFolder);
    // save as TXT
    Line3D->save3DLinesAsTXT(outputFolder);

    // cleanup
    delete Line3D;
}
