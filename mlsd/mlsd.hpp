/*----------------------------------------------------------------------------  
  This code is part of the following publication and was subject
  to peer review:
  "Multiscale line segment detector for robust and accurate SfM" by
  Yohann Salaun, Renaud Marlet, and Pascal Monasse
  ICPR 2016
  
  Copyright (c) 2016 Yohann Salaun <yohann.salaun@imagine.enpc.fr>
  
  This program is free software: you can redistribute it and/or modify
  it under the terms of the Mozilla Public License as
  published by the Mozilla Foundation, either version 2.0 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  Mozilla Public License for more details.
  
  You should have received a copy of the Mozilla Public License
  along with this program. If not, see <https://www.mozilla.org/en-US/MPL/2.0/>.

  ----------------------------------------------------------------------------*/

#ifndef MLSD_HPP
#define MLSD_HPP

//#include "interface.hpp"
#include "lsd.hpp"
#include <cmath>
#include <vector>
#include <set>
#include <fstream>


// OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "openMVG/numeric/numeric.h"

using namespace std;

// descriptors parameters
const int widthOfBand_ = 7;
const int numOfBand_ = 9;
const short descriptorSize = numOfBand_ * 8;

// parameters for pyramid of images
const double prec = 3.0;
const double sigma_scale = 0.8;
const double scale_step = 2.0;
const int h_kernel = (unsigned int)ceil(sigma_scale * sqrt(2.0 * prec * log(10.0)));



// strucuture to keep track of segment detected
struct Segment{
	// segment coordinates
	double x1, y1, x2, y2;

	// segment geometric attributes
	double width, length, angle;

	// NFA related arguments
	double log_nfa, prec;

	// scale of last detection from 0 (rawest) to n (finest)
	int scale;

	// descriptor
	std::vector<float> descriptor;

	// used for matching
	openMVG::Vec2 m;
	openMVG::Vec3 line, p1, p2;
	int vpIdx;

	// used for coplanar constraints
	std::vector<int> planes, lines3D, coplanar_cts;
	openMVG::Vec3 homogenous_line;

	Segment(){};
	Segment(const double X1, const double Y1, const double X2, const double Y2,
			const double w, const double p, const double nfa, const double s);

	// CLUSTERING METHOD
	bool isSame(int &ptr_l3D, const double angle_thresh, const double dist_thresh, const void* l3D) const;

	// FOR CALIBRATION/RECONSTRUCTION
	void normalize(const openMVG::Mat3 &K, const openMVG::Mat3 &Kinv);

	// FOR MULTISCALE LSD
	void upscale(const double k);

	// DISTANCE METHODS
	double distTo(const openMVG::Vec2 &p) const;
	double distTo(const Segment &s) const;

	// I/O METHODS for segments
	void readSegment(std::ifstream &file);
	void saveSegment(std::ofstream &file) const;

	// I/O METHODS for descriptors
	void readDescriptor(std::ifstream &file);
	void saveDescriptor(std::ofstream &file) const;

	double qlength();
	openMVG::Vec2 center();
	openMVG::Vec3 equation();
};

typedef std::vector<Segment> PictureSegments;
typedef std::vector<PictureSegments> PicturesSegments;


struct Line3D{
    // 3D extremities
    openMVG::Vec3 p1, p2;

    // for plane clustering
    std::vector<int> proj_ids, cam_ids;
    openMVG::Vec3 direction;
    std::set<int> planes, cop_cts;

    // for translation norm computation only
    openMVG::Vec3 mid;
    double lambda;

    Line3D(const openMVG::Vec3 &P1, openMVG::Vec3 &P2){
        p1 = P1;
        p2 = P2;
        direction = (p1-p2).normalized();
        mid = 0.5*(p1 + p2);
    }

    // compute 3D line up to the translation scale
    Line3D(const Segment &l, const Segment &m, const openMVG::Vec3 &d,
           const openMVG::Mat3 &R1, const openMVG::Mat3 &R2, const openMVG::Vec3 &t12,
           const int li, const int mi, const int i1, const int i2, const bool invert){
        proj_ids.push_back(li);
        proj_ids.push_back(mi);

        cam_ids.push_back(i1);
        cam_ids.push_back(i2);

        direction = d.normalized();

        mid = (invert)?  0.5*(m.p1 + m.p2) : 0.5*(l.p1 + l.p2);
        openMVG::Vec3 t = (invert)? -R1*R2.transpose()*t12 : t12;
        lambda = (invert)? t.dot(l.line)/((R2.transpose()*mid).dot(R1.transpose()*l.line))
                         : t.dot(m.line)/((R1.transpose()*mid).dot(R2.transpose()*m.line));

        p1 = -t12.dot(m.line)/((R1.transpose()*l.p1).dot(R2.transpose()*m.line))*R1.transpose()*l.p1;
        p2 = -t12.dot(m.line)/((R1.transpose()*l.p2).dot(R2.transpose()*m.line))*R1.transpose()*l.p2;
    }

    // only for t_norm method
    double distTo(const Line3D &l, const PicturesSegments &lines) const{
        return lines[cam_ids[1]][proj_ids[1]].distTo(lines[l.cam_ids[0]][l.proj_ids[0]]);
    }

    // general distance function
    double distTo(const Line3D &l) const{
        return std::min(std::min((p1-l.p1).norm(), (p2-l.p1).norm()),
                        std::min((p1-l.p2).norm(), (p2-l.p2).norm()));
    }

    void addProjection(const int i_cam, const int i_proj){
        proj_ids.push_back(i_proj);
        cam_ids.push_back(i_cam);
    }

    void addCopCts(Segment &s){
        for(int i = 0; i < s.coplanar_cts.size(); i++){
            cop_cts.insert(s.coplanar_cts[i]);
        }
    }

    bool isEqualUpTo(const Line3D &l, const double angle_thresh, const double dist_thresh) const{
        if(fabs(direction.dot(l.direction)) > angle_thresh){return false;}
        if(distTo(l) > dist_thresh){return false;}
        return true;
    }
};

class NFA_params{
  double value, theta, prec_divided_by_pi, prec;
  bool computed;
public:
  NFA_params();
  NFA_params(const double t, const double p);
  NFA_params(const NFA_params &n);

  rect regionToRect(vector<point> &data, image_double modgrad);
  void computeNFA(rect &rec, image_double angles, const double logNT);

  double getValue() const;
  double getTheta() const;
  double getPrec() const;
};

class Cluster{
  // pixels parameters
  vector<point> data;
  rect rec;

  // NFA parameters
  NFA_params nfa;
  double nfa_separated_clusters;

  // fusion parameters
  int index;
  bool merged;
  int scale;

public:

  Cluster();
  Cluster(image_double angles, image_double modgrad, const double logNT,
    vector<point> &d, const double t, const double p, const int i, const int s, const double n);
  Cluster(image_double angles, image_double modgrad, const double logNT,
    point* d, const int dsize, rect &r, const int i, const int s);
  
  Cluster mergedCluster(const vector<Cluster> &clusters, const set<int> &indexToMerge,
    image_double angles, image_double modgrad, const double logNT) const;
  bool isToMerge(image_double angles, const double logNT);
  
  double length() const;
  Segment toSegment();
  
  bool isMerged() const;
  void setMerged();
  double getNFA() const;
  int getIndex() const;
  int getScale() const;
  double getTheta() const;
  double getPrec() const;
  cv::Point2d getCenter() const;
  cv::Point2d getSlope() const;
  const vector<point>* getData() const;
  void setUsed(image_char used) const;
  void setIndex(int i);
};

// generate the number of scales needed for a given image
int scaleNb(const cv::Mat &im, const bool multiscale);

// filter some area in the image for better SfM results and a faster line detection
void denseGradientFilter(vector<int> &noisyTexture, const cv::Mat &im, 
			 const image_double &angles, const image_char &used,
			 const int xsize, const int ysize, const int N);

// compute the segments at current scale with information from previous scale
vector<Cluster> refineRawSegments(const vector<Segment> &rawSegments, vector<Segment> &finalLines, const int i_scale,
				  image_double angles, image_double modgrad, image_char used,
				  const double logNT, const double log_eps);

// merge segments at same scale that belong to the same line
void mergeSegments(vector<Cluster> &refinedLines, const double segment_length_threshold, const int i_scale,
		   image_double angles, image_double modgrad, image_char &used,
		   const double logNT, const double log_eps);

#endif /* !MULTISCALE_LSD_HEADER */
/*----------------------------------------------------------------------------*/
