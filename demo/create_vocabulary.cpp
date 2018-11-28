#include <iostream>
#include <string>
#include <vector>

// DBoW2
#include "DBoW2.h"  // defines OrbVocabulary and OrbDatabase
#include "list_dir.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

vector<vector<cv::Mat> > loadFeatures(const std::string &path2images);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void createVoc(const vector<vector<cv::Mat> > &features,
               const std::string &vocName);

// ----------------------------------------------------------------------------

int main(int argc, char const *argv[]) {
  if (argc < 3) {
    std::cout << "ERROR: Not enough input parameters.";
    std::cout << "Proper usage: ./create_vocabulary path2folder "
                 "voc_name.yml.gz"
              << std::endl;
    return 0;
  }
  std::string path2images = argv[1];
  std::string outVocName = argv[2];

  vector<vector<cv::Mat> > features;
  features = loadFeatures(path2images);

  createVoc(features, outVocName);
  return 0;
}

// ----------------------------------------------------------------------------

vector<vector<cv::Mat> > loadFeatures(const std::string &path2images) {
  std::vector<std::string> image_names = listDir(path2images);
  const int NIMAGES = image_names.size();

  vector<vector<cv::Mat> > features;
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for (int i = 0; i < NIMAGES; ++i) {
    cv::Mat image = cv::imread(image_names[i], 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat>());
    changeStructure(descriptors, features.back());
  }
  return features;
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) {
  out.resize(plain.rows);

  for (int i = 0; i < plain.rows; ++i) {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void createVoc(const vector<vector<cv::Mat> > &features,
               const std::string &vocName) {
  // branching factor and depth levels
  // const int k = 9;
  // const int L = 3;
  // parameters from the paper
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  OrbVocabulary voc(k, L, weight, score);

  cout << "Creating a vocabulary " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl << voc << endl << endl;

  // // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for (int i = 0; i < features.size(); i++) {
  //   voc.transform(features[i], v1);
  //   for (int j = 0; j < features.size(); j++) {
  //     voc.transform(features[j], v2);

  //     double score = voc.score(v1, v2);
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save(vocName);
  cout << "Done" << endl;
}
