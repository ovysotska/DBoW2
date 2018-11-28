/**
* partially taken from  Dorian Galvez-Lopez DBow2 demo
 */
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
vector<vector<cv::Mat>> loadFeatures(const std::string &path2images);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void matchFeatures(const vector<vector<cv::Mat>> &queryFeatures,
                   const vector<vector<cv::Mat>> &databaseFeatures,
                   const std::string &path2voc);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait() {
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

int main(int argc, char const *argv[]) {
  if (argc < 4) {
    std::cout << "[ERROR]: Not enough input parameters.";
    std::cout << "Proper usage: ./match_images path2queryFoler path2RefFolder "
                 "path2voc"
              << std::endl;
    return 0;
  }
  std::string path2queryImages = argv[1];
  std::string path2RefImages = argv[2];
  std::string vocName = argv[3];

  vector<vector<cv::Mat>> queryFeatures, refFeatures;
  queryFeatures = loadFeatures(path2queryImages);
  refFeatures = loadFeatures(path2RefImages);
  matchFeatures(queryFeatures, refFeatures, vocName);

  return 0;
}

// ----------------------------------------------------------------------------

vector<vector<cv::Mat>> loadFeatures(const std::string &path2images) {
  std::vector<std::string> image_names = listDir(path2images);
  const int NIMAGES = image_names.size();

  vector<vector<cv::Mat>> features;
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

void matchFeatures(const vector<vector<cv::Mat>> &queryFeatures,
                   const vector<vector<cv::Mat>> &databaseFeatures,
                   const std::string &path2voc) {
  cout << "Loading the vocabulary..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc(path2voc);

  OrbDatabase db(voc, false, 0);  // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (int i = 0; i < databaseFeatures.size(); i++) {
    db.add(databaseFeatures[i]);
  }
  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  vector<vector<double>> cost_matrix;
  for (int i = 0; i < queryFeatures.size(); i++) {
    db.query(queryFeatures[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    // cout << "Searching for Image " << i << ". " << ret << endl;
    std::vector<double> row(databaseFeatures.size());
    for (const auto res : ret) {
      // std::cout << res.Id << " " << res.Score << std::endl;
      row.at(res.Id) = res.Score;
    }
    cost_matrix.push_back(row);
  }
  cout << endl;

  std::cout << "Printing cost matrix to file: results.txt\n";

  std::ofstream out("results.txt");
  for (int i = 0; i < cost_matrix.size(); ++i) {
    for (int j = 0; j < cost_matrix[i].size(); ++j) {
      // std::cout << cost_matrix[i][j] << " ";
      out << cost_matrix[i][j] << " ";
    }
    // std::cout << std::endl;
    out << std::endl;
  }
  out.close();
  std::cout << "Results are printed\n";
}
