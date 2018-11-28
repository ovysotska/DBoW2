
#include "list_dir.h"

#include <dirent.h>
#include <algorithm>
#include <iostream>

std::vector<std::string> listDir(const std::string &dir_name) {
  std::vector<std::string> file_names;
  DIR *dir;
  if ((dir = opendir(dir_name.c_str())) != NULL) {
    struct dirent *ent;
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      file_names.push_back( dir_name + ent->d_name);
    }
    std::sort(file_names.begin(), file_names.end());
    closedir(dir);
    // deleting "." and ".." files from consideration.
    file_names.erase(file_names.begin());
    file_names.erase(file_names.begin());
  } else {
    /* could not open directory */
    printf("[ERROR][List_dir] The directory could not be opened %s\n",
           dir_name.c_str());
    exit(EXIT_FAILURE);
  }
  return file_names;
}