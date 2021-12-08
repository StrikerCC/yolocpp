//
// Created by cheng on 12/6/21.
//

#ifndef YOLOCPP_DETECTION_H
#define YOLOCPP_DETECTION_H
#include <iostream>
#include "opencv2/core.hpp"
#include <python3.8/Python.h>

class Detection {
public:
    Detection(const std::string &python_dir_path, const std::string &python_module_name, const std::string &py_class_name);
    ~Detection();

public:
    std::vector<std::vector<int>> detect(const cv::Mat &img) const;


private:
    PyObject * detection_cls_from_py = nullptr;
    PyObject* detector_from_py = nullptr;
    std::string detect_func_name_from_py = "detect";
};


#endif //YOLOCPP_DETECTION_H
