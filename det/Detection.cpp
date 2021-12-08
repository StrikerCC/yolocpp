//
// Created by cheng on 12/6/21.
//

#include "Detection.h"

Detection::Detection(const std::string &python_dir_path, const std::string &python_module_name, const std::string &py_class_name) {
    /// set python project root path
    Py_SetPythonHome(L"/home/cheng/anaconda3/envs/yolov5");
    setenv("PYTHONPATH", python_dir_path.c_str(), 1);

    /// start python
    Py_Initialize();

    /// setup python path
    PyRun_SimpleString("import os, sys");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString(("sys.path.append('" + python_dir_path + "')").c_str());
    PyRun_SimpleString("print(sys.path)");

    /// python file
    auto pm = PyImport_ImportModule(python_module_name.c_str());
    if (pm == nullptr) {
        std::cout << "Can't find the python file!" << std::endl;
        return;
    }
    std::cout << "Find file " << python_dir_path << std::endl;

    /// python class
    this->detection_cls_from_py = PyObject_GetAttrString(pm, py_class_name.c_str());
    if (!this->detection_cls_from_py) {
        std::cout << "Can't find " << py_class_name << " in " << python_module_name << ".py at " << python_dir_path << std::endl;
        return;
    }
    std::cout << "Find class " << py_class_name << " in " << python_module_name << ".py at " << python_dir_path << std::endl;

    /// python object
    this->detector_from_py = PyObject_CallObject(this->detection_cls_from_py, NULL);
    if (!this->detector_from_py) {
        std::cout << "Can't instantiate "  << py_class_name << std::endl;
        return;
    }
    std::cout << "Instantiated "  << py_class_name << std::endl;
}

std::vector<std::vector<int>> Detection::detect(const cv::Mat &img) const {
    std::vector<std::vector<int>> detections_v;
    /// call python detection function
    auto result = PyObject_CallMethod(this->detector_from_py, this->detect_func_name_from_py.c_str(), "");
    PyErr_Print();

    if (! PyList_Check(result)) {
        std::cout << "No result " << std::endl;
    } else {
        std::cout << "Get result len " << PyList_Size(result) << std::endl;
        PyObject* detection_py;
        PyObject* item_py;

        // for each label: expect class_id, x, y, x, y as one label
        for (int i = 0; i < PyList_Size(result); i++) {
            std::cout << i << ": " << ": ";
            detection_py = PyList_GetItem(result, i);
            std::vector<int> detection_v;

            // for each number in label: expect class_id, x, y, x, y as one label
            for (int j = 0; j < /*PyList_Size(detection_py)*/ 4; j++) {
                item_py = PyList_GetItem(detection_py, j);
                int item_int = (int) PyLong_AsLong(item_py);
                std::cout << " " << item_int;
                detection_v.push_back(item_int);
            }
            std::cout << std::endl;
            detections_v.push_back(detection_v);
        }
        Py_DECREF(item_py);
        Py_DECREF(detection_py);
        std::cout << "Result reading done " << std::endl;
    }
    return detections_v;
}

Detection::~Detection() {
    Py_DECREF(this->detection_cls_from_py);
//    delete this->detection_cls_from_py;
    Py_DECREF(this->detector_from_py);
//    delete this->detector_from_py;

    /// end python
    Py_Finalize();
}
