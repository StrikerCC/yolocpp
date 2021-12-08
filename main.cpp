#include <iostream>
#include "opencv4/opencv2/core.hpp"
#include <python3.8/Python.h>
#include <string>
#include "Detection.h"


int py_test();


int main() {
//    py_test();

    cv::Mat img;

    Detection det = Detection("/home/cheng/proj/det/yolocpp/python_scripts", "detect", "Yolo");
//    Detection det = Detection("/home/cheng/proj/det/yolocpp/python_scripts", "test", "yolo");
    std::cout << "det instantiated" << std::endl;

    auto result = det.detect(img);
    std::cout << "detection done: result is" << std::endl;
    for (const auto& label : result) {
        for (const auto& ele : label) {
            std::cout << ele << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    result = det.detect(img);
    std::cout << "detection done" << std::endl;

    result = det.detect(img);
    std::cout << "detection done" << std::endl;
    return 0;
}


int py_test() {
    /// init python object
    setenv("PYTHONPATH", "/home/cheng/proj/det/yolocpp", 1);
    Py_Initialize();
//    PyRun_SimpleString("print('Hello World Python!')");
//    PyRun_SimpleString("print('3 + 4 = ', 3+4)");

    /// setup python path
    PyRun_SimpleString("import os, sys");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("print(os.getcwd())");
    PyRun_SimpleString("print(sys.path)");

    PyRun_SimpleString("sys.path.append('/home/cheng/proj/det/yolov5')");

    /// python exe
    auto pm = PyImport_ImportModule("test");
//    auto pm = PyImport_ImportModule("test")

    if (pm == nullptr)
    {
        std::cout << "Can't find the python file!" << std::endl;
        return 0;
    }
    std::cout << "find file succeed " << std::endl;

    PyObject* pDict = PyModule_GetDict(pm); //获得Python模块中的函数列
    if (pDict == nullptr)
    {
        std::cout << "Can't find the dictionary!" << std::endl;
        return 0;
    }
    std::cout << "find dictionary succed" << std::endl;

    /// func
    auto fun_1 = PyObject_GetAttrString(pm, "multiply");
//    auto arg_1 = PyTuple_New(2);
//
//    PyTuple_SetItem(arg_1, 0, Py_BuildValue("i", 6));
//    PyTuple_SetItem(arg_1, 1, Py_BuildValue("i", 6));

    auto re_1 = PyObject_CallObject(fun_1, NULL);
    int result = (int) PyLong_AsLong(re_1);

//    auto fun = PyDict_GetItemString(pDict, "hello");
    std::cout << "func 1 succeed " << result << std::endl;
    std::cout << "##############################" << std::endl;


    auto cls = PyObject_GetAttrString(pm, "com");
//    auto cls = PyDict_GetItemString(pDict, "com");
    if (!cls) {
        std::cout << "can't find com " << std::endl;
    }
    auto ins = PyObject_CallObject(cls, NULL);
    if (!ins) {
        printf("不能找到 ins");
        return -1;
    }

    auto arg = PyTuple_New(2);
    PyTuple_SetItem(arg, 0, Py_BuildValue("i", 6));
    PyTuple_SetItem(arg, 1, Py_BuildValue("i", 6));

    std::cout << "args done" << std::endl;

    auto re = PyObject_CallMethod(ins, "add", "");
    if (!re)
    {
        printf("不能找到 pRet");
        return -1;
    }

    result = (int) PyLong_AsLong(re);

    std::cout << "call done " << result << std::endl;

    /// end python
    Py_Finalize();
}