#ifndef POINT2FCOMPARE
#define POINT2FCOMPARE

#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

// struct Point2fCompare
// {
//    bool operator <(const cv::Point2f& lhs, const cv::Point2f& rhs) const
//    {
//        cout << "comparing:" << endl;
//        cout << lhs << ", " << rhs << endl;
//        return (lhs.x == rhs.x && lhs.y == rhs.y);
//    }
// };

namespace std
{
    template<> struct  less<cv::Point2f>
    {
        bool operator() (const cv::Point2f& lhs, const cv::Point2f& rhs) const
        {
            return lhs.x < rhs.x || lhs.x == rhs.x && (lhs.y < rhs.y);
        }
    };

}

#endif