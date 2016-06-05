#ifndef __XNN_onboard__
#define __XNN_onboard__

#include "opencv2/opencv.hpp"

namespace nn {

typedef std::vector<cv::UMat> Volume;

struct Problem
{
    virtual void train(int n, Volume &data, Volume &labels) = 0;
    virtual void test(int n, Volume &data, Volume &labels) = 0;
    virtual Size inputSize() = 0; 
    virtual Size outputSize() = 0;
    virtual cv::String desc() = 0;
};

struct Layer
{
    virtual float forward(const Volume &upstream, Volume &downstream, bool training) = 0;
    virtual float backward(Volume &upstream, const Volume &downstream, bool training) = 0;
    virtual bool write(cv::FileStorage &fs) { return false; }
    virtual bool read(const cv::FileNode &fn) { return false; }    
    virtual cv::String desc() { return "generic"; }
    virtual cv::String type() { return "generic"; }
    virtual void show(String winName) {}
};

struct Network
{
    virtual float forward(const Volume &up, Volume &dn, bool training) = 0;
    virtual float backward(Volume &up, const Volume &dn, bool training) = 0;
    virtual bool save(cv::String fn) = 0;
    virtual bool load(cv::String fn) = 0;
    virtual cv::String desc() = 0;
    virtual void show()  = 0;
};

cv::Ptr<Problem> createProblem(cv::String name);
cv::Ptr<Network> createNetwork(cv::String name);

Mat viz(const Volume &v, int patchSize);
Mat viz(const UMat &weights);

} // namespace nn;

#endif // __XNN_onboard__
