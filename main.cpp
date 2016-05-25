#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#include "xnn.h"
#include "profile.h"

using namespace nn;

Mat viz(const Volume &v, int patchSize)
{
    PROFILEX("viz_vol")
    int n = (int)sqrt(double(v.size()*2));
    //cerr << v.size() << " " << patchSize << " " << n << " " << (n*patchSize*n*patchSize) << endl;
    Mat draw(n*patchSize, n*patchSize, CV_32F, 0.0f);
    for (size_t i=0; i<v.size(); i++)
    {
        Mat m = v[i].getMat(ACCESS_READ).reshape(1,patchSize);
        int r = patchSize * (i / n);
        int c = patchSize * (i % n);
        m.copyTo(draw(Rect(c,r,patchSize,patchSize)));
    }
    return draw;
}
Mat viz(const UMat &weights)
{
    PROFILEX("viz_weights")
    int pn = (int)sqrt(double(weights.cols)) + 1;
    int ps = (int)sqrt(double(weights.rows));
    Mat draw(pn*ps+2,pn*ps+2,CV_32F,0.0f);
    for (int i=0; i<weights.cols; i++)
    {
        Mat f = weights.getMat(ACCESS_READ).col(i).clone().reshape(1,ps);
        int r = ps * int(i / pn);
        int c = ps * int(i % pn);
        f.copyTo(draw(Rect(c,r,ps,ps)));
    }
    return draw;
}
float accuracy(const Volume &labels, const Volume &predicted)
{
    float acc = 0.0f;
    for (size_t i=0; i<predicted.size(); i++)
    {
        UMat r1,r2,same;
        labels[i].convertTo(r1,CV_32S);
        predicted[i].convertTo(r2,CV_32S);
        compare(r1,r2,same,CMP_EQ);
        int ok = countNonZero(same);
        acc += float(ok) / r1.total();
    }
    return acc;
}

int main(int argc, char **argv)
{

    PROFILE;

    const char *keys =
            "{ help h usage ? |     | show this message }"
            "{ ocl o          |     | toggle usage of ocl }"
            "{ gen g          |1000 | number of train generations }"
            "{ problem p      |mnist| input problem(att,mnist,digits,numbers,tv10) }"
            "{ network n      |nn/mnist_3.xml| preconfigured network to load }"
            ;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }
    String network(parser.get<String>("network"));
    String prob(parser.get<String>("problem"));
    int ngen(parser.get<int>("gen"));
    bool useOCL(parser.get<bool>("ocl"));

    ocl::setUseOpenCL(useOCL);
    cout << "ocl " << cv::ocl::useOpenCL() << endl;
    //theRNG().state = getTickCount();

    Ptr<Problem> problem = nn::createProblem(prob);
    cout << problem->desc() << endl;

    Ptr<Network> nn = nn::createNetwork(network);
    cout << nn->desc() << endl;
    
    for (int g=1; g<=ngen; g++)
    {
        PROFILEX("generation")
        Volume data, res,res1,labels;
        {
            PROFILEX("problem.train")
            problem->train(32,data,labels);
        }
        float e1=0,e2=0,x0=0;
        {
            PROFILEX("train_pass")
            x0 = nn->forward(data,res,true);
            e2 = nn->backward(res1,labels);
        }
        
        if (g%50==0)
        {
            PROFILEX("report")
            float acc = accuracy(labels, res);
            cout << g << " " << e2 << " " << acc / res.size() << endl;
            if (data[0].cols>7)
            {
                imshow("input", viz(data, problem->inputSize().width));
                imshow("back", viz(res1, problem->inputSize().width));
                //nn->show();
            }
            if (waitKey(50)==27) return 0;
            
        }
    }  

    Volume data,predicted,labels;
    problem->test(100,data,labels);
   
    nn->forward(data,predicted,false);
    
    float acc = accuracy(labels, predicted);
    cout << "final acc : " << acc / predicted.size() << endl;
    
    nn->save("my.xml");
    return 0;
}  