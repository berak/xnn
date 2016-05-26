#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#include "xnn.h"
#include "profile.h"

using namespace nn;

float accuracy(const Volume &labels, const Volume &predicted)
{
    int ok = 0;
    for (size_t i=0; i<predicted.size(); i++)
    {
        Point a,b;
        minMaxLoc(labels[i], 0, 0, 0, &a);
        minMaxLoc(predicted[i], 0, 0, 0, &b);
        ok += (a.x == b.x);
    }
    float acc = float(ok) / predicted.size();
    return acc;
}

int main(int argc, char **argv)
{

    PROFILE;

    const char *keys =
            "{ help h usage ? |     | show this message }"
            "{ ocl o          |     | toggle ocl usage (0,1) }"
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
    int useOCL(parser.get<int>("ocl"));

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
            PROFILEX("train_problem")
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
            cout << g << " " << e2 << " " << acc << endl;
            if (data[0].cols>7)
            {
                imshow("input", viz(data, problem->inputSize().width));
                imshow("back", viz(res1, problem->inputSize().width));
                nn->show();
            }
            if (waitKey(50)==27) return 0;
            
        }
    }  

    Volume data,predicted,labels;
    problem->test(100,data,labels);
   
    nn->forward(data,predicted,false);
    
    float acc = accuracy(labels, predicted);
    cout << "final acc : " << acc << endl;
    
    nn->save("my.xml");
    return 0;
}