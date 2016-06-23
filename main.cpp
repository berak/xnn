#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#include "xnn.h"
#include "profile.h"

using namespace nn;

// mse
float loss(const Volume &a, const Volume &b)
{
    PROFILE;
    UMat s(a[0].reshape(1,1).size(), CV_32F, 0.0f);
    for (size_t i=0; i<a.size(); i++)
    {
        UMat A = a[i].reshape(1,1);
        UMat B = b[i].reshape(1,1);
        UMat c;
        subtract(A, B, c);
        multiply(c, c, c);
        add(s, c, s);
    }
    sqrt(s,s);
    Mat sum_cpu; s.copyTo(sum_cpu);
    return sum(sum_cpu)[0] / a.size();
}

float accuracy(const Volume &labels, const Volume &predicted)
{
    PROFILE;
    int ok = 0;
    for (size_t i=0; i<predicted.size(); i++)
    {
        Point a,b;
        minMaxLoc(labels[i],    0, 0, 0, &a);
        minMaxLoc(predicted[i], 0, 0, 0, &b);
        ok += (a.x == b.x);
    }
    return float(ok) / predicted.size();
}

int main(int argc, char **argv)
{

    PROFILE;
    //theRNG().state = getTickCount();

    const char *keys =
            "{ help h usage ? |     | show this message }"
            "{ ocl o          |     | toggle ocl usage (0,1) }"
            "{ gen g          |1000 | number of train generations }"
            "{ report r       |50   | report frequency }"
            "{ batch b        |32   | train batch size }"
            "{ loss L         |0.05 | stop training if loss is less }"
            "{ learn l        |0.0001| global learning rate (overrides layer property) }"
            "{ problem p      |mnist| input problem(eyes,att,cifar10,mnist,digits,numbers,spiral,tv10) }"
            "{ network n      |nn/mnist_3.xml| preconfigured network to load }"
            "{ save s         |my.xml| filename of saved trained model }"
            "{ visual v       |1    | show visuals in gui(or save to file) }"
            ;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }
    String network(parser.get<String>("network"));
    String prob(parser.get<String>("problem"));
    String saveFile(parser.get<String>("save"));
    int ngen(parser.get<int>("gen"));
    int report(parser.get<int>("report"));
    int useOCL(parser.get<int>("ocl"));
    int batchSize(parser.get<int>("batch"));
    float minLoss(parser.get<float>("loss"));
    float globLearn(parser.get<float>("learn"));
    bool visual(parser.get<int>("visual"));
    ocl::setUseOpenCL(useOCL);
    cout << "ocl " << cv::ocl::useOpenCL() << endl;

    Ptr<Problem> problem = nn::createProblem(prob);
    cout << problem->desc() << endl;

    Ptr<Network> nn = nn::createNetwork(network);
    cout << nn->desc() << endl;

    for (int g=1; g<=ngen; g++)
    {
        PROFILEX("generation")
        Volume data, res, res1, labels;
        {
            PROFILEX("train_problem")
            problem->train(batchSize, data, labels);
        }
        {
            PROFILEX("train_pass")
            nn->forward(data, res, true);
            nn->backward(res1, labels, true, globLearn);
        }

        if (g % report == 0)
        {
            PROFILEX("report")
            float acc = accuracy(labels, res);
            float loss_fw = loss(labels, res);
            float loss_bw = loss(data, res1);
            cout << format("%-5d  %3.3f  %3.3f  %3.6f", g, loss_fw, loss_bw, acc) << endl;
            if (data[0].cols>7)
            {
                if (visual)
                {
                    imshow("input", viz(data, problem->inputSize().width));
                    imshow("back",  viz(res1, problem->inputSize().width));
                    nn->show();
                }
                else
                {
                    Mat gr = viz(data, problem->inputSize().width);
                    gr.convertTo(gr,CV_8U,255);
                    imwrite("img/input.png", gr);

                    gr = viz(res1, problem->inputSize().width);
                    gr.convertTo(gr,CV_8U,255);
                    imwrite("img/back.png", gr);
                }
            }
            if (visual && waitKey(50)==27) return 0;
            if (loss_fw < minLoss) break;
        }
    }

    Volume data,predicted,labels,res1;
    problem->test(200, data, labels);

    nn->forward(data, predicted, false);
    float loss_fw = loss(labels, predicted);

    nn->backward(res1, labels, false);
    float loss_bw = loss(res1, data);

    float acc = accuracy(labels, predicted);
    cout << "final loss : " << loss_fw << " : " << loss_bw << " acc :  " << acc << endl;

    nn->save(saveFile);
    return 0;
}
