#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
#include <deque>
using namespace std;

#include "xnn.h"
#include "profile.h"

namespace nn {
namespace impl {

int  DBGLEV = 0;


typedef UMat (*proc)(const UMat &);

#include "activation.cpp"
#include "optimizer.cpp"

String info(const Volume &v)
{
    if (v.empty()) return "[]";
    return format("[%dx%dx%d]",v[0].rows,v[0].cols,v.size());
}
static UMat col(const UMat &m)
{
    PROFILE;
    return m.reshape(1, m.total());
}
static UMat row(const UMat &m)
{
    PROFILE;
    return m.reshape(1, 1);
}

static UMat mse(const UMat &a, const UMat &b)
{
    PROFILEX("mse");
    UMat c, res;
    subtract(a, b, c);
    multiply(c, c, res);
    return res;
}

static UMat num_grad(const UMat &x, const UMat &W, float h=0.00001f)
{
    PROFILEX("num_grad");
    UMat x1,x2, y1,y2, res;
    UMat xr = row(x);
    subtract(xr, h, x1);
    subtract(xr,-h, x2);
    gemm(x1, W, 1, noArray(), 0, y1);
    gemm(x2, W, 1, noArray(), 0, y2);
    subtract(y1, y2, res);
    divide(res, 2 * h, res);
    return res;
}

static double grad_check(const UMat &fa, const UMat &fn)
{
    PROFILEX("grad_check");
    int nrm = NORM_L2;
    double eps = 0.00001;
    UMat pa;
    reduce(fa, pa, 0, REDUCE_AVG);
    double a = norm(pa, nrm);
    double n = norm(fn, nrm);
    if (a<eps || n<eps) return -1;
    double c = norm(pa, fn, nrm);
    //cerr << "gc " << a << " " << n << " " << c << " " << (c / max(a, n)) << endl;;
    return c / max(a, n);
}

static UMat sqr(const UMat &a)
{
    PROFILEX("sqr");
    UMat res;
    multiply(a, a, res);
    return res;
}

struct LossLinear
{
    UMat operator()(const UMat &predicted, const UMat &truth) const
    {
        PROFILEX("loss_linear");
        UMat res;
        subtract(predicted, truth, res);
        return res;
    }
};

struct LossSoftmax
{
    UMat operator()(const UMat &predicted, const UMat &truth) const
    {
        PROFILEX("loss_softmax");
        UMat prob, mask, res;

        exp(predicted, prob);
        Scalar total = sum(prob);
        divide(prob, total[0], prob);

        truth.convertTo(mask, CV_8U);//, 1.0/255);
        subtract(prob, -1, res, mask);
        return res;
    }
};

struct LossMseSqr
{
    UMat operator()(const UMat &predicted, const UMat &truth) const
    {
        PROFILEX("loss_mse");
        return mse(row(predicted), row(truth));
    }
};


struct BaseWeights :  Layer
{
    UMat weights, weights_t, bias, grad, gradB, gradn;
    Volume cache_up, cache_dn;
    float learn, weight_init;
    String name;

    BaseWeights(String n="fully"): name(n) {}

    virtual bool write(FileStorage &fs)
    {
        Mat w = weights.getMat(ACCESS_READ);
        Mat b = bias.getMat(ACCESS_READ);
        fs << "size" << weights.size();
        fs << "weights" << w;
        fs << "bias" << b;
        fs << "learn" << learn;
        fs << "weight_init" << weight_init;
        return true;
    }
    virtual bool read(const FileNode &fn)
    {
        Size siz;
        fn["size"] >> siz;
        Mat w,b;
        fn["weights"] >> w;
        fn["bias"] >> b;
        weight_init = 0.5f;
        fn["weight_init"] >> weight_init;
        if (w.empty() && siz.area())
        {
            weights = rand(siz.height, siz.width, weight_init);
            bias = UMat(siz.width, 1, CV_32F, 0.0f);
        }
        else
        {
            w.copyTo(weights);
            b.copyTo(bias);
        }
        weights_t = weights.t();
        fn["learn"] >> learn;
        return true;
    }
    virtual String type() { return name; }
    virtual String desc() { return format("%s(%d,%d,%1.3f,%1.3f)",name.c_str(), weights.cols, weights.rows, learn, weight_init); }
    double nrm(const UMat &m)
    {
        return sqrt(sum(sqr(weights))[0]);
    }
    String check()
    {
        double s1=nrm(weights);
        double s2=nrm(grad) * learn;
        double s3=nrm(bias);
        double s4=nrm(gradB) * learn;
        double err = grad_check(grad, gradn);
        return format("W(%5.4f) dW(%5.4f) dW/W(%5.4f) B(%5.4f) dB(%5.4f) err(%5.8f)", s1, s2, (s2/s1), s3, s4, err);
    }
};


template <typename Optimizer, typename Loss>
struct Fully : public BaseWeights
{
    Optimizer optim_w, optim_b;
    Loss loss;

    Fully(String n="fully")
        : BaseWeights(n)
        , optim_w("opt_w_")
        , optim_b("opt_b_")
    {}


    virtual bool write(FileStorage &fs)
    {
        BaseWeights::write(fs);
        optim_w.write(fs);
        optim_b.write(fs);
        return true;
    }

    virtual bool read(const FileNode &fn)
    {
        BaseWeights::read(fn);
        optim_w.read(fn);
        optim_b.read(fn);
        return true;
    }

    virtual void show(String winName)
    {
        //cerr << name << " " << check() << endl;
        namedWindow(winName);
        imshow(winName,viz(weights));
        namedWindow(winName+"_grad");
        imshow(winName+"_grad",viz(grad));
    }

    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        PROFILEX("fully_forward");
        if (DBGLEV) cerr << name << "_fw \t" << info(upstream);
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            // dn = up * weights + bias
            UMat up = row(upstream[i]);
            gemm(up, weights, 1, bias, 1, downstream[i], GEMM_3_T);
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
            gradn = num_grad(upstream[0], weights);
        }
        if (DBGLEV) cerr << " " << info(downstream) << endl;
        return 0;
    }

    void update(const UMat &cache_u, const UMat &residual)
    {
        // dx = c.t() * res;
        // grad += dx;
        UMat dx, c = col(cache_u);
        //if (DBGLEV) cerr << name << "_up " << c.size() << "  " << residual.size();

        gemm(c, residual, 1, noArray(), 0, dx);
        add(grad, dx, grad);
        //if (DBGLEV) cerr << name << " " << grad.size() << "  " << dx.size() << endl;

        // bias, too.
        UMat db;
        reduce(dx, db, 0, REDUCE_SUM);
        add(gradB, col(db), gradB);
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        PROFILEX("fully_backward");
        if (DBGLEV) cerr << name << "_bw \t" << info(downstream);
        upstream.resize(downstream.size());
        grad = UMat(weights.size(), weights.type(), 0.0f);
        gradB = UMat(bias.size(), bias.type(), 0.0f);
        //#pragma omp parallel for
        for (size_t i=0; i<downstream.size(); i++)
        {
            // up = dn * wt;
            UMat dn = downstream[i];
            gemm(dn, weights_t, 1, noArray(), 0, upstream[i]); // bias ?
            if (! training)
                continue;

            // residual = predicted - truth
            UMat pred = cache_dn[i];
            UMat res = loss(pred, dn);
            update(cache_up[i], res);
        }
        if (training)
        {
            // grad /= downstream.size();
            // weights -= grad * learn;
            float lrn = (globLearn > 0) ? globLearn : learn;
            optim_w(grad, weights, lrn/downstream.size());
            optim_b(gradB, bias, lrn/downstream.size());
            weights_t = weights.t();
        }
        if (DBGLEV) cerr << " "  << info(upstream) << endl;

        return 0.0f;
    }

};

template <typename Optimizer,typename Loss>
struct Rnn : BaseWeights
{
    Optimizer optim_w, optim_u;
    Loss loss;
    deque<UMat> past;
    proc act_fw, act_bw;
    UMat U, U_t,gradU;
    int hidden;

    Rnn(String n="recurrent")
        : BaseWeights(n)
        , optim_w("opt_w_")
        , optim_u("opt_u_")
        , act_fw(tanh2_fw)
        , act_bw(tanh2_bw)
        , hidden(3)
    {}

    static UMat signal(const UMat &W, const UMat &U, const UMat &x, const UMat &last, proc act)
    {
        // act(last*U + x*W);
        // cerr << W.size() << " " << U.size() << " " << x.size() << " " << last.size() << " " ;
        UMat r;
        gemm(x, W, 1, noArray(), 0, r);
        if (! last.empty())
        {
            UMat r2;
            gemm(last, U, 1, r, 1, r2);
            r = r2;
        }
        return act(r);
    }

    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        PROFILEX("rnn_forward");
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
        {
            // dn = up * weights + bias
            UMat up = upstream[i].reshape(1,1);
            if (past.size() >= hidden)
            {
                past.pop_front();
            }
            past.push_back(up);
            UMat dn; // empty for the most left(oldest)
            for (size_t h=0; h<past.size(); h++)
            {
                dn = signal(weights, U, past[h], dn, act_fw);
            }
            downstream[i] = dn;
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
        //cerr << "fw " << past.size() << "  " << downstream.size() << " " << training << endl;
        return 0;
    }

    void update(const UMat &cache_u, const UMat &residual)
    {
        // dx = c.t() * res;
        // grad += dx;
        UMat dx, c = col(cache_u);
        gemm(c, residual, 1, noArray(), 0, dx);
        add(grad, dx, grad);
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        //cerr << "bw " << downstream.size() << " " << training << endl;
        PROFILEX("rnn_backward");
        if (training)
        {
            grad = UMat(weights.size(), weights.type(), 0.0f);
            gradB = UMat(bias.size(), bias.type(), 0.0f);
            gradU = UMat(U.size(), U.type(), 0.0f);
        }

        upstream.resize(downstream.size());
        for (size_t i=0; i<downstream.size(); i++)
        {
            // up = dn * wt;
            UMat up, dn = downstream[i];
            gemm(dn, weights_t, 1, noArray(), 0, up);
            upstream[i] = up;
            if (! training)
                continue;

            // residual = predicted - truth
            UMat pred = cache_dn[i];
            UMat res = loss(pred, dn);
            update(cache_up[i], res);

            UMat dU;
            gemm(res, U_t, 1, noArray(), 0, dU);
            for (size_t h=past.size()-1; h>0; h--)
            {
                UMat sig;
                gemm(past[h], weights, 1, dU, 1, sig);
                add(act_bw(sig), dU, dU);;
            }

            gemm(col(dU), pred, 1/hidden, gradU, 1, gradU);
        }
        if (! training)
            return 0.0f;


        // grad /= downstream.size();
        // weights -= grad * learn;
        float lrn = (globLearn > 0) ? globLearn : learn;
        optim_w(grad, weights, lrn/downstream.size());
        weights_t = weights.t();
        optim_u(gradU, U, lrn/downstream.size());
        U_t = U.t();

        return 0.0f;
    }

    String check()
    {
        double s1=nrm(U);
        double s2=nrm(gradU) * learn;
        String s = BaseWeights::check();
        return s + format(" U(%5.4f) dU(%5.4f) dU/U(%5.4f)", s1,s2,(s2/s1));
    }
    virtual void show(String winName)
    {
        //cerr << name << " " << check() << endl;
        namedWindow(winName);
        imshow(winName,viz(weights));
        namedWindow(winName+"_g");
        imshow(winName+"_g",viz(grad));
        namedWindow(winName+"U");
        imshow(winName+"U",viz(U));
        namedWindow(winName+"dU");
        imshow(winName+"dU",viz(gradU));
    }
    virtual bool write(FileStorage &fs)
    {
        BaseWeights::write(fs);
        Mat u = U.getMat(ACCESS_READ);
        fs << "U" << u;
        fs << "hidden" << hidden;
        optim_w.write(fs);
        optim_u.write(fs);
        return true;
    }
    virtual bool read(const FileNode &fn)
    {
        BaseWeights::read(fn);
        Mat u;
        fn["U"] >> u;
        if (u.empty())
        {
            U = rand(weights.cols, weights.cols, weight_init);
        }
        else
        {
            u.copyTo(U);
        }
        U_t = U.t();
        int h=0;
        fn["hidden"] >> h;
        if (h>0) hidden=h;
        optim_w.read(fn);
        optim_u.read(fn);
        return true;
    }
    virtual String desc() { return format("%s %d(%d,%d),(%d,%d),(%1.3f,%1.3f)", name.c_str(), hidden, weights.cols, weights.rows, U.cols, U.rows, learn, weight_init); }
};


template <typename Optimizer,typename Loss>
struct Conv : Layer
{
    String name;
    float learn, weight_init;
    int filterSize, numFilters;
    Optimizer optim_w, optim_u;
    Loss loss;
    Volume filters, grads, cache_up, cache_dn;

    Conv(String n="conv")
        : name(n)
        , filterSize(7)
        , numFilters(6)
        , optim_w("opt_w_")
        , optim_u("opt_u_")
    {}

    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        PROFILEX("conv_forward");
        if (DBGLEV) cerr << name  << "_fw \t" << info(upstream);
        downstream.clear();
        for (size_t i=0; i<upstream.size(); i++)
        {
            UMat up = upstream[i];
            for (size_t j=0; j<filters.size(); j++)
            {
                UMat &f = filters[j];
                UMat dn;
                filter2D(up, dn, -1, f);
                downstream.push_back(dn);
            }
        }
        if (training)
        {
            cache_up = upstream;
            cache_dn = downstream;
        }
        if (DBGLEV) cerr << "  " << info(downstream) << endl;
        return 0;
    }

    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        if (DBGLEV) cerr << name << "_bw \t" << info(downstream);
        PROFILEX("conv_backward");

        for (size_t i=0; training && i<filters.size(); i++)
        {
            grads[i] = UMat(filters[0].size(), filters[0].type(), 0.0f);
        }


        upstream.clear();
        for (size_t k=0; k<downstream.size(); k+=numFilters)
        {
            //backward(up, downstream, k);
            //cout << "bw_1 " << info(downstream) << " " << off << endl;
            UMat dn0 = downstream[0];
            UMat usum(dn0.size(), dn0.type(), 0.0f);
            for (size_t i=0; i<filters.size(); i++)
            {
                UMat dn = downstream[k + i];
                UMat ft = filters[i].t();
                UMat u;
                filter2D(dn, u, -1, ft);
                add(u, usum, usum);

                if (! training) continue;

                UMat pred = cache_dn[k + i];
                UMat res  = loss(pred, dn);
                // grad += dx;
                UMat dx, ff = filters[i];//, c = col(cache_up[k/numFilters]);
                //flip(filters[i],ff,0);
                Point anchor(-1,-1);//ff.cols/2 - 1, ff.rows/2 - 1);
                filter2D(ff, dx, -1, res, anchor, 0, BORDER_REFLECT);
                // cerr << "*$$  " <<  dx.size() << endl;
                //add(dx, grads[i], grads[i]);
                gemm(ff.t(), dx, 1, grads[i], 1, grads[i]);
            }
            UMat up;
            divide(usum, numFilters, up);
            upstream.push_back(up);
        }
        if (DBGLEV) cerr << " " << info(upstream) << endl;
        if (! training)
            return 0.0f;

        for (size_t i=0; i<filters.size(); i++)
        {
            scaleAdd(grads[i], -learn/downstream.size(), filters[i], filters[i]);
        }
        return 0.0f;
    }

    virtual void show(String winName)
    {
        //cerr << desc() << endl;
        namedWindow(winName);
        imshow(winName,viz(filters, filterSize));
        namedWindow(winName+"_g");
        imshow(winName+"_g",viz(grads, filterSize));
    }
    virtual bool write(FileStorage &fs)
    {
        fs << "learn" << learn;
        fs << "weight_init" << weight_init;
        fs << "numFilters" << numFilters;
        fs << "filterSize" << filterSize;
        fs << "filters" << "[";
        for (size_t i=0; i<filters.size(); i++)
        {
            Mat f = filters[i].getMat(ACCESS_READ);
            fs << f;
        }
        fs << "]";
        optim_w.write(fs);
        optim_u.write(fs);
        return true;
    }
    virtual bool read(const FileNode &fn)
    {
        int p=0;
        fn["numFilters"] >> p;
        if (p) numFilters = p;
        fn["filterSize"] >> p;
        if (p) filterSize = p;
        float l=0;
        fn["weight_init"] >> l;
        if (l>0) weight_init = l;
        fn["learn"] >> l;
        if (l>0) learn = l;

        FileNode no = fn["filters"];
        if (no.empty())
        {
            for (int i=0; i<numFilters; i++)
            {
                UMat u = rand(filterSize, filterSize, weight_init);
                filters.push_back(u);
                grads.push_back(u);
            }
        }
        else
        {
            for (FileNodeIterator it=no.begin(); it!=no.end(); ++it)
            {
                Mat  m;
                (*it) >> m;

                UMat um;
                m.copyTo(um);
                filters.push_back(um);
                grads.push_back(um);
            }
            filterSize = filters[0].rows;
            numFilters = filters.size();
        }
        optim_w.read(fn);
        optim_u.read(fn);
        return true;
    }
    virtual String type() { return name;}
    virtual String desc() { return format("%s %d %d (%1.3f,%1.3f)", name.c_str(), numFilters, filterSize, learn, weight_init); }
};


#include "layer.cpp"

struct XNN : Network
{
    vector<Ptr<Layer>> layers;
    String name;

    virtual String desc()
    {
        String d=format("%s %d gens\r\n", name.c_str(), ngens);
        for (size_t i=0; i<layers.size(); i++)
        {
            d += layers[i]->desc() + "\r\n";
        }
        return d;
    }
    virtual float forward(const Volume &up, Volume &dn, bool training)
    {
        Volume a(up), b;
        for (size_t i=0; i<layers.size(); i++)
        {
            //cerr << "fw " << i << " " << layers[i]->desc() << " " << a.size() << " " << b.size() << " " << training << endl;
            layers[i]->forward(a,b, training);
            cv::swap(a,b);
        }
        dn = a;
        return 0;
    }
    virtual float backward(Volume &up, const Volume &dn, bool training, float globLearn)
    {
        ngens ++;
        float e=0;
        Volume a, b(dn);
        for (int i=int(layers.size())-1; i>=0; i--)
        {
            //cerr << "bw " << i << " " << layers[i]->desc() << " " << a.size() << " " << b.size() << " " << training << endl;
            e = layers[i]->backward(a,b,training,globLearn);
            cv::swap(a,b);
        }
        up = b;
        return e; // 1st layer
    }
    virtual bool save(String  fn)
    {
        FileStorage fs(fn,1);
        if (!fs.isOpened())
        {
            CV_Error(0, String("could not save to ") + fn);
        }
        fs << "layers" << "[";
        for (size_t i=0; i<layers.size(); i++)
        {
            fs << "{:";
            fs << "type" << layers[i]->type();
            layers[i]->write(fs);
            fs << "}";
        }
        fs << "]";
        fs << "ngens" << ngens;
        fs.release();
        return true;
    }
    virtual bool load(String  fn)
    {
        FileStorage fs(fn,0);
        if (!fs.isOpened())
        {
            CV_Error(0, String("could not load ") + fn);
        }
        name = fn;
        FileNode no = fs["layers"];
        int i=0;
        for (FileNodeIterator it=no.begin(); it!=no.end(); ++it)
        {
            const FileNode &n = *it;
            String type;
            n["type"] >> type;
            Ptr<Layer> layer;
            if (type=="sigmoid")  layer = makePtr<Activation>(sigmoid,sigmoid_bp,"sigmoid");
            if (type=="relu")     layer = makePtr<Activation>(relu,relu_bp,"relu");
            if (type=="tanh")     layer = makePtr<Activation>(tanh_fw,tanh_bw,"tanh");
            if (type=="tanh2")    layer = makePtr<Activation>(tanh2_fw,tanh2_bw,"tanh2");
            if (type=="mean")     layer = makePtr<Activation>(mean,mean,"mean");
            if (type=="minmax")   layer = makePtr<Activation>(minmax,minmax,"minmax");
            if (type=="pool22")   layer = makePtr<Activation>(poolavg22_fw,poolavg22_bw,"pool22");
            if (type=="pool23")   layer = makePtr<Activation>(poolavg23_fw,poolavg23_bw,"pool23");
            if (type=="pool33")   layer = makePtr<Activation>(poolavg33_fw,poolavg33_bw,"pool33");
            if (type=="dropout")  layer = makePtr<Dropout>();
            if (type=="batchnorm")layer = makePtr<BatchNorm>();
            if (type=="fully")    layer = makePtr<Fully<SGD, LossLinear>>("fully");
            if (type=="flatten")  layer = makePtr<Flatten>("flatten");
            if (type=="fully_mom")layer = makePtr<Fully<momentum, LossLinear>>("fully_mom");
            if (type=="fully_ada")layer = makePtr<Fully<adagrad, LossLinear>>("fully_ada");
            if (type=="fully_rms")layer = makePtr<Fully<RMSprop, LossLinear>>("fully_rms");
            if (type=="softmax")  layer = makePtr<Fully<SGD, LossSoftmax>>("softmax");
            if (type=="softmax_mom") layer = makePtr<Fully<momentum, LossSoftmax>>("softmax_mom");
            if (type=="softmax_rms") layer = makePtr<Fully<RMSprop, LossSoftmax>>("softmax_rms");
            if (type=="conv")      layer = makePtr<Conv<SGD,LossLinear>>();
            //if (type=="rbm")      layer = makePtr<RBM<SGD>>();
            if (type=="recurrent")layer = makePtr<Rnn<SGD,LossLinear>>();
            if (type=="visu")     layer = makePtr<Visu>();
            if (layer.empty())
            {
                CV_Error(0, format("unknown layer(%d): ", i) + type);
            }
            layer->read(n);
            layers.push_back(layer);
            i++;
        }
        fs["ngens"] >> ngens;
        fs.release();
        return true;
    }
    virtual void show()
    {
        for (size_t i=0; i<layers.size(); i++)
        {
            String n = format("%s_%d", layers[i]->type().c_str(), i);
            layers[i]->show(n);
        }
    }
    int ngens;
};

} // namespace impl


Ptr<Network> createNetwork(String name)
{
    Ptr<Network> nn = makePtr<impl::XNN>();
    nn->load(name);
    return nn;
}


// public helper:
Mat viz(const Volume &v, int patchSize)
{
    PROFILEX("viz_vol")
    int n = (int)sqrt(double(v.size()*2));
    Mat draw(n*patchSize, n*patchSize, CV_32F, 0.0f);
    for (size_t i=0; i<v.size(); i++)
    {
        Mat m2, m = v[i].getMat(ACCESS_READ).reshape(1,patchSize);
        normalize(m, m2, 1, 0, NORM_MINMAX);
        int r = patchSize * (i / n);
        int c = patchSize * (i % n);
        m.copyTo(draw(Rect(c,r,patchSize,patchSize)));
    }
    if (draw.total() < 400*400)
        resize(draw,draw, Size(400,400),-1,INTER_CUBIC);
    return draw;
}
Mat viz(const UMat &weights, float fac)
{
    PROFILEX("viz_weights")
    int pn = (int)sqrt(double(weights.cols)) + 1;
    int ps = (int)sqrt(double(weights.rows));
    Mat draw(pn*ps+2,pn*ps+2,CV_32F,0.0f);
    for (int i=0; i<weights.cols; i++)
    {
        Mat f = weights.getMat(ACCESS_READ).col(i).clone().reshape(1,ps);
        normalize(f, f, fac, 0, NORM_MINMAX);
        int r = ps * int(i / pn);
        int c = ps * int(i % pn);
        f.copyTo(draw(Rect(c, r, ps, ps)));
    }
    if (draw.total() < 400*400)
        resize(draw,draw, Size(400,400),-1,INTER_CUBIC);
    return draw;
}

} // namespace nn
