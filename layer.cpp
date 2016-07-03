
struct Activation : Layer
{
    proc fw,bw;
    String _n;
    Activation(proc fw, proc bw, String n="chicken") : fw(fw), bw(bw), _n(n) {}

    float pipe(const Volume &from, Volume &to, proc act)
    {
        if (DBGLEV) cerr << _n << (act==fw?"_fw":"_bw") << " \t" << info(from);
        to.resize(from.size());
        for (size_t i=0; i<from.size(); i++)  to[i] = act(from[i]);
        if (DBGLEV) cerr << " " << info(to) << endl;
        return 0;
    }

    virtual float forward(const Volume &upstream, Volume &downstream, bool)
    {
        return pipe(upstream, downstream, fw);
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        return pipe(downstream, upstream, bw);
    }
    virtual String type() {return _n;}
    virtual String desc() {return _n;}
};

struct Flatten : Layer
{
    String name;
    int numInputs;
    Size oldShape;
    Flatten(String n="") : name(n), numInputs(1) {}
    virtual bool write(FileStorage &fs)
    {
        fs << "numInputs" << numInputs;
        return true;
    }

    virtual bool read(const FileNode &fn)
    {
        fn["numInputs"] >> numInputs;
        return numInputs != 0;
    }
    virtual String type() {return name;}
    virtual String desc() {return format("%s(%d)",name.c_str(),numInputs);}
    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        PROFILEX("flat_forward");
        if (DBGLEV) cerr << name << "_fw \t" << info(upstream);
        downstream.clear();
        for (size_t i=0; i<upstream.size(); i+=numInputs)
        {
            int x=0,y=0,w=upstream[i].cols, h=upstream[i].rows;
            UMat flat(h*numInputs, w, upstream[i].type());
            for (int p=0; p<numInputs; p++)
            {
                UMat up = upstream[i+p];
                up.copyTo(flat(Rect(x,y,w,h)));
                y += h;
            }
            oldShape = Size(w,h);
            downstream.push_back(flat.reshape(1,1));
        }
        if (DBGLEV) cerr << " " << info(downstream) << endl;
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        PROFILEX("flat_backward");
        if (DBGLEV) cerr << name << "_bw \t" << info(downstream) << " " << oldShape;

        upstream.clear();
        for (size_t i=0; i<downstream.size(); i++)
        {
            UMat dn = downstream[i].reshape(1,oldShape.height*numInputs);
            Rect r(Point(),oldShape);
            for (int j=0; j<numInputs; j++)
            {
                UMat u = dn(r);
                r.y += oldShape.height;
                upstream.push_back(u);
            }
        }
        if (DBGLEV) cerr << " " << info(upstream) << endl;
        return 0;
    }
};

struct Visu : Layer
{
    String _n;
    Visu(String n="visu") : _n(n) {}

    Volume cup,cdn;

    virtual void show(String winName)
    {
        //cerr << desc() << endl;
        if (!cup.empty())
        {
            int filterSize = (int)sqrt(double(cup[0].total()));
            if (size_t(filterSize*filterSize) == cup[0].total())
                imshow(winName + "_up", viz(cup, filterSize));
        }
        if (!cdn.empty())
        {
            int filterSize = (int)sqrt(double(cdn[0].total()));
            if (size_t(filterSize*filterSize) == cdn[0].total())
                imshow(winName + "_dn", viz(cdn, filterSize));
        }
    }
    virtual float forward(const Volume &upstream, Volume &downstream, bool)
    {
        cup = upstream;
        downstream = upstream;
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        cdn = downstream;
        upstream = downstream;
        return 0;
    }
    virtual String type() {return _n;}
    virtual String desc() {return _n;}
};



struct Dropout : Layer
{
    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        if (! training)
        {
            downstream = upstream;
            return 0;
        }
        downstream.resize(upstream.size());
        for (size_t i=0; i<upstream.size(); i++)
            downstream[i] = dropout(upstream[i]);
        return 0;
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        upstream = downstream;
        return 0;
    }
    virtual String type() { return "dropout"; }
    virtual String desc() { return "dropout"; }
};


struct BatchNorm : Layer
{
    virtual float batch(const Volume &from, Volume &to)
    {
        PROFILEX("batchnorm")
        to.resize(from.size());
        /*// global
        Scalar m,s, M, S;
        for (size_t i=0; i<from.size(); i++)
        {
            meanStdDev(from[0], m, s);
            M += m;
            S += s;
        }
        M[0] /= from.size();
        S[0] /= from.size();
        for (size_t i=0; i<from.size(); i++)
        {
            subtract(from[0], M, to[i]);
            divide(to[i], S[0]+0.0000001, to[i]);
        }*/
        // per pixel
        UMat m(from[0].size(), from[0].type(), 0.0f);
        for (size_t i=0; i<from.size(); i++)
            add(from[i], m, m);
        divide(m, from.size(), m);

        UMat v(from[0].size(), from[0].type(), 0.00000001f);
        for (size_t i=0; i<from.size(); i++)
        {
            UMat s;
            subtract(from[i], m, s);
            multiply(s,s,s);
            add(v,s,v);
        }
        divide(v, from.size(), v);
        for (size_t i=0; i<from.size(); i++)
        {
            subtract(from[i], m, to[i]);
            divide(to[i], v, to[i]);
        }
        return 0;
    }
    virtual float forward(const Volume &upstream, Volume &downstream, bool training)
    {
        if (! training)
        {
            downstream = upstream;
            return 0;
        }
        return batch(upstream, downstream);
    }
    virtual float backward(Volume &upstream, const Volume &downstream, bool training, float globLearn)
    {
        if (! training)
        {
            upstream = downstream;
            return 0;
        }
        return batch(downstream, upstream);
    }
    virtual String type() { return "batchnorm"; }
    virtual String desc() { return "batchnorm"; }
};
