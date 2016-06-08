UMat identity(const UMat &m)
{
    return m;
}

UMat minmax(const UMat &m)
{
    PROFILE;
    normalize(m,m,1,0,NORM_MINMAX);
    return m; 
}
UMat mean(const UMat &m)
{
    PROFILE;
    Scalar _m, _s;
    meanStdDev(m, _m, _s);
    UMat res;
    subtract(m, _m, res);
    return res; 
}

UMat relu(const UMat &m)
{
    PROFILE;
    UMat u;
    max(m, 0, u);
    return u;
}
UMat relu_bp(const UMat &m)
{
    PROFILE;
    UMat mask;
    compare(m, 0, mask, CMP_GT);
    UMat u = m;
    u.setTo(1, mask);
    return u;
}

UMat sigmoid(const UMat &m)
{
    PROFILE;
    UMat P;
    multiply(m, -1, P);
    exp(P,P);
    add(P, 1, P);
    divide(1,P, P);
    return P;
}
UMat sigmoid_bp(const UMat &m)
{
    PROFILE;
    UMat res;
    subtract(1.0, m, res);
    res = res.mul(m);
    return res;
}

UMat tanh_fw(const UMat &m)
{
    PROFILE;
    UMat _m,ep,en,sp,sn,res;
    multiply(m, -1, _m);
    exp(m, ep);
    exp(_m, en);
    add(ep, en, sp);
    subtract(ep, en, sn);
    divide(sn, sp, res);
    return res;
}
UMat tanh_bw(const UMat &m)
{
    PROFILE;
    UMat _m, res;
    subtract(1, m, _m);
    sqrt(_m, res);
    return res;
}

// s tan_h, but scaled to match the other functions
UMat tanh2_fw(const UMat &m)
{
    PROFILE;
    UMat _m,ep,en,sp,sn,res;
    multiply(m, -1, _m);
    exp(m, ep);
    exp(_m, en);
    add(ep, en, sp);
    divide(ep, sp, res);
    return res;
}
UMat tanh2_bw(const UMat &m)
{
    PROFILE;
    UMat _m, s1, res;
    subtract(1, m, _m);
    multiply(_m, m, s1);
    multiply(s1, 2, res);
    return res;
}

UMat rand(int r, int c, double v=0.5f)
{
    PROFILE;
    UMat m(r, c, CV_32F);
    if (v < 0.00001f)
        randn(m,0,sqrt(2.0/(r*c)));
    else
        randn(m, 0.0, v);
    return m;
}

UMat dropout(const UMat &m)
{
    PROFILE;
    float prob = 0.5f;
    UMat m1 = rand(m.rows, m.cols);
    UMat mask;
    compare(m1, prob, mask, CMP_GT);
    UMat res;
    multiply(m, mask, res, 1.0/(prob*255), CV_32F);
    return res;
}

UMat softmax(const UMat &m)
{
    UMat prob;
    double maxVal=0;
    minMaxLoc(m, 0, &maxVal);
    subtract(m, maxVal, prob);
    exp(prob, prob);
    Scalar total = sum(prob);
    divide(prob, total[0], prob);
    return prob;  
}
