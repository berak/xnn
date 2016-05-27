// from opencv_contrib/dnn
#ifdef HAVE_OPENCL
void im2col_ocl(UMat &img,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                UMat &col)
{
    int h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    CV_Assert(img.isContinuous() && col.isContinuous());
    CV_Assert(img.total() == (size_t)channels * height * width);
    CV_Assert(col.total() == (size_t)channels * kernel_h * kernel_w * h_out * w_out);

    ocl::Kernel im2col_ker("im2col", ocl::dnn::im2col_oclsrc);
    CV_Assert(!im2col_ker.empty());

    im2col_ker.args(ocl::KernelArg::PtrReadOnly(img), (int)img.offset,
             channels, height, width,
             kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
             h_out, w_out,
             ocl::KernelArg::PtrWriteOnly(col), (int)col.offset
        );

    size_t localSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t globalSize = (size_t)channels * h_out * w_out;

    CV_Assert(im2col_ker.run(1, &globalSize, &localSize, true));
}
#else
void im2col(const float* data_im,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                Dtype* data_col)
{
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] =
                    data_im[(c_im * height + h_pad) * width + w_pad];
                else
                    data_col[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }
}

#endif // HAVE_OPENCL
