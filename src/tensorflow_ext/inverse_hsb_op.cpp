
// Inverse hierarchical stick breaking transformation

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"

#include <cstdio>

using namespace tensorflow;

REGISTER_OP("InvHSB")
    .Input("x: float32")
    .Input("left_index: int32")
    .Input("right_index: int32")
    .Input("leaf_index: int32")
    .Output("y_logit: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle x;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));

        shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));

        shape_inference::DimensionHandle m = c->Dim(x, 0);
        shape_inference::DimensionHandle n = c->Dim(x, 1);
        shape_inference::DimensionHandle n_minus_one = c->MakeDim(c->Value(n) - 1);
        c->set_output(0, c->MakeShape({m, n_minus_one}));
        return Status::OK();
    });


class InvHSBOp : public OpKernel {
    public:
    explicit InvHSBOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_x           = context->input(0);
        const Tensor& input_left_index  = context->input(1);
        const Tensor& input_right_index = context->input(2);
        const Tensor& input_leaf_index  = context->input(3);

        // TODO: check sizes

        const int64 m = input_x.dim_size(0);
        const int64 n = input_x.dim_size(1);

        Tensor* output_y_logit = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(
                0, TensorShape({m, n-1}), &output_y_logit));

        auto x           = input_x.flat_inner_dims<float>();
        auto y_logit     = output_y_logit->flat_inner_dims<float>();
        auto left_index  = input_left_index.flat_inner_dims<int32>();
        auto right_index = input_right_index.flat_inner_dims<int32>();
        auto leaf_index  = input_leaf_index.flat_inner_dims<int32>();

        auto InvHSBBatch = [&, context](int start_batch, int limit_batch) {
            // fprintf(stderr, "start_batch = %d, limit_batch = %d\n",
            //         (int) start_batch, (int) limit_batch);

            // intermediate values
            auto shp = TensorShape({2*n - 1});
            auto u_tens = Tensor(DT_DOUBLE, shp);
            auto u = u_tens.flat<double>();
            double* u_data = &u(0);

            for (int i = start_batch; i < limit_batch; ++i) {
                const float* x_i = &x(i, 0);
                float* y_logit_i = &y_logit(i, 0);
                const int32* left_index_i  = &left_index(i, 0);
                const int32* right_index_i = &right_index(i, 0);
                const int32* leaf_index_i  = &leaf_index(i, 0);

                int k = n - 2;
                for (int j = 2*n-2; j >= 0; --j) {
                    // leaf node
                    if (leaf_index_i[j] >= 0) {
                        u_data[j] = x_i[leaf_index_i[j]];
                    }
                    // internal node
                    else {
                        double u_left = u_data[left_index_i[j]];
                        double u_right = u_data[right_index_i[j]];
                        u_data[j] = u_left + u_right;
                        double y_ik = u_left / u_data[j];
                        y_logit_i[k] = (float) (log(y_ik) - log(1.0 - y_ik));
                        k -= 1;
                    }
                }
            }
        };

        const int64 cost_est = n * 10; // total wild guess
        auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers,
              m, cost_est, InvHSBBatch);
    }
};


REGISTER_KERNEL_BUILDER(Name("InvHSB").Device(DEVICE_CPU), InvHSBOp);


REGISTER_OP("InvHSBGradOp")
    .Input("gradients: float32")
    .Input("x: float32")
    .Input("left_index: int32")
    .Input("right_index: int32")
    .Input("leaf_index: int32")
    .Output("backprops: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle x;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));

        shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &unused));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));

        c->set_output(0, x);
        return Status::OK();
    });


class InvHSBGradOp : public OpKernel {
    public:
    explicit InvHSBGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // TODO:
    }
};


REGISTER_KERNEL_BUILDER(Name("InvHSBGrad").Device(DEVICE_CPU), InvHSBGradOp);

