#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/sparse_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

void test_sparse_dense_matmul() {
  auto root = Scope::NewRootScope();
  auto session = std::unique_ptr<ClientSession>(new ClientSession(root));

  auto a_indice = Const(root, {{static_cast<int64>(0), static_cast<int64>(0)},
                               {static_cast<int64>(0), static_cast<int64>(1)},
                               {static_cast<int64>(1), static_cast<int64>(0)},
                               {static_cast<int64>(1), static_cast<int64>(2)}});
  auto a_value  = Const(root, {0.1f, 0.2f, 0.2f, 0.2f});
  auto a_shape  = Const(root, {static_cast<int64>(2), static_cast<int64_t>(3)});

  auto b = Const(root, {{0.1f, 0.1f},{0.1f, 0.1f},{0.1f, 0.1f}});
  auto c = SparseTensorDenseMatMul(root, a_indice, a_value, a_shape, b);

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({c}, &outputs));

  std::cout << outputs[0].matrix<float>() << std::endl;
}

void test_sparse_dense_add_const() {
  auto root = Scope::NewRootScope();
  auto session = std::unique_ptr<ClientSession>(new ClientSession(root));

  int64_t dim = 5;

  auto a_indice = Const(root, {{static_cast<int64_t >(0)}, {static_cast<int64_t>(1)}});
  auto a_value  = Const(root, {0.1f, 0.2f});
  auto a_shape  = Const(root, {dim});

  auto b = Const(root, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f});

  auto c = SparseTensorDenseAdd(root, a_indice, a_value, a_shape, b);

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({c}, &outputs));
  std::cout << outputs[0].flat<float>() << std::endl;
}

void test_sparse_add_const() {
  auto root = Scope::NewRootScope();
  auto session = std::unique_ptr<ClientSession>(new ClientSession(root));

  int64_t dim = 5;

  auto a_indice = Const(root, {{static_cast<int64_t >(0)}, {static_cast<int64_t>(1)}});
  auto a_value  = Const(root, {0.1f, 0.2f});
  auto a_shape  = Const(root, {dim});

  auto b_indice = Const(root, {{static_cast<int64_t >(1)}, {static_cast<int64_t>(3)}});
  auto b_value  = Const(root, {0.1f, 0.2f});
  auto b_shape  = Const(root, {dim});

  auto threshold = Const(root, 0);

  auto c = SparseAdd(root, a_indice, b_value, a_shape,
    b_indice, b_value, b_shape, threshold);

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({c.sum_indices, c.sum_values, c.sum_shape}, &outputs));
  std::cout << outputs[0].flat<int64_t>() << std::endl;
  std::cout << outputs[1].flat<float>() << std::endl;
  std::cout << outputs[2].flat<int64_t>() << std::endl;
}

void test_sparse_dense_matmul_placeholder() {
  auto root = Scope::NewRootScope();
  auto session = std::unique_ptr<ClientSession>(new ClientSession(root));

  Tensor a_indice(DT_INT64, {4, 2});
  Tensor a_value (DT_FLOAT, {4});

  // set indices and values
  int64* indice_ptr = a_indice.flat<int64>().data();
  float* value_ptr  = a_value.vec<float>().data();
  indice_ptr[0] = 0; indice_ptr[1] = 0; indice_ptr[2] = 0;indice_ptr[3] = 1;
  indice_ptr[4] = 1; indice_ptr[5] = 0; indice_ptr[6] = 1;indice_ptr[7] = 2;
  value_ptr[0] = 0.1f; value_ptr[1] = 0.1f;
  value_ptr[2] = 0.4f; value_ptr[3] = 0.2f;

  std::cout << a_indice.DebugString() << std::endl;



  auto a_shape = Const(root, {static_cast<int64 >(2), static_cast<int64 >(3)});
  auto b = Const(root, {{0.1f, 0.1f},{0.1f, 0.1f},{0.1f, 0.1f}});

  auto a_indice_place = Placeholder(root, DT_INT64);
  auto a_value_place  = Placeholder(root, DT_FLOAT);

  auto c = SparseTensorDenseMatMul(root, a_indice_place, a_value_place, a_shape, b);

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({{a_indice_place, a_indice}, {a_value_place, a_value}}, {c}, &outputs));

  std::cout << outputs[0].matrix<float>() << std::endl;

}

void set_tensor_value() {
  Tensor t(DT_FLOAT, TensorShape({2, 3}));
  float* data = t.flat<float>().data();
  data[0] = 0.1f;
  data[1] = 0.0f;
  data[2] = 0.0f;
  data[3] = 0.0f;
  data[4] = 0.0f;
  data[5] = 0.0f;

  auto root = Scope::NewRootScope();

  auto x = Placeholder(root, DataType::DT_FLOAT);
  auto zero = Const(root, 0.0f, {2, 3});
  auto y = Add(root, x, zero);

  auto session = new ClientSession(root);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({{x, t}}, {y}, &outputs));

  LOG(INFO) << outputs[0].flat<float>();
  delete session;
}

int main() {
//  set_tensor_value();
//  test_sparse_tensor();
//  test_sparse_dense_add_const();
//  test_sparse_add_const();
//  test_sparse_dense_matmul();
  test_sparse_dense_matmul_placeholder();
  return 0;
}