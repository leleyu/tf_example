//
// Created by leleyu on 2018-12-28.
//

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
  auto root = Scope::NewRootScope();
  auto x = Variable(root, {5, 2}, DataType::DT_FLOAT);
  auto assign_x = Assign(root, x, RandomNormal(root, {5, 2}, DataType::DT_FLOAT));
  auto y = Variable(root, {2, 3}, DataType::DT_FLOAT);
  auto assign_y = Assign(root, y, RandomNormal(root, {2, 3}, DataType::DT_FLOAT));

  auto xy = MatMul(root, x, y);
  auto z = Const(root, 2.f, {5, 3});
  auto xyz = Add(root, xy, z);

  auto session = new ClientSession(root);
  TF_CHECK_OK(session->Run({assign_x, assign_y}, nullptr));

  std::vector<Tensor> outputs;
  TF_CHECK_OK(session->Run({xyz}, &outputs));

  LOG(INFO) << "xyz = " << outputs[0].matrix<float>() ;
  return 0;
}