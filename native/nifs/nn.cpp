#include "erl_nif.h"

#include "cblas.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
double *GetDoubleMatrix(ErlNifEnv *env, ERL_NIF_TERM list_term, std::vector<size_t> &shape) {
  unsigned int length;

  if (!enif_get_list_length(env, list_term, &length)) {
    return nullptr;
  }
  shape.push_back(length);

  ERL_NIF_TERM head, tail;

  double *result = nullptr;
  size_t idx = 0;
  for (unsigned int i = 0; i < length; ++i) {
    if (!enif_get_list_cell(env, list_term, &head, &tail)) {
      return result;
    }
    unsigned int inner_length;
    if (!enif_get_list_length(env, head, &inner_length)) {
      return result;
    }
    if (i == 0) {
      shape.push_back(inner_length);
      result = (double *)enif_alloc(sizeof(double) * length * inner_length);
    }
    ERL_NIF_TERM inner_head, inner_tail;
    for (unsigned int j = 0; j < inner_length; ++j) {
      if (!enif_get_list_cell(env, head, &inner_head, &inner_tail)) {
        enif_free(result);
        return nullptr;
      }
      double actualHead;
      if (!enif_get_double(env, inner_head, &actualHead)) {
        enif_free(result);
        return nullptr;
      }
      result[idx++] = actualHead;
      head = inner_tail;
    }
    list_term = tail;
  }
  return result;
}

int GetDoubleArray(ErlNifEnv *env, ERL_NIF_TERM list_term, const unsigned int length, int idx,
                   double *out) {
  ERL_NIF_TERM head, tail;
  for (unsigned int i = 0; i < length; ++i) {
    if (!enif_get_list_cell(env, list_term, &head, &tail)) {
      return -1;
    }
    double actualHead;
    if (!enif_get_double(env, head, &actualHead)) {
      return -1;
    }
    out[idx++] = actualHead;
    list_term = tail;
  }
  return idx;
}

int GetDoubleArrayNd(ErlNifEnv *env, ERL_NIF_TERM list_term, const int dim, int idx, double *out) {
  unsigned int length;
  if (!enif_get_list_length(env, list_term, &length)) {
    return -1;
  }
  if (dim == 1) {
    return GetDoubleArray(env, list_term, length, idx, out);
  }

  ERL_NIF_TERM head, tail;
  for (unsigned int b = 0; b < length; ++b) {
    if (!enif_get_list_cell(env, list_term, &head, &tail)) {
      return -1;
    }
    idx = GetDoubleArrayNd(env, head, dim - 1, idx, out);
    if (idx == -1) return -1;
    list_term = tail;
  }
  return idx;
}

void GetLengthVector(ErlNifEnv *env, ERL_NIF_TERM list_term, std::vector<size_t> &shape, int &dims,
                     int &size) {
  unsigned int length;
  size = 1;
  dims = 0;
  while (enif_get_list_length(env, list_term, &length)) {
    ++dims;
    size *= length;
    shape.push_back(length);
    ERL_NIF_TERM head, tail;
    if (!enif_get_list_cell(env, list_term, &head, &tail)) {
      break;
    }
    list_term = head;
  }
}

ERL_NIF_TERM MakeDoubleMatrix(ErlNifEnv *env, const double *src, const std::vector<int> shape) {
  ERL_NIF_TERM result_matrix = enif_make_list(env, 0);
  for (int i = shape[0]; i;) {
    --i;
    ERL_NIF_TERM result_list = enif_make_list(env, 0);
    for (int j = shape[1]; j;) {
      --j;
      result_list =
          enif_make_list_cell(env, enif_make_double(env, src[i * shape[1] + j]), result_list);
    }
    result_matrix = enif_make_list_cell(env, result_list, result_matrix);
  }
  return result_matrix;
}

ERL_NIF_TERM MakeDouble1dList(ErlNifEnv *env, const double *src, const std::vector<size_t> shape,
                              size_t dim, int idx) {
  ERL_NIF_TERM list_term = enif_make_list(env, 0);
  for (size_t i = 1; i <= shape[dim]; ++i) {
    list_term = enif_make_list_cell(env, enif_make_double(env, src[idx - i]), list_term);
  }
  return list_term;
}

ERL_NIF_TERM MakeDoubleNdList(ErlNifEnv *env, const double *src, const std::vector<size_t> shape,
                              size_t dim, int idx) {
  if (dim + 1 == shape.size()) {
    return MakeDouble1dList(env, src, shape, dim, idx);
  }
  ERL_NIF_TERM result_list = enif_make_list(env, 0);
  size_t base_size = 1;
  for (size_t i = dim + 1; i < shape.size(); ++i) {
    base_size *= shape[i];
  }
  for (size_t i = 0; i < shape[dim]; ++i) {
    ERL_NIF_TERM inner_list = MakeDoubleNdList(env, src, shape, dim + 1, idx);
    result_list = enif_make_list_cell(env, inner_list, result_list);
    idx -= base_size;
  }
  return result_list;
}

}  // namespace

// static ERL_NIF_TERM conv2d_backward(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
//   unsigned int length_dout, length_weight, length_shape;

//   if (!enif_get_list_length(env, argv[0], &length_dout) ||
//       !enif_get_list_length(env, argv[1], &length_weight) ||
//       !enif_get_list_length(env, argv[2], &length_shape)) {
//     return false;
//   }

//   std::vector<double> dout = GetDoubleVector(env, argv[0], length_dout);
//   std::vector<double> weight = GetDoubleVector(env, argv[1], length_weight);
//   const std::vector<int> dout_shape = GetIntVector(env, argv[2], length_shape);
//   const std::vector<int> original_shape = GetIntVector(env, argv[3], length_shape);
//   int filter_size = enif_get_int(argv[4]);
//   int stride = enif_get_int(argv[5]);
//   int padding = enif_get_int(argv[6]);

//   unsigned int length =
//       original_shape[0] * original_shape[1] * original_shape[2] * original_shape[3];

//   const int batch_size = original_shape[0];
//   const int channel_size = original_shape[1];
//   const int height = original_shape[2];
//   const int width = original_shape[3];

//   const int filter_height = dout_shape[2];
//   const int filter_width = dout_shape[3];
//   const int step = static_cast<int>(std::ceil(filter_size * 1.0 / stride));
//   std::vector<double> result(0.0, length);
//   for (int b = 0; b < batch_size; ++b) {
//     const size_t batch_base = b * (channel_size * height * width);
//     const size_t dout_batch_base = b * (dout_shape[1] * filter_height * filter_width);
//     for (int c = 0; c < channel_size; ++c) {
//       const size_t channel_base = c * (height * width);
//       for (int h = 0; h < height; ++h) {
//         const size_t height_base = h * width;
//         for (int w = 0; w < width; ++w) {
//           const size_t index = batch_base + channel_base + height_base + w;
//           for (int fh = 0; fh < filter_size; ++fh) {
//             if ((padding + h - fh) < 0 || (padding + h + fh) % stride) continue;
//             const int out_y = padding + (h * width);
//             for (int fw = 0; fw < filter_size; ++fw) {
//               if ((padding + w - fw) < 0 || (w + filter_size - padding)(w + fw) % stride)
//               continue; for (int fc = 0; fc < dout_shape[1]; ++fc) {
//                 const size_t weight_index =
//                     (fc * dout_shape[1] * filter_size * filter_size) + (filter_size * fh) + fw;
//                 result[index] += dout[dout_batch_base +
//               }
//             }
//           }
//         }
//       }
//     }
//   }

//   return MakeDoubleList(env, result, length);
// }
// g++ -fPIC -I/Users/tamura/.asdf/installs/erlang/21.3.8.6/erts-10.3.5.4/include -dynamiclib
// -undefined dynamic_lookup -o priv/nn.so native/nif/nn.cpp
static ERL_NIF_TERM nif_transpose2d(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  std::vector<size_t> shape;
  double *src = GetDoubleMatrix(env, argv[0], shape);
  const int outer_size = shape[1];
  const int inner_size = shape[0];
  double *dst = (double *)enif_alloc(sizeof(double) * outer_size * inner_size);
  size_t dst_idx = 0;
  for (int o = 0; o < outer_size; ++o) {
    for (int i = 0; i < inner_size; ++i) {
      dst[dst_idx++] = src[o + i * outer_size];
    }
  }
  std::vector<int> new_shape{outer_size, inner_size};
  ERL_NIF_TERM result = MakeDoubleMatrix(env, dst, new_shape);

  enif_free(src);
  enif_free(dst);

  return result;
}

static ERL_NIF_TERM nif_transpose4d(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  int size = 1;
  int dim = 0;
  std::vector<size_t> shape;
  GetLengthVector(env, argv[0], shape, dim, size);
  shape.reserve(4);
  double *src = (double *)enif_alloc(sizeof(double) * size);
  if (GetDoubleArrayNd(env, argv[0], 4, 0, src) == -1) {
    return argv[0];
  }
  int axis0, axis1, axis2, axis3;
  if (!enif_get_int(env, argv[1], &axis0) || !enif_get_int(env, argv[2], &axis1) ||
      !enif_get_int(env, argv[3], &axis2) || !enif_get_int(env, argv[4], &axis3)) {
    return argv[0];
  }
  std::vector<size_t> sizes{shape[axis0], shape[axis1], shape[axis2], shape[axis3]};
  std::vector<size_t> base_sizes{shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
  double *dst = (double *)enif_alloc(sizeof(double) * size);

  size_t dst_idx = 0;
  for (size_t a0 = 0; a0 < sizes[0]; ++a0) {
    const size_t a0_base = a0 * base_sizes[axis0];
    for (size_t a1 = 0; a1 < sizes[1]; ++a1) {
      const size_t a1_base = a1 * base_sizes[axis1];
      for (size_t a2 = 0; a2 < sizes[2]; ++a2) {
        const size_t a2_base = a2 * base_sizes[axis2];
        for (size_t a3 = 0; a3 < sizes[3]; ++a3) {
          dst[dst_idx++] = src[a0_base + a1_base + a2_base + a3 * base_sizes[axis3]];
        }
      }
    }
  }
  ERL_NIF_TERM result_term = MakeDoubleNdList(env, dst, sizes, 0, size);

  enif_free(src);
  enif_free(dst);

  return result_term;
}

static ERL_NIF_TERM nif_dot(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  std::vector<size_t> shape1;
  std::vector<size_t> shape2;
  double *src1 = GetDoubleMatrix(env, argv[0], shape1);
  double *src2 = GetDoubleMatrix(env, argv[1], shape2);
  int m = shape1[0];
  int k = shape1[1];
  int n = shape2[1];

  double *dst = (double *)enif_alloc(sizeof(double) * shape1[0] * shape2[1]);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, src1, k, src2, n, 0.0, dst,
              n);
  std::vector<int> new_shape{m, n};
  ERL_NIF_TERM result = MakeDoubleMatrix(env, dst, new_shape);

  enif_free(src1);
  enif_free(src2);
  enif_free(dst);

  return result;
}

static ERL_NIF_TERM nif_dot_nt(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  std::vector<size_t> shape1;
  std::vector<size_t> shape2;
  double *src1 = GetDoubleMatrix(env, argv[0], shape1);
  double *src2 = GetDoubleMatrix(env, argv[1], shape2);
  int m = shape1[0];
  int k = shape1[1];
  int n = shape2[0];

  double *dst = (double *)enif_alloc(sizeof(double) * m * n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, src1, k, src2, k, 0.0, dst, n);
  std::vector<int> new_shape{m, n};
  ERL_NIF_TERM result = MakeDoubleMatrix(env, dst, new_shape);

  enif_free(src1);
  enif_free(src2);
  enif_free(dst);

  return result;
}

static ERL_NIF_TERM nif_dot_tn(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  std::vector<size_t> shape1;
  std::vector<size_t> shape2;
  double *src1 = GetDoubleMatrix(env, argv[0], shape1);
  double *src2 = GetDoubleMatrix(env, argv[1], shape2);
  int m = shape1[1];
  int k = shape2[0];
  int n = shape2[1];

  double *dst = (double *)enif_alloc(sizeof(double) * m * n);
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0, src1, m, src2, n, 0.0, dst, n);
  std::vector<int> new_shape{m, n};
  ERL_NIF_TERM result = MakeDoubleMatrix(env, dst, new_shape);

  enif_free(src1);
  enif_free(src2);
  enif_free(dst);

  return result;
}

static ERL_NIF_TERM nif_add(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  int size_a = 1;
  int dim_a = 0;
  int size_b = 1;
  int dim_b = 0;
  std::vector<size_t> shape1;
  std::vector<size_t> shape2;
  GetLengthVector(env, argv[0], shape1, dim_a, size_a);
  GetLengthVector(env, argv[1], shape2, dim_b, size_b);

  double *src1 = (double *)enif_alloc(sizeof(double) * size_a);
  double *src2 = (double *)enif_alloc(sizeof(double) * size_b);
  if (GetDoubleArrayNd(env, argv[0], dim_a, 0, src1) == -1) {
    return argv[0];
  }
  if (GetDoubleArrayNd(env, argv[1], dim_b, 0, src2) == -1) {
    return argv[1];
  }
  if (size_a != size_b) {
    size_t copy_cnt = size_a / size_b;
    double *broadcast = (double *)enif_alloc(sizeof(double) * size_a);
    for (size_t i = 0; i < copy_cnt; ++i) {
      memcpy(&broadcast[i * size_b], src2, sizeof(double) * size_b);
    }
    enif_free(src2);
    src2 = broadcast;
  }
  cblas_daxpy(size_a, 1.0, src1, 1, src2, 1);

  ERL_NIF_TERM result = MakeDoubleNdList(env, src2, shape1, 0, size_a);

  enif_free(src1);
  enif_free(src2);

  return result;
}

static ERL_NIF_TERM nif_subtract(ErlNifEnv *env, int, const ERL_NIF_TERM argv[]) {
  int size_a = 1;
  int dim_a = 0;
  int size_b = 1;
  int dim_b = 0;
  std::vector<size_t> shape1;
  std::vector<size_t> shape2;
  GetLengthVector(env, argv[0], shape1, dim_a, size_a);
  GetLengthVector(env, argv[1], shape2, dim_b, size_b);

  double *src1 = (double *)enif_alloc(sizeof(double) * size_a);
  double *src2 = (double *)enif_alloc(sizeof(double) * size_b);
  if (GetDoubleArrayNd(env, argv[0], dim_a, 0, src1) == -1) {
    return argv[0];
  }
  if (GetDoubleArrayNd(env, argv[1], dim_b, 0, src2) == -1) {
    return argv[1];
  }
  if (size_a != size_b) {
    size_t copy_cnt = size_a / size_b;
    double *broadcast = (double *)enif_alloc(sizeof(double) * size_a);
    for (size_t i = 0; i < copy_cnt; ++i) {
      memcpy(&broadcast[i * size_b], src2, sizeof(double) * size_b);
    }
    enif_free(src2);
    src2 = broadcast;
  }
  cblas_daxpy(size_a, -1.0, src2, 1, src1, 1);

  ERL_NIF_TERM result = MakeDoubleNdList(env, src1, shape1, 0, size_a);

  enif_free(src1);
  enif_free(src2);

  return result;
}

static ErlNifFunc nif_funcs[] = {
    // {erl_function_name, erl_function_arity, c_function, dirty_flag}
    {"nif_transpose2d", 1, nif_transpose2d, 0},
    {"nif_transpose4d", 5, nif_transpose4d, 0},
    {"nif_dot", 2, nif_dot, 0},
    {"nif_dot_nt", 2, nif_dot_nt, 0},
    {"nif_dot_tn", 2, nif_dot_tn, 0},
    {"nif_add", 2, nif_add, 0},
    {"nif_subtract", 2, nif_subtract, 0}};

ERL_NIF_INIT(Elixir.Broca.Nif.NN, nif_funcs, NULL, NULL, NULL, NULL)
