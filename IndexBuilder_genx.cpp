
#include <cm/cm.h>
#include <cm/cmtl.h>
#include "K2tree.h"
#include "bits.h"
#include "radix.h"

inline _GENX_ uint32_t cnt_naive(vector<uint32_t, SIMD_SIZE> x)
{
  uint32_t res = 0;
  for (int j = 0; j < SIMD_SIZE; ++j)
    for (int i = 0; i < 32; ++i) {
      res += x[j] & 1;
      x[j] >>= 1;
    }
  return res;
}

inline _GENX_ uint32_t cnt32_naive(uint32_t x)
{
  uint32_t res = 0;
  for (int i = 0; i < 32; ++i) {
    res += x & 1;
    x >>= 1;
  }
  return res;
}

inline _GENX_ uint32_t cnt11_naive(uint64_t x)
{
  uint32_t res = 0;
  for (int i = 0; i < 64; ++i) {
    if ((x & 3) == 3) {
      ++res;
      ++i;
      x >>= 1;
    }
    x >>= 1;
  }
  return res;
}

inline _GENX_ uint32_t sel_naive(uint64_t x, uint32_t i)
{
  uint32_t pos = 0;
  while (x) {
    i -= x & 1;
    x >>= 1;
    if (!i) break;
    ++pos;
  }
  return pos;
}

inline _GENX_ uint32_t sel11_naive(uint64_t x, uint32_t i)
{
  for (uint32_t j = 0; j < 63; ++j) {
    if ((x & 3) == 3) {
      i--;
      if (!i) return j + 1;
      x >>= 1;
      ++j;
    }
    x >>= 1;
  }
  return 63;
}

inline _GENX_ uint32_t hi_naive(uint32_t x)
{
  uint32_t res = 31;
  while (x) {
    if (x & 0x80000000U) {
      return res;
    }
    --res;
    x <<= 1;
  }
  return 0;
}

inline _GENX_ uint32_t hi_naive(vector<uint32_t, SIMD_SIZE> x)
{
  for (int i = SIMD_SIZE - 1; i >= 0; i--) {
    uint32_t res = (i + 1) * 32 - 1;
    while (x[i]) {
      if (x[i] & 0x80000000U) {
        return res;
      }
      --res;
      x[i] <<= 1;
    }
  }
  return 0;
}

inline _GENX_ bool test_hi()
{
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t x = 1U << i;
    if (hi_naive(x) != hi(x)) {
      printf("hi_naive(x)=%d hi(x)=%d\n", hi_naive(x), hi(x));
      return false;
    }
  }
  return true;
}

inline _GENX_ bool test_vector_hi()
{
  vector<uint32_t, SIMD_SIZE> x = 0;
  for (uint32_t j = 0; j < SIMD_SIZE; j++) {
    x = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      x[j] = 1U << i;
      if (hi_naive(x) != hi(x)) {
        printf("hi_naive=%d  hi=%d\n", hi_naive(x), hi(x));
        return false;
      }
    }
  }
  return true;
}

inline _GENX_ uint32_t lo_naive(uint32_t x)
{
  if (x & 1)
    return 0;
  x >>= 1;
  for (int i = 1; i < 32; ++i)
    if (x & 1)
      return i;
    else
      x >>= 1;
  return 31;
}

inline _GENX_ uint32_t lo_naive(vector<uint32_t, SIMD_SIZE> x)
{
  int cont = 0;
  for (int j = 0; j < SIMD_SIZE; j++)
    for (int i = 0; i < 32; ++i, cont++)
      if (x[j] & 1)
        return cont;
      else
        x[j] >>= 1;
  return cont;
}

inline _GENX_ bool test_vector_cnt()
{
  vector<uint32_t, SIMD_SIZE> x = 0;
  for (uint32_t j = 0; j < SIMD_SIZE; j++) {
    x = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      x[j] = 1 << i;
      if (cnt_naive(x) != cnt(x)) {
        printf("cnt_naive=%d  cnt=%d\n", cnt_naive(x), cnt(x));
        return false;
      }
    }
  }
  return true;
}

inline _GENX_ bool test_cnt32()
{
  uint32_t x = 0;
  for (uint32_t i = 0; i < 32; ++i) {
    x = 1 << i;
    if (cnt32_naive(x) != cnt32(x)) {
      printf("cnt32_naive=%d  cnt32=%d\n", cnt32_naive(x), cnt32(x));
      return false;
    }
  }
  return true;
}

_GENX_ bool test_sel()
{
  uint64_t x = 1ULL;
  for (uint32_t i = 0; i < 64; ++i) {
    if (i != sel(x << i, 1)) {
      printf("i=%d != sel=%d\n", i, sel(x << i, 1));
      return false;
    }
  }
  //for (uint64_t i = 0; i < this->m_data.size(); ++i) {
  //uint64_t x = this->m_data[i];
  /*
  uint64_t x = 0;
  uint32_t ones = 0;
  for (uint32_t j = 0; j<64; ++j) {
    if ((x >> j) & 1) {
      ++ones;
      if (j != sel(x, ones)) {
        printf("j=%d != sel=%d\n", j, sel(x, ones));
        return false;
      }
    }
  }*/
  //}
  return true;
}



inline _GENX_ bool test_lo()
{
  for (uint64_t i = 0; i < 32; ++i) {
    uint32_t x = 1 << i;
    if (lo_naive(x) != lo(x)) {
      printf("lo_naive=%d  lo=%d\n", lo_naive(x), lo(x));
      return false;
    }
  }
  return true;
}

inline _GENX_ bool test_vector_lo()
{
  vector<uint32_t, SIMD_SIZE> x = 0;
  for (uint32_t j = 0; j < SIMD_SIZE; j++) {
    x = 0;
    for (uint32_t i = 0; i < 32; ++i) {
      x[j] = 1 << i;
      if (lo_naive(x) != lo(x)) {
        printf("lo_naive=%d  lo=%d\n", lo_naive(x), lo(x));
        return false;
      }
    }
  }
  return true;
}

inline _GENX_ uint32_t rev_naive(uint32_t x)
{
  uint32_t y = 0;
  for (size_t i = 0; i < 32; i++) {
    if (x&(1U << i)) {
      y |= (1U << (31 - i));
    }
  }
  return y;
}

inline _GENX_ vector<uint32_t, SIMD_SIZE> rev_naive(vector<uint32_t, SIMD_SIZE> x)
{
  vector<uint32_t, SIMD_SIZE> y = 0;
  for (int xx = 0, yy = SIMD_SIZE - 1; xx < SIMD_SIZE && yy >= 0; xx++, yy--) {
    for (int i = 0; i < 32; i++) {
      if (x[xx] & (1U << i)) {
        y[xx] |= (1U << (31 - i));
      }
    }
  }
  return y;
}

inline _GENX_ bool test_rev()
{
  uint32_t x = 0x80808080U;
  uint32_t rx = rev(x);
  if (rev_naive(x) != rx) {
    printf("rev_naive(x)=%d  rev(x)=%d\n", rev_naive(x), rx);
    return false;
  }
  if (x != rev(rx)) {
    printf("x=%d  rev(rx)=%d\n", x, rev(rx));
    return false;
  }
  return true;
}

inline _GENX_ bool test_vector_rev()
{
  vector<uint32_t, SIMD_SIZE> x;
  uint32_t base = 0x80808080U;
  for (int i = 0; i < SIMD_SIZE; i++)
    x[i] = base >> i;
  vector<uint32_t, SIMD_SIZE> rx = rev(x);
  vector<uint32_t, SIMD_SIZE> rn = rev_naive(x);
  if ((rn != rx).any()) {
    printf("rev_naive(x) !=  rev(x)\n");
    return false;
  }

  if ((x != rev(rx)).any()) {
    printf("x != rev(rx)\n");
    return false;
  }
  return true;
}


_GENX_MAIN_ void cmk_bits_test() {
  if (test_vector_cnt())
    printf(":: TEST vector_cnt PASSED\n");
  else
    printf(":: TEST vector_cnt FAILED\n");

  if (test_cnt32())
    printf(":: TEST cnt32 PASSED\n");
  else
    printf(":: TEST cnt32 FAILED\n");

  if (test_sel())
    printf(":: TEST sel PASSED\n");
  else
    printf(":: TEST sel FAILED\n");

  if (test_hi())
    printf(":: TEST hi PASSED\n");
  else
    printf(":: TEST hi FAILED\n");

  if (test_vector_hi())
    printf(":: TEST vector_hi PASSED\n");
  else
    printf(":: TEST vector_hi FAILED\n");

  if (test_vector_lo())
    printf(":: TEST vector_lo PASSED\n");
  else
    printf(":: TEST vector_lo FAILED\n");

  if (test_lo())
    printf(":: TEST lo PASSED\n");
  else
    printf(":: TEST lo FAILED\n");

  if (test_rev())
    printf(":: TEST rev PASSED\n");
  else
    printf(":: TEST rev FAILED\n");

  if (test_vector_rev())
    printf(":: TEST vector_rev PASSED\n");
  else
    printf(":: TEST vector_rev FAILED\n");

  vector<uint32_t, 4> vec = 255;
  uint32_t x = 129;
  vector<uint32_t, 4> res = cm_bf_insert<uint32_t>(10, 5, x, vec);
  printf("vec %d %d %d %d\n", res[0], res[1], res[2], res[3]);
}

template<typename ty, unsigned int size>
inline _GENX_ void cmk_read(SurfaceIndex index, unsigned int offset, vector_ref<ty, size> v, uint total_threads) {
#pragma unroll
  for (unsigned int i = 0; i < size; i += BLOCK_SZ) {
    read(index, offset + (i * total_threads) * sizeof(ty), v.template select<BLOCK_SZ, 1>(i));
  }
}

template<typename ty, unsigned int size>
inline _GENX_ void cmk_read(SurfaceIndex index, unsigned int offset, vector_ref<ty, size> v) {
#pragma unroll
  for (unsigned int i = 0; i < size; i += 32) {
    read(DWALIGNED(index), offset + (i) * sizeof(ty), v.template select<32, 1>(i));
  }
}

template<typename ty, unsigned int size>
inline _GENX_ void cmk_write(SurfaceIndex index, unsigned int offset, vector_ref<ty, size> v) {
#pragma unroll
  for (unsigned int i = 0; i < size; i += 32) {
    write(index, offset + i * sizeof(ty), v.template select<32, 1>(i));
  }
}

template<typename ty, unsigned int size>
inline _GENX_ void cmk_write_long(SurfaceIndex index, unsigned int offset, vector_ref<ty, size> v) {
#pragma unroll
  for (unsigned int i = 0; i < size; i += 16) {
    write(index, offset + i * sizeof(ty), v.template select<16, 1>(i));
  }
}


_GENX_MAIN_ void cmk_last_levels_construction(SurfaceIndex matrix_in, SurfaceIndex L_out, SurfaceIndex T_out, uint total_threads, uint x, uint y) {

  // h_pos indicates which 256-element chunk the kernel is processing
  uint h_pos = x*(BLOCK_SZ*BLOCK_SZ*(total_threads)) + y*BLOCK_SZ;
  //uint h_pos = total_threads;
  // each thread handles K2_ENTRIES entries.
  unsigned int offset = (h_pos) << 2;
  vector<unsigned int, K2_ENTRIES> submatrix;
  cmk_read<unsigned int, K2_ENTRIES>(matrix_in, offset, submatrix, total_threads);
  matrix_ref<unsigned int, BLOCK_SZ, BLOCK_SZ> smatrix = submatrix.format<unsigned int, BLOCK_SZ, BLOCK_SZ>();

  bool active = false;
  vector<uint, 4> block = 0;
  vector<ushort, 4> T0 = 0;
  vector<ushort, 16> T1 = 0;
  vector<ushort, 64> L = 0;
  uint L_index = 0, T_index = 0;

#pragma unroll
  for (uint i = 0; i < BLOCK_SZ; i += K2_VALUE*K2_VALUE) {
    for (uint j = 0; j < BLOCK_SZ; j += K2_VALUE*K2_VALUE) {
      if ((smatrix.select<4, 1, 4, 1>(i, j) == 1).any()) {
        T0[i / 2 + j / 4] = 1;
        // sub block 0
        if ((smatrix.select<2, 1, 2, 1>(i, j) == 1).any()) {
          T1[T_index++] = 1;
          L.select<4, 1>(L_index) = smatrix.select<2, 1, 2, 1>(i, j);
          L_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 1
        if ((smatrix.select<2, 1, 2, 1>(i, j + 2) == 1).any()) {
          T1[T_index++] = 1;
          L.select<4, 1>(L_index) = smatrix.select<2, 1, 2, 1>(i, j + 2);
          L_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 2
        if ((smatrix.select<2, 1, 2, 1>(i + 2, j) == 1).any()) {
          T1[T_index++] = 1;
          L.select<4, 1>(L_index) = smatrix.select<2, 1, 2, 1>(i + 2, j);
          L_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 3
        if ((smatrix.select<2, 1, 2, 1>(i + 2, j + 2) == 1).any()) {
          T1[T_index++] = 1;
          L.select<4, 1>(L_index) = smatrix.select<2, 1, 2, 1>(i + 2, j + 2);
          L_index += 4;
        }
        else
          T1[T_index++] = 0;
      }
    }
  }


  // compress L and write it into Surface
  // 64 edge values can be store in 2 dwords + 1 for number of bits
  if ((L == 1).any()) {
    vector<uint, 4> L_compressed = 0;
    L_compressed[0] = L_index;
    L_compressed[1] = cm_pack_mask(L.select<32, 1>(0));
    L_compressed[2] = cm_pack_mask(L.select<32, 1>(32));
    write(L_out, (x * (BLOCK_SZ*BLOCK_SZ / WORD_SZ + 2) * (total_threads)+y * (BLOCK_SZ*BLOCK_SZ / WORD_SZ + 2)) * 4, L_compressed);
  }
  // compress T and write it into Surface
  // T1
  // 16 edge values can be store in 1 dword + 1 for number of bits
  if ((T1 == 1).any()) {
    vector<uint, 4> T1_compressed = 0;
    T1_compressed[1] = cm_pack_mask(T1);
    T1_compressed[0] = T_index;
    write(T_out, (total_threads*total_threads * 4 + (x * 4 * (total_threads)+y * 4)) * 4, T1_compressed); // x*4*(threads/2) + y*4 ( 1 dword for values + 1 dword for bitcount)
  }
  // T0
  // 4 edge values can be store in 1 dword + 1 for number of bits
  if ((T0 == 1).any()) {
    vector<uint, 4> T0_compressed = 0;
    T0_compressed[0] = 4;
    T0_compressed[1] = cm_pack_mask(T0);
    write(T_out, (x * 4 * (total_threads)+y * 4) * 4, T0_compressed); // x*4*(threads/2) + y*4 ( 1 dword for values + 1 dword for bitcount)
  }


  // Write matrix_out
  // Every thread writes only one 1-0 final result to the matrix
  if ((T0 == 1).any())
    write(matrix_in, x * (total_threads), y, 1);
  else
    write(matrix_in, x* (total_threads), y, 0);
  /*
  if (x == 0 && y == 3) {
    //printf("thread %d T0[%d,%d,%d,%d]\n", h_pos, T0[0], T0[1], T0[2], T0[3]);
    //printf("thread %d T1[%d,%d,%d,%d  %d,%d,%d,%d ]\n", h_pos, T1[0], T1[1], T1[2], T1[3], T1[4], T1[5], T1[6], T1[7]);
    printf("thread %d L[%d,%d,%d,%d  %d,%d,%d,%d %d,%d,%d,%d]\n", h_pos, L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9], L[10], L[11]);
    //printf("thread %d L_compressed [count %d  %d %d]\n", h_pos, L_compressed[0], L_compressed[1], L_compressed[2]);
  }*/
}

_GENX_MAIN_ void cmk_mid_levels_construction(SurfaceIndex matrix_in, SurfaceIndex T_out, uint total_threads, uint x, uint y) {
  // h_pos indicates which 256-element chunk the kernel is processing
  uint h_pos = x*(BLOCK_SZ*BLOCK_SZ*(total_threads)) + y*BLOCK_SZ;

  //uint h_pos = total_threads;
  // each thread handles K2_ENTRIES entries.
  unsigned int offset = (h_pos) << 2;

  vector<unsigned int, K2_ENTRIES> submatrix;
  cmk_read<unsigned int, K2_ENTRIES>(matrix_in, offset, submatrix, total_threads);
  matrix_ref<unsigned int, BLOCK_SZ, BLOCK_SZ> smatrix = submatrix.format<unsigned int, BLOCK_SZ, BLOCK_SZ>();

  bool active = false;
  vector<uint, 4> block = 0;
  vector<ushort, 4> T0 = 0;
  vector<ushort, 16> T1 = 0;
  vector<ushort, 64> T2 = 0;
  vector<ushort, 64> subT = 0;
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      subT.select<4, 1>(i * 2 + j * 2) = smatrix.select<2, 1, 2, 1>(i * 2, j * 2);
    }
  }
  uint T2_index = 0, T_index = 0;
  write(T_out, 0, subT);
  for (uint i = 0; i < BLOCK_SZ; i += K2_VALUE*K2_VALUE) {
    for (uint j = 0; j < BLOCK_SZ; j += K2_VALUE*K2_VALUE) {
      if ((smatrix.select<4, 1, 4, 1>(i, j) == 1).any()) {
        T0[i / 2 + j / 4] = 1;
        // sub block 0
        if ((smatrix.select<2, 1, 2, 1>(i, j) == 1).any()) {
          T1[T_index++] = 1;
          T2.select<4, 1>(T2_index) = smatrix.select<2, 1, 2, 1>(i, j);
          T2_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 1
        if ((smatrix.select<2, 1, 2, 1>(i, j + 2) == 1).any()) {
          T1[T_index++] = 1;
          T2.select<4, 1>(T2_index) = smatrix.select<2, 1, 2, 1>(i, j + 2);
          T2_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 2
        if ((smatrix.select<2, 1, 2, 1>(i + 2, j) == 1).any()) {
          T1[T_index++] = 1;
          T2.select<4, 1>(T2_index) = smatrix.select<2, 1, 2, 1>(i + 2, j);
          T2_index += 4;
        }
        else
          T1[T_index++] = 0;
        // sub block 3
        if ((smatrix.select<2, 1, 2, 1>(i + 2, j + 2) == 1).any()) {
          T1[T_index++] = 1;
          T2.select<4, 1>(T2_index) = smatrix.select<2, 1, 2, 1>(i + 2, j + 2);
          T2_index += 4;
        }
        else
          T1[T_index++] = 0;
      }
    }
  }

  // compress T2 and write it into Surface
  // 64 values can be store in 2 dwords + 1 for number of bits
  if ((T2 == 1).any()) {
    vector<uint, 4> T2_compressed = 0;
    T2_compressed[0] = T2_index;
    T2_compressed[1] = cm_pack_mask(T2.select<32, 1>(0));
    T2_compressed[2] = cm_pack_mask(T2.select<32, 1>(32));
    write(T_out, (x * (BLOCK_SZ*BLOCK_SZ / WORD_SZ + 2) * (total_threads)+y * (BLOCK_SZ*BLOCK_SZ / WORD_SZ + 2)) * 4, T2_compressed);
  }
  // compress T and write it into Surface
  // T1
  // 16 values can be store in 1 dwords + 1 for number of bits
  if ((T1 == 1).any()) {
    vector<uint, 4> T1_compressed = 0;
    T1_compressed[1] = cm_pack_mask(T1);
    T1_compressed[0] = T_index;
    write(T_out, (total_threads*total_threads * 4 + (x * 4 * (total_threads)+y * 4)) * 4, T1_compressed); // x*4*(threads/2) + y*4 ( 1 dword for values + 1 dword for bitcount)
  }
  // T0
  // 4 values can be store in 1 dwords + 1 for number of bits
  if ((T0 == 1).any()) {
    vector<uint, 4> T0_compressed = 0;
    T0_compressed[0] = 4;
    T0_compressed[1] = cm_pack_mask(T0);
    write(T_out, (x * 4 * (total_threads)+y * 4) * 4, T0_compressed); // x*4*(threads/2) + y*4 ( 1 dword for values + 1 dword for bitcount)
  }


  // Write matrix_out
  // Every thread writes only one 1-0 final result to the matrix
  if ((T0 == 1).any())
    write(matrix_in, x * (total_threads), y, 1);
  else
    write(matrix_in, x* (total_threads), y, 0);

  //printf("thread %d T0[%d,%d,%d,%d]\n", h_pos, T0[0], T0[1], T0[2], T0[3]);
  //printf("thread %d T1[%d,%d,%d,%d  %d,%d,%d,%d ]\n", h_pos, T1[0], T1[1], T1[2], T1[3], T1[4], T1[5], T1[6], T1[7]);
  //printf("thread %d L[%d,%d,%d,%d  %d,%d,%d,%d %d,%d,%d,%d]\n", h_pos, L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9], L[10], L[11]);
  //printf("thread %d L_compressed [count %d   %d %d]\n", h_pos, L_compressed[0], L_compressed[1], L_compressed[2]);
}

static const ushort OFFSETS_X[16] = { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
static const ushort OFFSETS_Y[16] = { 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62 };
static const ushort SHIFTS[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 };

_GENX_MAIN_ void cmk_generate_morton_numbers(SurfaceIndex input, SurfaceIndex ouput) {

  vector<uint, 16> widths = 2;
  vector<uint, 16> offsetsX(OFFSETS_X), offsetsY(OFFSETS_Y), shifts(SHIFTS);
  vector<uint, 64> val;
  vector<uint, 32> src = 0;
  vector<unsigned long long, 32> result = 0;

  vector<unsigned long long, 32> aux = 0;
  unsigned long long mask = 3;

    cmk_read<uint, 64>(input, 0, val);

    for (uint j = 0; j < 32; j += 32) {
      for (uint k = 0; k < 64; k+=4) { // x
        result.select<32, 1>(0) |= cm_bf_insert<unsigned long long>(2, k, val.select<32, 2>(j), src);
        val.select<32, 2>(j) >>= 2;
      }
    }


    for (uint j = 1; j < 64; j += 32) {
      for (uint k = 2; k < 32; k += 4) { // y
        result.select<32, 1>(0) |= cm_bf_insert<unsigned long long>(2, k, val.select<32, 2>(j), src);
        val.select<32, 2>(j) >>= 2;
      }
    }


    cmk_write_long<unsigned long long, 32>(ouput, 0, result);
}


static const ushort OFFSETS[8] = { 0, 4, 8, 12, 16, 20, 24, 28 };


static const uint PRECOMPUTED_SUM[256] = { 0, 0, 0, 16, 0, 256, 256, 528, 0, 4096, 4096, 8208, 4096, 8448, 8448, 12816, 0, 65536, 65536, 131088,
65536, 131328, 131328, 197136, 65536, 135168, 135168, 204816, 135168, 205056, 205056, 274960, 0, 1048576, 1048576, 2097168, 1048576, 2097408,
2097408, 3146256, 1048576, 2101248, 2101248, 3153936, 2101248, 3154176, 3154176, 4207120, 1048576, 2162688, 2162688, 3276816, 2162688, 3277056,
3277056, 4391440, 2162688, 3280896, 3280896, 4399120, 3280896, 4399360, 4399360, 5517840, 0, 16777216, 16777216, 33554448, 16777216, 33554688,
33554688, 50332176, 16777216, 33558528, 33558528, 50339856, 33558528, 50340096, 50340096, 67121680, 16777216, 33619968, 33619968, 50462736, 33619968,
50462976, 50462976, 67306000, 33619968, 50466816, 50466816, 67313680, 50466816, 67313920, 67313920, 84161040, 16777216, 34603008, 34603008, 52428816,
34603008, 52429056, 52429056, 70255120, 34603008, 52432896, 52432896, 70262800, 52432896, 70263040, 70263040, 88093200, 34603008, 52494336, 52494336,
70385680, 52494336, 70385920, 70385920, 88277520, 52494336, 70389760, 70389760, 88285200, 70389760, 88285440, 88285440, 106181136, 0, 268435456,
268435456, 536870928, 268435456, 536871168, 536871168, 805306896, 268435456, 536875008, 536875008, 805314576, 536875008, 805314816, 805314816,
1073754640, 268435456, 536936448, 536936448, 805437456, 536936448, 805437696, 805437696, 1073938960, 536936448, 805441536, 805441536, 1073946640,
805441536, 1073946880, 1073946880, 1342452240, 268435456, 537919488, 537919488, 807403536, 537919488, 807403776, 807403776, 1076888080, 537919488,
807407616, 807407616, 1076895760, 807407616, 1076896000, 1076896000, 1346384400, 537919488, 807469056, 807469056, 1077018640, 807469056, 1077018880,
1077018880, 1346568720, 807469056, 1077022720, 1077022720, 1346576400, 1077022720, 1346576640, 1346576640, 1616130576, 268435456, 553648128, 553648128,
838860816, 553648128, 838861056, 838861056, 1124074000, 553648128, 838864896, 838864896, 1124081680, 838864896, 1124081920, 1124081920, 1409298960,
553648128, 838926336, 838926336, 1124204560, 838926336, 1124204800, 1124204800, 1409483280, 838926336, 1124208640, 1124208640, 1409490960, 1124208640,
1409491200, 1409491200, 1694773776, 553648128, 839909376, 839909376, 1126170640, 839909376, 1126170880, 1126170880, 1412432400, 839909376, 1126174720,
1126174720, 1412440080, 1126174720, 1412440320, 1412440320, 1698705936, 839909376, 1126236160, 1126236160, 1412562960, 1126236160, 1412563200, 1412563200,
1698890256, 1126236160, 1412567040, 1412567040, 1698897936, 1412567040, 1698898176, 1698898176, 1985229328 };



template<typename ty, unsigned int size>
inline _GENX_ void write_int_tree(vector_ref<ty, size> tree, vector_ref<ushort, 2> indices,
  uint x, uint len) {

  tree[indices[0]] |= (x << (indices[1] * 4));
  tree[indices[0] + 1] = (x >> (32 - indices[1] * 4 ));

  indices[1] += len;
  indices[0] += indices[1] / 8;
  indices[1] %= 8;
}

_GENX_MAIN_ void cmk_construction_from_edges(SurfaceIndex input, uint offset_start, uint start, uint end) {

  uint total_elems = end - start;
  vector<uint, 32> edges; // worst case, all vertices have links
  vector<uint, 32> shifts;
  vector<ushort, 32> mask;
  vector<uint, 2> range, div; //cStart, cEnd

  uint i = 0, sum;
  uint cumSum = 0;
  cmk_read<uint, 32>(input, offset_start << 2, edges);
  range[0] = edges[0];
  uint diff = 32 - (range[0] % 32);
  range[1] = range[0] + diff;

  vector<uint, 16> tree = 0;

  vector<uint, 256> precomputed(PRECOMPUTED_SUM);
  // L vector
  vector<ushort, 256> L = 0; // Index = 0

  // T vectors
  vector<ushort, 64> T2 = 0; uint t2_idx = 0; // Index = 9
  vector<ushort, 32> T1 = 0; uint t1_idx = 0; // Index = 12  (SZ should be 16)  ==== ERROR: cm_pack_mask of a vector<ushort, 16> ====
  vector<ushort, 32> T0 = 0; uint t0_idx = 0; // Index = 14 (SZ should be 4)


  // check every submatrix
  // first submatrix
  vector<uint, 32> subvec;
  vector<uint, 8> subsubvec, resultVec, precomputedMask;
  vector<ushort, 8> offsets(OFFSETS), width = 4, submask, subshifts;

  ushort current8th = 0, l_idx = 0;

  vector<uint, 2> widths = 0, offs = 0, vals;

  // first block
  subvec = edges.select<32, 1>(i);
  mask = (subvec >= range[0]) & (subvec < range[1]);
  if (mask.any()) {
    t0_idx = subvec[0] / 64;
    shifts = 1 << subvec;
    subvec = 0;
    subvec.merge(shifts, mask);
    sum = cm_sum<uint>(subvec); // final result
    i += cm_cbit(cm_pack_mask(mask));

    // store L
    // L - 32 bits
    L.select<32, 1>(l_idx) = cm_unpack_mask<uint, 32>(sum);
    l_idx += 32;
    // store T
    // T2 - 8 bits
    subsubvec = sum;
    resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
    submask = (resultVec != 0);
    T2.select<8, 1>(t2_idx) = submask;
    t2_idx += 8;
    // T1 - 2 bits
    T1[t1_idx++] = (submask.select<4, 1>(0)).any();
    T1[t1_idx++] = (submask.select<4, 1>(4)).any();

    // T0 - 1 bit

  }
  range[0] += diff;
  range[1] += 32;

  

  // second and rest of the blocks
  for (uint j = 0; j < 7; j++) {

    subvec = edges.select<32, 1>(i);
    mask = (subvec >= range[0]) & (subvec < range[1]);
    if (mask.any()) {
      t0_idx = subvec[0] / 64;
      shifts = 1 << subvec;
      subvec = 0;
      subvec.merge(shifts, mask);
      sum = cm_sum<uint>(subvec); // final result
      i += cm_cbit(cm_pack_mask(mask));

      // store L
      // L - 32 bits
      L.select<32, 1>(l_idx) = cm_unpack_mask<uint, 32>(sum);
      l_idx += 32;
      // store T
      // T2 - 8 bits
      subsubvec = sum;
      resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
      submask = (resultVec != 0);
      T2.select<8, 1>(t2_idx) = submask;
      t2_idx += 8;
      // T1 - 2 bits
      T1[t1_idx++] = (submask.select<4, 1>(0)).any();
      T1[t1_idx++] = (submask.select<4, 1>(4)).any();

      // T0 - 1 bit

    }
    range += 32;
  }



  // compress vectors into bitstrings
  // compress L
  uint val;
  vector<ushort, 2> L_indices = 0;
  L_indices[0] = L_INDEX + 1;
//#pragma unroll
  for (uint j = 0; j < 256; j += 32) {
    val = cm_pack_mask(L.select<32, 1>(j));
    if (!val)
      continue;
    subsubvec = val;
    resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
    submask = (resultVec != 0);
    precomputedMask = precomputed[cm_pack_mask(submask)]; // compress 4-bit substrings with only 0's
    subshifts = cm_bf_extract<uint>(width, offsets, precomputedMask);
    resultVec <<= (subshifts * 4);
    val = cm_sum<uint>(resultVec); // compress 32-bit string

    // store into final tree;
    write_int_tree<uint, 16>(tree, L_indices, val, cm_cbit(cm_pack_mask(submask)));

  }
  tree[L_INDEX] = (L_indices[0] - (L_INDEX + 1)) * 8 + L_indices[1];

  // compress T2
  vector<ushort, 2> T2_indices = 0;
  T2_indices[0] = T2_INDEX + 1;
#pragma unroll
  for (uint j = 0; j < 64; j += 32) {
    val = cm_pack_mask(T2.select<32, 1>(j));
    if (!val)
      continue;
    subsubvec = val;
    resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
    submask = (resultVec != 0);
    precomputedMask = precomputed[cm_pack_mask(submask)]; // compress 4-bit substrings with only 0's
    subshifts = cm_bf_extract<uint>(width, offsets, precomputedMask);
    resultVec <<= (subshifts * 4);
    val = cm_sum<uint>(resultVec); // compress 32-bit string

                                   // store into final tree;
    write_int_tree<uint, 16>(tree, T2_indices, val, cm_cbit(cm_pack_mask(submask)));
  }
  tree[T2_INDEX] = (T2_indices[0] - (T2_INDEX + 1)) * 8 + T2_indices[1];

  // compress T1
  vector<ushort, 2> T1_indices = 0;
  T1_indices[0] = T1_INDEX + 1;
  val = cm_pack_mask(T1);
  if (val) {
    subsubvec = val;
    resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
    submask = (resultVec != 0);
    precomputedMask = precomputed[cm_pack_mask(submask)]; // compress 4-bit substrings with only 0's
    subshifts = cm_bf_extract<uint>(width, offsets, precomputedMask);
    resultVec <<= (subshifts * 4);
    val = cm_sum<uint>(resultVec); // compress 32-bit string

                                   // store into final tree;
    write_int_tree<uint, 16>(tree, T1_indices, val, cm_cbit(cm_pack_mask(submask)));

    tree[T1_INDEX] = (T1_indices[0] - (T1_INDEX + 1)) * 8 + T1_indices[1];

    // compress T0
    val = cm_pack_mask(submask);
    tree[T0_INDEX] = 1;
    tree[T0_INDEX + 1] = val;
  }

  printf("tree [%u, %u, %u, %u, %u, %u, %u, %u ]\n", tree[0], tree[1], tree[2], tree[3], tree[4], tree[5], tree[6], tree[7]);
  printf("tree [%u, %u, %u, %u, %u, %u, %u, %u ]\n", tree[8], tree[9], tree[10], tree[11], tree[12], tree[13], tree[14], tree[15]);
  // write final result

}

// Stack structure used for keeping track of search path
// TODO: resize vector if it becomes full
inline _GENX_ int push(vector_ref<uint, STACK_SZ> stack, uint elem) {
  if (stack[0] != STACK_SZ - 1) {
    // not full
    stack[0]++;
    stack[stack[0]] = elem;
    return 1;
  }
  else {
    // throw exception
#if DEBUG_MODE
    printf("stack full!!\n");
#endif
    return 0;
  }
}

inline _GENX_ int pop(vector_ref<uint, STACK_SZ> stack) {
  if (stack[0] != 0) {
    uint elem = stack[stack[0]];
    stack[0]--;
    return elem;
  }
  else // no elements
    return -1;
}

inline _GENX_ int top(vector_ref<uint, STACK_SZ> stack) {
  if (stack[0] != 0) {
    return stack[stack[0]];
  }
  else // no elements
    return -1;
}

inline _GENX_ int readInt(SurfaceIndex S, uint idx) {
  vector<uint, 4> ret;
  //uint readDiv = idx / 4;
  //uint readRest = idx % 4;
  //read(S, readDiv * 4 * 4, ret);
  read(DWALIGNED(S), idx * 4, ret);
  //return ret[readRest];
  return ret[0];
}

inline _GENX_ int rank(SurfaceIndex T, SurfaceIndex T_rank, uint idx) {
  // assert( idx <= t_size )
  vector<int, 32> ret;
  int i = idx;
  int p = (int)(i / 512) * 4; // (i/s)
  read(T_rank, p * 4, ret);
  int resp = ret[0];
  int aux = p * 16 / 4; // (p*factor)
  //printf("first rep %u\n", resp);

  read(DWALIGNED(T), aux * 4, ret);
  //uint readDiv = aux / 32;
  //uint readRest = aux % 32;
  //read(T, readDiv * 32 * 4, ret);

  //uint k = readRest;
  uint k = 0;
  for (int a = aux; a < i / WORD_SZ; a++) {
    // TODO: read several dwords at a time

    resp += cnt32(ret[k++]);
    //printf("%u.- rep %u\n", k, resp);
  }
  //read(T, ((uint)i / WORD_SZ)*4, ret);
  resp += cnt32(ret[k] & ((1 << (i & 31)) - 1));
  //printf("%u.- rep %u\n", k, resp);
  return resp;

}



_GENX_MAIN_
void cmk_neighbors_test(SurfaceIndex T, SurfaceIndex L, SurfaceIndex T_rank, uint t_size, uint l_size, uint height, uint k, uint start, uint end) {
  vector<uint, STACK_SZ> stack;
  vector<uint, 4> ret;
  uint div, mod, readDiv, readRest;
  if (l_size == 0 && t_size == 0)
    return;
  // n = k^h / k
  // k^h - dimension n of matrix nxn
  // /k  - to calculate div only once and not for for all parameter again, always (n/k)
  uint n = cm_pow((float)k, (float)height) / k;
  // y = k * i/n
  //printf("n = %d\n", n);
  for (uint i = start; i < end; i++) {
    stack = 0;
    uint y = k * cm_rndd<uint>(i / n);
    //printf("Neighbors for %d: ", i);
    //printf("y = %d\n", y);
    for (unsigned j = 0; j < k; j++) {
      uint nn = n / k;
      uint row = i % n;
      uint col = n * j;
      uint level = y + j;
      push(stack, nn); push(stack, row); push(stack, col); push(stack, level);
      //_neigh(n / k, i % n, n * j, y + j, acc);
      //--------------
      while (top(stack) != -1) { // while stack isn't empty
        //printf("NEXT CALL \n");
        level = pop(stack); col = pop(stack); row = pop(stack); nn = pop(stack);
        //printf("nn = %d\n", nn);
        //printf("row = %d\n", row);
        //printf("col = %d\n", col);
        //printf("level = %d\n", level);
        if (level >= t_size) { // Last level
          div = (level - t_size) / WORD_SZ;
          mod = (level - t_size) % WORD_SZ;
          //readDiv = div / 4;
          //readRest = div % 4;
          //read(L, readDiv * 4, ret);
          if (((readInt(L, div) >> mod) & 0x1) == 1)
            //if (i == 0)printf("%d ", col);
            continue;
        }
        div = level / WORD_SZ;
        mod = (level == 0) ? WORD_SZ - 1 : level % WORD_SZ;
        //readDiv = div;
        //readRest = div % 4;
        //read(T, readDiv * 4, ret);
        //printf("ret[0] = %u  |  ret[0] >> %u = %u\n", readInt(T, div), mod, (readInt(T, div) >> mod));
        if (((readInt(T, div) >> mod) & 0x1) == 1) {
          //printf("rank = %d\n", rank(T, T_rank, level + 1));
          uint yy = rank(T, T_rank, level + 1) * cm_pow((float)k, (float) 2.0) + k * cm_rndd<uint>(row / n);
          //printf("yy = %d\n", yy);
          uint nnn = nn;
          for (int q = k - 1; q >= 0; q--) {
            push(stack, nnn / k); push(stack, row % nnn); push(stack, col + nnn * q); push(stack, yy + q);
            //_neigh(n / k_k, row % n, col + n * j, y + j, acc);
          }
        }
      }
      //-----------
    }
    //if (i == 0)printf("\n");
  }
}

_GENX_MAIN_
void cmk_range_test(SurfaceIndex T, SurfaceIndex L, SurfaceIndex T_rank, uint t_size, uint l_size, uint height, uint k, uint start, uint end) {
  uint row1, row2, col1, col2, div, mod;
  uint s_n, s_row1, s_row2, s_col1, s_col2, s_dr, s_dc, s_z;
  uint totalFound = 0;

  uint x5 = get_thread_origin_x()*start;
  uint y5 = get_thread_origin_y()*start;


  //for (uint it = start; it < end; it++) {
  row1 = x5;
  col1 = y5;
  row2 = x5 + start - 1;
  col2 = y5 + start - 1;
  //printf("Will work at [%d,%d] -> [%d,%d]\n", row1,col1, row2, col2);
  //return;
  uint n = cm_pow((float)k, (float)height) / k;
  vector<uint, STACK_SZ> states = 0;
  // TODO: push several items at a time
  push(states, n); push(states, row1); push(states, row2); push(states, col1); push(states, col2); push(states, 0); push(states, 0); push(states, MAX_INT);

  while (top(states) != -1) {
    // TODO pop several items at a time
    s_z = pop(states); s_dc = pop(states); s_dr = pop(states); s_col2 = pop(states); s_col1 = pop(states); s_row2 = pop(states); s_row1 = pop(states); s_n = pop(states);
    //TODO: peel first loop where z==-1 atm
    if (s_z != MAX_INT && s_z >= t_size) { // Last level
      div = (s_z - t_size) / WORD_SZ;
      mod = (s_z - t_size) % WORD_SZ;
      if (((readInt(L, div) >> mod) & 0x1) == 1) // New edge found [s_dr, s_dc]
        totalFound++;
      //printf("Edge found [%d, %d]\n", s_dr, s_dc);
    }
    else if (s_z == MAX_INT || ((readInt(T, (s_z / WORD_SZ)) >> (s_z % WORD_SZ)) & 0x1) == 1) {

      uint y = rank(T, T_rank, s_z + 1) * k * k;

      for (uint i = s_row1 / s_n; i <= s_row2 / s_n; ++i) {
        uint row1new, row2new;
        //TODO: loop peeling, first iteration and last iteration special
        if (i == s_row1 / s_n) row1new = s_row1 % s_n; else row1new = 0;
        if (i == s_row2 / s_n) row2new = s_row2 % s_n; else row2new = s_n - 1;

        for (uint j = s_col1 / s_n; j <= s_col2 / s_n; ++j) {
          uint col1new, col2new;
          //TODO: loop peeling, first iteration and last iteration special
          if (j == s_col1 / s_n) col1new = s_col1 % s_n; else col1new = 0;
          if (j == s_col2 / s_n) col2new = s_col2 % s_n; else col2new = s_n - 1;

          push(states, s_n / k); push(states, row1new); push(states, row2new); push(states, col1new);
          push(states, col2new); push(states, s_dr + s_n*i); push(states, s_dc + s_n*j); push(states, y + k*i + j);

        }
      }
    }
  }

  //}


}



// There are three kernels:
// 1.	cmk_radix_count: which counts how many elements in each bin locally
//    within each HW thread.
// 2.	prefix sum : which cumulates the number of elements of bins
//    of all threads.
// 3.	cmk_radix_bucket : which reads a chunk of data, 256 elements, bins
//    them into buckets, finally writes elements in each bucket to
//    the output buffer based on the global positions calculated in step 2.


#define TABLE_SZ 16
const uchar init16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
const uchar init_rev16[16] = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
const ushort init_mask[BIN_NUM] = { 0, BIN_SZ, 0, BIN_SZ };

// On GPU.We apply Map-Reduce mechanism to the bin counting process.
// First, we divide input data into chunks and Map computing local bin_cnt
// for a data chunk to each HW thread. Then apply Reduce operation to
// calculate prefix sum for all local bin_cnt.
// cmk_radix_count basically reads in one chunk of data, 256 elements,
// and counts how many elements in each bin.
_GENX_MAIN_ void cmk_radix_count(SurfaceIndex input, SurfaceIndex output, unsigned int n)
{
  // h_pos indicates which 256-element chunk the kernel is processing
  uint h_pos = get_thread_origin_x() + get_thread_origin_y()*MAX_TS_WIDTH;
  // byte offset of the data chunk
  unsigned int offset = (h_pos * BASE_SZ) << 2;
  // to take advantage of SIMD architecture, we process counting 32
  // elements as a batch rather than counting each element serially.
  // Here we create a 4x32 counters. Each time, 32 elements are read.
  // We can view them as 32 lanes of data. Each lane has its own
  // dedicated bin counters.
  matrix<unsigned short, 4, 32> counters;
  // after we are done with 32-element batch counting, we perform sum
  // reduction to calculate the bin counts for 256 elements. The results
  // are in bin_cnt[]
  vector<unsigned int, BIN_NUM> bin_cnt;
  counters = 0;
  unsigned int mask = 0x3 << n; // which 2 bits we want to extract

                                //#pragma unroll
  for (int i = 0; i < BASE_SZ; i += 32) {
    // read and process 32 elements each time
    vector<unsigned int, 32> A;
    cmk_read<unsigned int, 32>(input, offset + i * sizeof(unsigned int), A);
    // extract n-th and (n+1)-th bits out.
    // val is the bin number, data will be put. E.g., val[i] is bin # for A(i)
    vector<unsigned int, 32> val = (A & mask) >> n;
    // row(0) is for bin0 for all 32 lanes.
    // merge operation to increase its own corresponding counters
    // val == 0 indicate which lanes have 0. Only those channels are
    // incrementing.
    counters.row(0).merge(counters.row(0) + 1, val == 0);
    counters.row(1).merge(counters.row(1) + 1, val == 1);
    counters.row(2).merge(counters.row(2) + 1, val == 2);
    counters.row(3).merge(counters.row(3) + 1, val == 3);
  }
  // bin counters for 32 lanes are complete. Perform sum reduction to obtain
  // final bin counters for 256 elements. The most intuitive way is to
  // perform sum reduction for each row, e.g.,
  // bin_cnt[i] = cm_sum<unsigned int>(counters.row(i));
  // although cm_sum intrinsic is doing efficient sum reduction operations
  // however, there are more parallelism we can exploit when doing reduction
  // with all 4 rows together.
  matrix<unsigned short, 4, 16> tmp_sum16;
  // reduction from 32 to 16
  tmp_sum16 = counters.select<4, 1, 16, 1>(0, 0) + counters.select<4, 1, 16, 1>(0, 16);
  matrix<unsigned short, 4, 8> tmp_sum8;
  // reduction from 16 to 8
  tmp_sum8 = tmp_sum16.select<4, 1, 8, 1>(0, 0) + tmp_sum16.select<4, 1, 8, 1>(0, 8);
  matrix<unsigned short, 4, 4> tmp_sum4;
  // reduction from 8 to 4
  tmp_sum4 = tmp_sum8.select<4, 1, 4, 1>(0, 0) + tmp_sum8.select<4, 1, 4, 1>(0, 4);
  matrix<unsigned short, 4, 2> tmp_sum2;
  // reduction from 4 to 2
  tmp_sum2 = tmp_sum4.select<4, 1, 2, 1>(0, 0) + tmp_sum4.select<4, 1, 2, 1>(0, 2);
  // reduction from 2 to 1
  bin_cnt = tmp_sum2.select<4, 1, 1, 1>(0, 0) + tmp_sum2.select<4, 1, 1, 1>(0, 1);
  //printf(" %d %d %d %d\n", bin_cnt[0], bin_cnt[1], bin_cnt[2], bin_cnt[3]);
  // write out count (number of elements in each bin) for each bin
  // bin_cnt[0] = bin_cnt[1] = 110;
  write(output, (h_pos * BIN_NUM) << 2, bin_cnt);
}

_GENX_MAIN_ void cmk_radix_bucket(
  SurfaceIndex input, // input data to be sorted
  SurfaceIndex table, // Prefix sum table
  SurfaceIndex output,  // output for binning result
  unsigned int bin0_cnt, // global bin0 count,
  unsigned int bin1_cnt, // global bin1 count
  unsigned int bin2_cnt, // global bin2 count
  unsigned int bin3_cnt, // global bin3 count
  unsigned int n)     // binning based n-th and (n+1)-th bits
{
  // h_pos indicates which 256-element chunk the kernel is processing
  uint h_pos = get_thread_origin_x() + get_thread_origin_y()*MAX_TS_WIDTH;
  // byte offset of the data chunk
  unsigned int offset = (h_pos * BASE_SZ) << 2;

  vector<unsigned int, BIN_NUM> prefix = 0;
  // loading PrefixSum[h_pos-1]
  // the information tells how many cumulated elements from thread 0 to
  // h_pos-1 in each bin. Thread0 has no previous prefix sum so 0 is
  // initialized.
  if (h_pos != 0) {
    read(table, ((h_pos - 1)*BIN_NUM) << 2, prefix);
  }

  unsigned int mask = 0x3 << n;
  // the location where the next 32 elements can be put in each bin
  vector<unsigned int, BIN_NUM> next;
  next[0] = prefix[0];
  next[1] = bin0_cnt + prefix[1];
  next[2] = bin0_cnt + bin1_cnt + prefix[2];
  next[3] = bin0_cnt + bin1_cnt + bin2_cnt + prefix[3];

  for (int i = 0; i < BASE_SZ; i += 32) {
    // read and process 32 elements at a time
    vector<unsigned int, 32> A;
    cmk_read<unsigned int, 32>(input, offset + i * sizeof(unsigned int), A);
    // calculate bin # for each element
    vector<unsigned short, 32> val = (A & mask) >> n;
    vector<unsigned int, 4> bitset;
    // val has bin # for each element. val == 0 is a 32-element Boolean vector.
    // The ith element is 1 (true) if val[i] == 0, 0 (false) otherwise
    // cm_pack_mask(val == 0) turns the Boolean vector into one unsigned
    // 32-bit value. The i-th value is the corresponding i-th Boolean value.
    bitset(0) = cm_pack_mask(val == 0);
    bitset(1) = cm_pack_mask(val == 1);
    bitset(2) = cm_pack_mask(val == 2);
    bitset(3) = cm_pack_mask(val == 3);
    // calculate how many elements in each bin
    vector<unsigned short, 4> n_elems = cm_cbit<unsigned int>(bitset);

    // calculate prefix sum
    // For each bin, there is a corresponding "next" index pointing to
    // the next available slot. "val == 0" tells us which A elements
    // should be put into bin0. From position 0 to 31,
    // if val[i] == 0 then A[i] will be placed into bin0[next],
    // then bin0[next+1], bin0[next+2], etc.
    // For instance, "val == 0":   0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 -- LSB
    // A[5] is placed in bin0[next], A[6] in bin0[next+1],
    // A[9] in bin0[next+2], A[13] in bin0[next+3]
    // Calculate prefix sum for "val == 0" we get
    // prefix_val0 = 4 4 4 3 3 3 3 2 2 2 1 0 0 0 0 0  --- LSB
    // the 5, 6, 9 and 13-th value of "next + prefix_val0 - 1" is the locations
    // where A[5], A[6], A[9] and A[13] will be stored in bin0.
    matrix<unsigned short, 4, 32> idx;
    idx.row(0) = (val == 0);
    idx.row(1) = (val == 1);
    idx.row(2) = (val == 2);
    idx.row(3) = (val == 3);
    // step 1 of prefix-sum. Sum up every pair of even and odd elements
    // and store the result in even position. In each step, we process 4 bins
    idx.select<4, 1, 16, 2>(0, 1) += idx.select<4, 1, 16, 2>(0, 0);
    // step 2
    idx.select<4, 1, 8, 4>(0, 2) += idx.select<4, 1, 8, 4>(0, 1);
    idx.select<4, 1, 8, 4>(0, 3) += idx.select<4, 1, 8, 4>(0, 1);
    // step 3
    // for a vector: 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
    // this step adds 3 to 4, 5, 6, 7 and adds 11 to 12, 13, 14, 15.
    // replicate<16,8,4,0>(0,3) duplicates 3rd, 11th, 19th, 27th 4 times each
    matrix<unsigned short, 4, 16> t;
    t = idx.replicate<16, 8, 4, 0>(0, 3);
    // Gen ISA describes only one destination stride. That is,
    // one instruction cannot write 4 consecutive elements and then write
    // another consecutive 4 elements with a stride distance. Due to this
    // restriction, a straightforward implementation has to break step 3
    // into 4 instructions, each adding 4 elements. we format matrix of
    // uw type into unsigned long long type. One unsigned long long has 4 uw.
    // The maximum value of prefix sum is 32, only i.e., every bit is set.
    // No overflow will happen during prefix sum computation. One long long
    // type add is equivalent to 4 uw additions. 16 additions of uw types
    // can be collapsed into 4 qword additions. What is more, one add
    // instruction can express those 4 qword additions without running into
    // the destination stride restriction.
    matrix_ref<unsigned long long, 4, 8> m1 = idx.format<unsigned long long, 4, 8>();
    matrix_ref<unsigned long long, 4, 4> t1 = t.format<unsigned long long, 4, 4>();
    m1.select<4, 1, 4, 2>(0, 1) += t1;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // step 4
      idx.select<1, 1, 8, 1>(j, 8) += idx(j, 7);
      idx.select<1, 1, 8, 1>(j, 24) += idx(j, 23);
      // step 5
      idx.select<1, 1, 16, 1>(j, 16) += idx(j, 15);
    }

    // calculate the positions of elements in their corresponding bins
    vector<unsigned int, 32> voff;
    // add bin0 element offsets to the bin0-batch-start
    voff.merge(idx.row(0) + next(0) - 1, val == 0);
    // add bin1 element offsets to the bin1-batch-start
    voff.merge(idx.row(1) + next(1) - 1, val == 1);
    // add bin2 element offsets to the bin2-batch-start
    voff.merge(idx.row(2) + next(2) - 1, val == 2);
    // add bin3 element offsets to the bin3-batch-start
    voff.merge(idx.row(3) + next(3) - 1, val == 3);

    // scatter write, 16-element each
    write(output, 0, voff.select<16, 1>(0), A.select<16, 1>(0));
    write(output, 0, voff.select<16, 1>(16), A.select<16, 1>(16));

    // update the next pointers, move onto the next 32 element
    next += n_elems;
  }
}

