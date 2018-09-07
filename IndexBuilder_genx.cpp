
#include <cm/cm.h>
#include <cm/cmtl.h>
#include "K2tree.h"
#include "bits.h"

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
    read(index, offset + (i) * sizeof(ty), v.template select<32, 1>(i));
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

_GENX_MAIN_ void cmk_construction_from_edges(SurfaceIndex input, uint offset_start, uint start, uint end) {

  uint total_elems = end - start;
  vector<uint, 64> edges; // worst case, all vertices have links
  vector<uint, 32> shifts;
  vector<ushort, 32> mask;
  vector<uint, 2> range; //cStart, cEnd
  range[0] = start;
  range[1] = start + 32;
  uint i = 0, sum;
  uint cumSum = 0;
  cmk_read<uint, 64>(input, offset_start, edges);

  vector<ushort, 32> L;

  vector<uint, 256> precomputed(PRECOMPUTED_SUM);

  // check every submatrix
  // first submatrix
  vector<uint, 32> subvec;
  vector<uint, 8> subsubvec, resultVec, precomputedMask;
  vector<ushort, 8> offsets(OFFSETS), width = 4, submask, subshifts;

  for (uint submatrix = 0; submatrix < 2; submatrix++) {
    subvec = edges.select<32, 1>(i);
    mask = (subvec >= range[0]) & (subvec < range[1]);
    if (mask.any()) {
      shifts = 1 << subvec;
      subvec = 0;
      subvec.merge(shifts, mask);
      sum = cm_sum<uint>(subvec); // final result
      i += cm_cbit(cm_pack_mask(mask));

      subsubvec = sum;
      resultVec = cm_bf_extract<uint>(width, offsets, subsubvec);
      submask = (resultVec != 0);
      precomputedMask = precomputed[cm_pack_mask(submask)];
      subshifts = cm_bf_extract<uint>(width, offsets, precomputedMask);
      resultVec <<= (subshifts * 4);
      sum = cm_sum<uint>(resultVec);
      printf("compressed vec %d\n", sum);
    }
    range += 32;

  }



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
