/* sdsl - succinct data structures library
Copyright (C) 2008 Simon Gog

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/ .
*/
/*! \file bits.hpp
\brief bits.hpp contains the sdsl::bits class.
\author Simon Gog
*/


//! Namespace for the succinct data structure library.

  //! A helper class for bitwise tricks on 64 bit words.
  /*!
  bits is a helper class for bitwise tricks and
  techniques. For the basic tricks and techiques we refer to Donald E. Knuth's
  "The Art of Computer Programming", Volume 4A, Chapter 7.1.3 and
  the informative website of Sean E. Anderson about the topic:
  http://www-graphics.stanford.edu/~seander/bithacks.html .

  We have added new functions like: cnt11 and sel11.

  All members of this class are static variables or methods.
  This class cannot be instantiated.

  \author Simon Gog
  */
typedef uint32_t size_t;



  //struct bits {
   // bits() = delete;
    //! 64bit mask with all bits set to 1.
    constexpr static uint64_t  all_set{ -1ULL };

    //! This constant represents a de Bruijn sequence B(k,n) for k=2 and n=6.
    /*! Details for de Bruijn sequences see
    http://en.wikipedia.org/wiki/De_bruijn_sequence
    deBruijn64 is used in combination with the
    array lt_deBruijn_to_idx.
    */
    constexpr static uint64_t deBruijn64{ 0x0218A392CD3D5DBFULL };

    //! This table maps a 6-bit subsequence S[idx...idx+5] of constant deBruijn64 to idx.
    /*! \sa deBruijn64
    */
    //static vector<uint32_t, 64> lt_deBruijn_to_idx;

    //! Array containing Fibonacci numbers less than \f$2^64\f$.
    //static vector<uint64_t, 92> lt_fib;

    //! Lookup table for byte popcounts.
    //static vector<uint8_t, 256> lt_cnt;

    //! Lookup table for most significant set bit in a byte.
    //static vector<uint32_t, 256> lt_hi;

    //! lo_set[i] is a 64-bit word with the i least significant bits set and the high bits not set.
    /*! lo_set[0] = 0ULL, lo_set[1]=1ULL, lo_set[2]=3ULL...
    */
    //static vector<uint64_t, 65> lo_set;

    //! lo_unset[i] is a 64-bit word with the i least significant bits not set and the high bits set.
    /*! lo_unset[0] = FFFFFFFFFFFFFFFFULL, lo_unset_set[1]=FFFFFFFFFFFFFFFEULL, ...
    */
    //static vector<uint64_t, 65> lo_unset;

    //! Lookup table for least significant set bit in a byte.
    //static vector<uint8_t, 256> lt_lo;

    //! Lookup table for select on bytes.
    /*! Entry at idx = 256*j + i equals the position of the
    (j+1)-th set bit in byte i. Positions lie in the range \f$[0..7]\f$.
    */
    //vector<uint8_t, 256*8> lt_sel(_lt_sel);

    //! Use to help to decide if a prefix sum stored in a byte overflows.
    //vector<uint64_t, 65> ps_overflow(_ps_overflow);

    //! Counts the number of set bits in x.
    /*! \param  x 64-bit word
    \return Number of set bits.
    */
    static uint32_t cnt(vector<uint32_t, SIMD_SIZE> x);

    //! Position of the most significant set bit the 64-bit word x
    /*! \param x 64-bit word
    \return The position (in 0..63) of the most significant set bit
    in `x` or 0 if x equals 0.
    \sa sel, lo
    */
    static uint32_t hi(uint32_t x);
    static uint32_t hi(vector<uint32_t, SIMD_SIZE> x);

    //! Calculates the position of the rightmost 1-bit in the 64bit integer x if it exists
    /*! \param x 64 bit integer.
    \return The position (in 0..63) of the rightmost 1-bit in the 64bit integer x if
    x>0 and 0 if x equals 0.
    \sa sel, hi
    */
    static uint32_t lo(uint32_t x);

    //! Counts the number of 1-bits in the 32bit integer x.
    /*! This function is a variant of the method cnt. If
    32bit multiplication is fast, this method beats the cnt.
    for 32bit integers.
    \param x 64bit integer to count the bits.
    \return The number of 1-bits in x.
    */
    static uint32_t cnt32(uint32_t x);

    //! Count the number of consecutive and distinct 11 in the 64bit integer x.
    /*!
    \param x 64bit integer to count the terminating sequence 11 of a Fibonacci code.
    \param c Carry equals msb of the previous 64bit integer.
    */
    static uint32_t cnt11(uint64_t x, uint64_t& c);

    //! Count the number of consecutive and distinct 11 in the 64bit integer x.
    /*!
    \param x 64bit integer to count the terminating sequence 11 of a Fibonacci code.
    */
    static uint32_t cnt11(uint64_t x);

    //! Count 10 bit pairs in the word x.
    /*!
    * \param x 64bit integer to count the 10 bit pairs.
    * \param c Carry equals msb of the previous 64bit integer.
    */
    static uint32_t cnt10(uint64_t x, uint64_t& c);

    //! Count 01 bit pairs in the word x.
    /*!
    * \param x 64bit integer to count the 01 bit pairs.
    * \param c Carry equals msb of the previous 64bit integer.
    */
    static uint32_t cnt01(uint64_t x, uint64_t& c);

    //! Map all 10 bit pairs to 01 or 1 if c=1 and the lsb=0. All other pairs are mapped to 00.
    static uint64_t map10(uint64_t x, uint64_t c = 0);

    //! Map all 01 bit pairs to 01 or 1 if c=1 and the lsb=0. All other pairs are mapped to 00.
    static uint64_t map01(uint64_t x, uint64_t c = 1);

    //! Calculate the position of the i-th rightmost 1 bit in the 64bit integer x
    /*!
    \param x 64bit integer.
    \param i Argument i must be in the range \f$[1..cnt(x)]\f$.
    \pre Argument i must be in the range \f$[1..cnt(x)]\f$.
    \sa hi, lo
    */
    static uint32_t sel(uint64_t x, uint32_t i);
    static uint32_t _sel(uint64_t x, uint32_t i);

    //! Calculates the position of the i-th rightmost 11-bit-pattern which terminates a Fibonacci coded integer in x.
    /*!	\param x 64 bit integer.
    \param i Index of 11-bit-pattern. \f$i \in [1..cnt11(x)]\f$
    \param c Carry bit from word before
    \return The position (in 1..63) of the i-th 11-bit-pattern which terminates a Fibonacci coded integer in x if
    x contains at least i 11-bit-patterns and a undefined value otherwise.
    \sa cnt11, hi11, sel

    */
    static uint32_t sel11(uint64_t x, uint32_t i, uint32_t c = 0);

    //! Calculates the position of the leftmost 11-bit-pattern which terminates a Fibonacci coded integer in x.
    /*! \param x 64 bit integer.
    \return The position (in 1..63) of the leftmost 1 of the leftmost 11-bit-pattern which
    terminates a Fibonacci coded integer in x if x contains a 11-bit-pattern
    and 0 otherwise.
    \sa cnt11, sel11
    */
    static uint32_t hi11(uint64_t x);

    //! Writes value x to an bit position in an array.
    static void write_int(uint64_t* word, uint64_t x, const uint8_t offset = 0, const uint8_t len = 64);

    //! Writes value x to an bit position in an array and moves the bit-pointer.
    static void write_int_and_move(uint64_t*& word, uint64_t x, uint8_t& offset, const uint8_t len);

    //! Reads a value from a bit position in an array.
    static uint64_t read_int(const uint64_t* word, uint8_t offset = 0, const uint8_t len = 64);

    //! Reads a value from a bit position in an array and moved the bit-pointer.
    static uint64_t read_int_and_move(const uint64_t*& word, uint8_t& offset, const uint8_t len = 64);

    //! Reads an unary decoded value from a bit position in an array.
    static uint64_t read_unary(const uint64_t* word, uint8_t offset = 0);

    //! Reads an unary decoded value from a bit position in an array and moves the bit-pointer.
    static uint64_t read_unary_and_move(const uint64_t*& word, uint8_t& offset);

    //! Move the bit-pointer (=uint64_t word and offset) `len` to the right.
    /*!\param word   64-bit word part of the bit pointer
    * \param offset Offset part of the bit pointer
    * \param len    Move distance. \f$ len \in [0..64] \f$
    * \sa move_left
    */
    static void move_right(const uint64_t*& word, uint8_t& offset, const uint8_t len);

    //! Move the bit-pointer (=uint64_t word and offset) `len` to the left.
    /*!\param word   64-bit word part of the bit pointer
    * \param offset Offset part of the bit pointer
    * \param len    Move distance. \f$ len \in [0..64] \f$
    * \sa move_right
    */
    static void move_left(const uint64_t*& word, uint8_t& offset, const uint8_t len);

    //! Get the first one bit in the interval \f$[idx..\infty )\f$
    static uint64_t next(const uint64_t* word, uint64_t idx);

    //! Get the one bit with the greatest position in the interval \f$[0..idx]\f$
    static uint64_t prev(const uint64_t* word, uint64_t idx);

    //! reverses a given 64 bit word
    static uint32_t rev(uint32_t x);
    static vector<uint32_t, SIMD_SIZE> rev(vector<uint32_t, SIMD_SIZE> x);
  //};


  // ============= inline - implementations ================

  // see page 11, Knuth TAOCP Vol 4 F1A

  inline uint32_t cnt(vector<uint32_t, SIMD_SIZE> x)
  {
    return cm_sum<uint32_t>(cm_cbit(x));
  }


  inline uint32_t cnt32(uint32_t x)
  {
    return cm_cbit(x);
  }


  inline uint32_t cnt11(uint64_t x, uint64_t& c)
  {
    // extract "11" 2bit blocks
    uint64_t ex11 = (x&(x >> 1)) & 0x5555555555555555ULL, t;
    // extract "10" 2bit blocks
    uint64_t ex10or01 = (ex11 | (ex11 << 1)) ^ x;

    x = ex11 | ((t = (ex11 | (ex11 << 1)) + (((ex10or01 << 1) & 0x5555555555555555ULL) | c))&(ex10or01 & 0x5555555555555555ULL));
    c = (ex10or01 >> 63) | (t < (ex11 | (ex11 << 1)));

    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (0x0101010101010101ULL * x >> 56);
  }


  inline uint32_t cnt11(uint64_t x)
  {
    // extract "11" 2bit blocks
    uint64_t ex11 = (x&(x >> 1)) & 0x5555555555555555ULL;
    // extract "10" 2bit blocks
    uint64_t ex10or01 = (ex11 | (ex11 << 1)) ^ x;

    x = ex11 | (((ex11 | (ex11 << 1)) + ((ex10or01 << 1) & 0x5555555555555555ULL))&(ex10or01 & 0x5555555555555555ULL));

    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (0x0101010101010101ULL * x >> 56);
  }

  inline uint32_t cnt10(uint64_t x, uint64_t& c)
  {
    uint32_t res = cnt((x ^ ((x << 1) | c)) & (~x));
    c = (x >> 63);
    return res;
  }

  inline uint64_t map10(uint64_t x, uint64_t c)
  {
    return ((x ^ ((x << 1) | c)) & (~x));
  }

  inline uint32_t cnt01(uint64_t x, uint64_t& c)
  {
    uint32_t res = cnt((x ^ ((x << 1) | c)) & x);
    c = (x >> 63);
    return res;
  }
  inline uint64_t map01(uint64_t x, uint64_t c)
  {
    return ((x ^ ((x << 1) | c)) &  x);
  }

  inline uint32_t sel(uint64_t x, uint32_t i)
  {
#ifdef __BMI2__
    // index i is 1-based here, (i-1) changes it to 0-based
    return __builtin_ctzll(_pdep_u64(1ull << (i - 1), x));
#elif defined(__SSE4_2__)
    uint64_t s = x, b;
    s = s - ((s >> 1) & 0x5555555555555555ULL);
    s = (s & 0x3333333333333333ULL) + ((s >> 2) & 0x3333333333333333ULL);
    s = (s + (s >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    s = 0x0101010101010101ULL * s;
    // now s contains 8 bytes s[7],...,s[0]; s[j] contains the cumulative sum
    // of (j+1)*8 least significant bits of s
    b = (s + ps_overflow[i]) & 0x8080808080808080ULL;
    // ps_overflow contains a bit mask x consisting of 8 bytes
    // x[7],...,x[0] and x[j] is set to 128-j
    // => a byte b[j] in b is >= 128 if cum sum >= j

    // __builtin_ctzll returns the number of trailing zeros, if b!=0
    int  byte_nr = __builtin_ctzll(b) >> 3;   // byte nr in [0..7]
    s <<= 8;
    i -= (s >> (byte_nr << 3)) & 0xFFULL;
    return (byte_nr << 3) + lt_sel[((i - 1) << 8) + ((x >> (byte_nr << 3)) & 0xFFULL)];
#else
    return _sel(x, i);
#endif
  }

  inline uint32_t _sel(uint64_t x, uint32_t i)
  {
#if 0
    uint64_t s = x, b;  // s = sum
    s = s - ((s >> 1) & 0x5555555555555555ULL);
    s = (s & 0x3333333333333333ULL) + ((s >> 2) & 0x3333333333333333ULL);
    s = (s + (s >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    s = 0x0101010101010101ULL * s;
    b = (s + ps_overflow[i]);//&0x8080808080808080ULL;// add something to the partial sums to cause overflow
    i = (i - 1) << 8;
    if (b & 0x0000000080000000ULL) // byte <=3
      if (b & 0x0000000000008000ULL) //byte <= 1
        if (b & 0x0000000000000080ULL)
          return    lt_sel[(x & 0xFFULL) + i];
        else
          return 8 + lt_sel[(((x >> 8) & 0xFFULL) + i - ((s & 0xFFULL) << 8)) & 0x7FFULL];//byte 1;
      else//byte >1
        if (b & 0x0000000000800000ULL) //byte <=2
          return 16 + lt_sel[(((x >> 16) & 0xFFULL) + i - (s & 0xFF00ULL)) & 0x7FFULL];//byte 2;
        else
          return 24 + lt_sel[(((x >> 24) & 0xFFULL) + i - ((s >> 8) & 0xFF00ULL)) & 0x7FFULL];//byte 3;
    else//  byte > 3
      if (b & 0x0000800000000000ULL) // byte <=5
        if (b & 0x0000008000000000ULL) //byte <=4
          return 32 + lt_sel[(((x >> 32) & 0xFFULL) + i - ((s >> 16) & 0xFF00ULL)) & 0x7FFULL];//byte 4;
        else
          return 40 + lt_sel[(((x >> 40) & 0xFFULL) + i - ((s >> 24) & 0xFF00ULL)) & 0x7FFULL];//byte 5;
      else// byte >5
        if (b & 0x0080000000000000ULL) //byte<=6
          return 48 + lt_sel[(((x >> 48) & 0xFFULL) + i - ((s >> 32) & 0xFF00ULL)) & 0x7FFULL];//byte 6;
        else
          return 56 + lt_sel[(((x >> 56) & 0xFFULL) + i - ((s >> 40) & 0xFF00ULL)) & 0x7FFULL];//byte 7;
    return 0;
#endif 
    uint32_t pos = 0;
    while (x) {
      i -= x & 1;
      x >>= 1;
      if (!i) break;
      ++pos;
    }
    return pos;
  }

  // using built-in method or
  // 64-bit version of 32-bit proposal of
  // http://www-graphics.stanford.edu/~seander/bithacks.html
  inline uint32_t hi(uint32_t x)
  {
    if (x == 0)
      return 0;
    return 31 - cm_fbh(x);

  }
  inline uint32_t hi(vector<uint32_t, SIMD_SIZE> x)
  {
    if ((x == 0).all())
      return 0;
    vector<ushort, SIMD_SIZE> vex = cm_fbh(x);
    vector<ushort, SIMD_SIZE> result = (vex != 0xFFFF);
    uint32_t pack = cm_pack_mask(result);
    return 32 * (31 -cm_fbh(pack)) + (31 - vex[31 - cm_fbh(pack)]);

  }

  // details see: http://citeseer.ist.psu.edu/leiserson98using.html
  // or page 10, Knuth TAOCP Vol 4 F1A
  inline _GENX_ uint32_t lo(uint32_t x)
  {
    if (x == 0)
      return 0;
    return cm_fbl(x);;
  }

  inline _GENX_ uint32_t lo(vector<uint32_t, SIMD_SIZE> x)
  {

    if ((x == 0).all())
      return 0;
    vector<ushort, SIMD_SIZE> vex = cm_fbl(x);
    vector<ushort, SIMD_SIZE> result = (vex != 0xFFFF);
    uint32_t pack = cm_pack_mask(result);
    return 32 * cm_fbl(pack) + vex[cm_fbl(pack)];

  }


  inline uint32_t hi11(uint64_t x)
  {
    // extract "11" 2bit blocks
    uint64_t ex11 = (x&(x >> 1)) & 0x5555555555555555ULL;
    // extract "10" 2bit blocks
    uint64_t ex10or01 = (ex11 | (ex11 << 1)) ^ x;
    // extract "10" 2bit blocks
    ex11 += (((ex11 | (ex11 << 1)) + ((ex10or01 << 1) & 0x5555555555555555ULL)) & ((ex10or01 & 0x5555555555555555ULL) | ex11));
    return hi(ex11);
  }


  inline uint32_t sel11(uint64_t x, uint32_t i, uint32_t c)
  {
    uint64_t ex11 = (x&(x >> 1)) & 0x5555555555555555ULL;
    uint64_t ex10or01 = (ex11 | (ex11 << 1)) ^ x;
    ex11 += (((ex11 | (ex11 << 1)) + (((ex10or01 << 1) & 0x5555555555555555ULL) | c)) & ((ex10or01 & 0x5555555555555555ULL) | ex11));
    return sel(ex11, i);
  }
#if 0
  inline void bits::write_int(uint64_t* word, uint64_t x, uint8_t offset, const uint8_t len)
  {
    x &= bits::lo_set[len];
    if (offset + len < 64) {
      *word &=
        ((bits::all_set << (offset + len)) | bits::lo_set[offset]); // mask 1..10..01..1
      *word |= (x << offset);
      //		*word ^= ((*word ^ x) & (bits::lo_set[len] << offset) );
      //      surprisingly the above line is slower than the lines above
    }
    else {
      *word &=
        ((bits::lo_set[offset]));  // mask 0....01..1
      *word |= (x << offset);
      if ((offset = (offset + len) & 0x3F)) { // offset+len > 64
        *(word + 1) &= (~bits::lo_set[offset]); // mask 1...10..0
                                                //			*(word+1) &= bits::lo_unset[offset]; // mask 1...10..0
                                                //          surprisingly the above line is slower than the line above
        *(word + 1) |= (x >> (len - offset));
      }
    }
  }

  inline void bits::write_int_and_move(uint64_t*& word, uint64_t x, uint8_t& offset, const uint8_t len)
  {
    x &= bits::lo_set[len];
    if (offset + len < 64) {
      *word &=
        ((bits::all_set << (offset + len)) | bits::lo_set[offset]); // mask 1..10..01..1
      *word |= (x << offset);
      offset += len;
    }
    else {
      *word &=
        ((bits::lo_set[offset]));  // mask 0....01..1
      *word |= (x << offset);
      if ((offset = (offset + len))>64) {// offset+len >= 64
        offset &= 0x3F;
        *(++word) &= (~bits::lo_set[offset]); // mask 1...10..0
        *word |= (x >> (len - offset));
      }
      else {
        offset = 0;
        ++word;
      }
    }
  }

  inline uint64_t bits::read_int(const uint64_t* word, uint8_t offset, const uint8_t len)
  {
    uint64_t w1 = (*word) >> offset;
    if ((offset + len) > 64) { // if offset+len > 64
      return w1 |  // w1 or w2 adepted:
        ((*(word + 1) & bits::lo_set[(offset + len) & 0x3F])   // set higher bits zero
          << (64 - offset));  // move bits to the left
    }
    else {
      return w1 & bits::lo_set[len];
    }
  }

  inline uint64_t bits::read_int_and_move(const uint64_t*& word, uint8_t& offset, const uint8_t len)
  {
    uint64_t w1 = (*word) >> offset;
    if ((offset = (offset + len)) >= 64) {  // if offset+len > 64
      if (offset == 64) {
        offset &= 0x3F;
        ++word;
        return w1;
      }
      else {
        offset &= 0x3F;
        return w1 |
          (((*(++word)) & bits::lo_set[offset]) << (len - offset));
      }
    }
    else {
      return w1 & bits::lo_set[len];
    }
  }

  inline uint64_t bits::read_unary(const uint64_t* word, uint8_t offset)
  {
    uint64_t w = *word >> offset;
    if (w) {
      return bits::lo(w);
    }
    else {
      if (0 != (w = *(++word)))
        return bits::lo(w) + 64 - offset;
      uint64_t cnt = 2;
      while (0 == (w = *(++word)))
        ++cnt;
      return bits::lo(w) + (cnt << 6) - offset;
    }
    return 0;
  }

  inline uint64_t bits::read_unary_and_move(const uint64_t*& word, uint8_t& offset)
  {
    uint64_t w = (*word) >> offset; // temporary variable is good for the performance
    if (w) {
      uint8_t r = bits::lo(w);
      offset = (offset + r + 1) & 0x3F;
      // we know that offset + r +1 <= 64, so if the new offset equals 0 increase word
      word += (offset == 0);
      return r;
    }
    else {
      uint8_t rr = 0;
      if (0 != (w = *(++word))) {
        rr = bits::lo(w) + 64 - offset;
        offset = (offset + rr + 1) & 0x3F;
        word += (offset == 0);
        return rr;
      }
      else {
        uint64_t cnt_1 = 1;
        while (0 == (w = *(++word)))
          ++cnt_1;
        rr = bits::lo(w) + 64 - offset;
        offset = (offset + rr + 1) & 0x3F;
        word += (offset == 0);
        return ((cnt_1) << 6) + rr;
      }
    }
    return 0;
  }

  inline void bits::move_right(const uint64_t*& word, uint8_t& offset, const uint8_t len)
  {
    if ((offset += len) & 0xC0) { // if offset >= 65
      offset &= 0x3F;
      ++word;
    }
  }

  inline void bits::move_left(const uint64_t*& word, uint8_t& offset, const uint8_t len)
  {
    if ((offset -= len) & 0xC0) {  // if offset-len<0
      offset &= 0x3F;
      --word;
    }
  }

  inline uint64_t bits::next(const uint64_t* word, uint64_t idx)
  {
    word += (idx >> 6);
    if (*word & ~lo_set[idx & 0x3F]) {
      return (idx & ~((size_t)0x3F)) + lo(*word & ~lo_set[idx & 0x3F]);
    }
    idx = (idx & ~((size_t)0x3F)) + 64;
    ++word;
    while (*word == 0) {
      idx += 64;
      ++word;
    }
    return idx + lo(*word);
  }

  inline uint64_t bits::prev(const uint64_t* word, uint64_t idx)
  {
    word += (idx >> 6);
    if (*word & lo_set[(idx & 0x3F) + 1]) {
      return (idx & ~((size_t)0x3F)) + hi(*word & lo_set[(idx & 0x3F) + 1]);
    }
    idx = (idx & ~((size_t)0x3F)) - 64;
    --word;
    while (*word == 0) {
      idx -= 64;
      --word;
    }
    return idx + hi(*word);
  }
#endif
  inline uint32_t rev(uint32_t x)
  {
	  vector<uint32_t, 1> xx = x;
    return cm_bf_reverse<uint32_t>(xx)[0];
  }

  inline vector<uint32_t, SIMD_SIZE> rev(vector<uint32_t, SIMD_SIZE> x)
  {
    return cm_bf_reverse<uint32_t>(x);
  }
