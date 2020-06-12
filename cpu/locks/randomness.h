#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Well Equidistributed Long-period Linear (WELL) Random Number Generator
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define R   32

#define M1   3
#define M2  24
#define M3  10

#define MAT0POS(t,v)  (v ^ (v >>  (t)))
#define MAT0NEG(t,v)  (v ^ (v << -(t)))
#define Identity(v)   (v)

#define V0            State[ w        ]
#define VM1           State[(w+M1) % R]
#define VM2           State[(w+M2) % R]
#define VM3           State[(w+M3) % R]
#define VRm1          State[(w+31) % R]

#define newV0         State[(w+31) % R]
#define newV1         State[ w        ]


class RNG
{
   private:

      uint32_t State[R];
      uint32_t w = 0;

   public:

      RNG(uint32_t tid = 0 , uint32_t seed = 0)
      {
         uint32_t x = seed;
         uint32_t y = 362436000;
         uint32_t z = tid;
         uint32_t c = 7654321;
         uint64_t t;

         for (int i = 0 ; i < R ; ++i)
         {
            x = 69069 * x + 12345;

            y ^= y << 13;
            y ^= y >> 17;
            y ^= y << 5;

            t = (uint64_t) 698769069 * z + c;
            c = t >> 32;
            z = (uint32_t) t;

            State[i] = x + y + z;
         }
      }

      uint32_t random()
      {
         uint32_t        z0;
         uint32_t        z1;
         uint32_t        z2;

         z0    = VRm1;
         z1    = Identity( V0)      ^ MAT0POS ( +8, VM1);
         z2    = MAT0NEG (-19, VM2) ^ MAT0NEG (-14, VM3);
         newV1 = z1                 ^ z2;
         newV0 = MAT0NEG (-11,  z0) ^ MAT0NEG ( -7,  z1) ^ MAT0NEG (-13,  z2);
         w     = (w + R - 1) % R;

         return State[w];
      }
};

