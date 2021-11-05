#ifndef FITNESS_CL
#define FITNESS_CL

#ifdef __cplusplus
struct MemberFitness {
#else
typedef struct {
#endif //__cplusplus

  float value;
  int index;

#ifdef __cplusplus
  bool operator<(const MemberFitness& other) const {
    return value < other.value;
  }
};
#else
} MemberFitness;
#endif //__cplusplus

#endif // FITNESS_CL
