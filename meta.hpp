#ifndef META_HPP_
#define META_HPP_

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <future>
#include <memory>
#include <cstdlib>
#include <cstring>

#ifdef __NVCC__
#define USE_CUDA
#else
#define USE_CPU
#endif

#define PASS std::cerr << "calling a passing function at " << __FILE__ << ":" << __LINE__ << " in " << __PRETTY_FUNCTION__ <<std::endl;

namespace Node
{
#ifndef Type
#define Type double
#endif
  using Base = Type;
#undef Type
  enum class Leg
    {
#define CreateLeg(x) Left##x, Right##x, Up##x, Down##x, Phy##x
     CreateLeg(), CreateLeg(1), CreateLeg(2), CreateLeg(3), CreateLeg(4)
#undef CreateLeg
    };

  using Rank  = unsigned int;
  using Size  = std::size_t;
  using Dims  = std::vector<Size>;
  using Legs  = std::vector<Leg>;
  using Order = std::vector<Rank>;
  using PlainData  = Base*;

  namespace internal
  {
    namespace leg
    {
#define IncEnum(p) {Leg::p, #p}
#define IncGroup(x) IncEnum(Left##x), IncEnum(Right##x), IncEnum(Up##x), IncEnum(Down##x), IncEnum(Phy##x)
      static const std::map<Leg, std::string> leg_str = {IncGroup(), IncGroup(1), IncGroup(2), IncGroup(3), IncGroup(4)};
#undef IncGroup
#undef IncEnum
    }
  }

  // 不知道为什么这里用引用会错误, 好像只是gdb的事
  inline std::ostream& operator<<(std::ostream& out, const Leg& value)
  {
    try
      {
        return out << internal::leg::leg_str.at(value);
      }
    catch(const std::out_of_range& e)
      {
        return out;
      }
  }

  namespace internal
  {
    namespace memory
    {
      class deleter
      {
      public:
        inline void operator()(Base*) const;
      };

      std::unique_ptr<Base[], deleter> newer(Size);// size是元素个数,需要乘上sizeof(Base)才是需要malloc的大小

      void memCopy(void*, const void*, Size);

      void memSend(void*, const void*, Size);

      void memRecv(void*, const void*, Size);
    }

    namespace shuffle
    {
      inline void make_plan(Order& plan, const Legs& new_legs, const Legs& legs)
      {
        const Rank& rank = legs.size();
        for(Rank i=0;i<rank;i++)
          {
            for(Rank j=0;j<rank;j++)
              {
                if(new_legs[i]==legs[j])
                  {
                    plan.push_back(j);
                    break;
                  }
              }
          }
      }

      inline void get_dims(Dims& new_dims, const Dims& dims, const Order& plan)
      {
        const Rank& rank = dims.size();
        for(Rank i=0;i<rank;i++)
          {
            new_dims.push_back(dims[plan[i]]);
          }
      }

      void shuffle(PlainData    data_new,
                   PlainData    data_old,
                   const Dims&  dims_new,
                   const Dims&  dims_old,
                   const Order& plan);
    }
  }

  class TensorData;

  class Tensor;
}

#define DefineLeg(x) static const Node::Leg x = Node::Leg::x
#define DefineLegs(n) DefineLeg(Left##n); DefineLeg(Right##n); DefineLeg(Up##n); DefineLeg(Down##n); DefineLeg(Phy##n)
DefineLegs(); DefineLegs(1); DefineLegs(2); DefineLegs(3); DefineLegs(4);
#undef DefineLegs
#undef DefineLeg

#endif
