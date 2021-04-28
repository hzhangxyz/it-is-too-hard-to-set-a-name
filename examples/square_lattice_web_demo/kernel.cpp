#include <TAT/TAT.hpp>
#include <memory>
#include <random>

using DTensor = TAT::Tensor<double>;
using ZTensor = TAT::Tensor<std::complex<double>>;

auto Sz = ZTensor({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {1, 0, 0, -1};
   return data[i++] / 2.;
});
auto Sx = ZTensor({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {0, 1, 1, 0};
   return data[i++] / 2.;
});
auto Sy = ZTensor({"I", "O"}, {2, 2}).set([]() {
   static int i = 0;
   static std::complex<double> data[4] = {0, {0, 1}, {0, -1}, 0};
   return data[i++] / 2.;
});
auto SzSz = Sz.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sz.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SxSx = Sx.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sx.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SySy = Sy.edge_rename({{"I", "I1"}, {"O", "O1"}})
                  .contract_all_edge(Sy.edge_rename({{"I", "I2"}, {"O", "O2"}}))
                  .transpose({"I1", "I2", "O1", "O2"})
                  .to<double>();
auto SS = SxSx + SySy + SzSz;

auto random_engine = std::default_random_engine(std::random_device()());

struct SpinLattice {
   DTensor state_vector;
   std::vector<DTensor> bonds;
   double energy;
   double approximate_energy;

   SpinLattice(const std::vector<std::string>& node_names, double approximate_energy = 0) : approximate_energy(std::abs(approximate_energy)) {
      auto edge_to_initial = std::vector<int>(node_names.size(), 2);
      auto dist = std::normal_distribution<double>(0, 1);
      state_vector = DTensor({node_names.begin(), node_names.end()}, {edge_to_initial.begin(), edge_to_initial.end()}).set([&]() {
         return dist(random_engine);
      });
   }

   void set_bond(const std::string& n1, const std::string& n2, const DTensor& matrix) {
      bonds.push_back(matrix.edge_rename({{"I1", n1}, {"I2", n2}, {"O1", "_" + n1}, {"O2", "_" + n2}}));
   }

   void update() {
      auto norm_max = double(state_vector.norm<-1>());
      energy = approximate_energy - norm_max;
      state_vector /= norm_max;
      auto state_vector_temporary = state_vector.same_shape().zero();
      for (const auto& bond : bonds) {
         const auto& name = bond.names;
         auto this_term = state_vector.contract_all_edge(bond).edge_rename({{name[2], name[0]}, {name[3], name[1]}});
         state_vector_temporary += this_term;
      }
      state_vector *= approximate_energy;
      state_vector -= state_vector_temporary;
   }

   template<typename Scalar>
   auto observe(const TAT::Tensor<Scalar>& op) const {
      std::map<TAT::DefaultName, TAT::DefaultName> map;
      for (const auto& n : op.names) {
         auto str = std::string(n);
         map["_" + str] = str;
      }
      if constexpr (TAT::is_complex_v<Scalar>) {
         auto v = state_vector.to<Scalar>();
         Scalar value = v.contract_all_edge(op).edge_rename(map).contract_all_edge(v);
         return value.real();
      } else {
         const auto& v = state_vector;
         Scalar value = v.contract_all_edge(op).edge_rename(map).contract_all_edge(v);
         return value;
      }
   }

   template<typename Scalar>
   auto observe_single_site(const std::string& n, const TAT::Tensor<Scalar>& matrix) const {
      auto op = matrix.edge_rename({{"I", n}, {"O", "_" + n}});
      return observe(op);
   }

   double get_observe_denominator() const {
      return state_vector.contract_all_edge(state_vector);
   }
};

struct SquareSpinLattice : SpinLattice {
   int n1;
   int n2;

   static auto get_node_names(int n1, int n2) {
      auto result = std::vector<std::string>();
      for (auto i = 0; i < n1; i++) {
         for (auto j = 0; j < n2; j++) {
            result.push_back(std::to_string(i) + "." + std::to_string(j));
         }
      }
      return result;
   }

   SquareSpinLattice(int n1, int n2, double approximate_energy = 0) : SpinLattice(get_node_names(n1, n2), approximate_energy), n1(n1), n2(n2) {}

   void set_bond(const std::tuple<int, int>& p1, const std::tuple<int, int>& p2, const DTensor& matrix) {
      std::string n1 = std::to_string(std::get<0>(p1)) + "." + std::to_string(std::get<1>(p1));
      std::string n2 = std::to_string(std::get<0>(p2)) + "." + std::to_string(std::get<1>(p2));
      SpinLattice::set_bond(n1, n2, matrix);
   }

   template<typename Scalar>
   auto observe_single_site(const std::tuple<int, int>& p, const TAT::Tensor<Scalar>& matrix) const {
      std::string n = std::to_string(std::get<0>(p)) + "." + std::to_string(std::get<1>(p));
      return SpinLattice::observe_single_site(n, matrix);
   }
};

std::unique_ptr<SquareSpinLattice> lattice;

extern "C" {

   int create_lattice(int n1, int n2) {
      lattice = std::make_unique<SquareSpinLattice>(n1, n2, n1 * n2 * 0.5);
      for (auto i = 0; i < n1 - 1; i++) {
         for (auto j = 0; j < n2; j++) {
            lattice->set_bond({i, j}, {i + 1, j}, SS);
         }
      }
      for (auto i = 0; i < n1; i++) {
         for (auto j = 0; j < n2 - 1; j++) {
            lattice->set_bond({i, j}, {i, j + 1}, SS);
         }
      }
      return 0;
   }

   int update_lattice(int step) {
      for (auto t = 0; t < step; t++) {
         lattice->update();
      }
      return 0;
   }

   double get_energy() {
      return lattice->energy / (lattice->n1 * lattice->n2);
   }

   double get_spin(int x, int y, int kind) {
      if (kind == 0) {
         return lattice->observe_single_site({x, y}, Sx);
      } else if (kind == 1) {
         return lattice->observe_single_site({x, y}, Sy);
      } else {
         return lattice->observe_single_site({x, y}, Sz);
      }
   }
   double get_den() {
      return lattice->get_observe_denominator();
   }
}
