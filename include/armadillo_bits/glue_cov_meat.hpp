// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup glue_cov
//! @{



template<typename T1, typename T2>
inline
void
glue_cov::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_cov>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword norm_type = X.aux_uword;
  
  const unwrap<T1> UA(X.A);
  const unwrap<T2> UB(X.B);
  
  const Mat<eT>& A = UA.M;
  const Mat<eT>& B = UB.M;
  
  const Mat<eT> AA( const_cast<eT*>(A.memptr()), A.n_cols, A.n_rows, false, true);
  const Mat<eT> BB( const_cast<eT*>(B.memptr()), B.n_cols, B.n_rows, false, true);
  
  const Mat<eT>& AAA = (A.n_rows == 1) ? AA : A;
  const Mat<eT>& BBB = (B.n_rows == 1) ? BB : B;
  
  arma_debug_assert_mul_size(AAA, BBB, true, false, "cov()");
  
  const uword N        = AAA.n_rows;
  const eT    norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
  
  const Mat<eT> tmp1 = AAA.each_row() - mean(AAA,0);
  const Mat<eT> tmp2 = BBB.each_row() - mean(BBB,0);
  
  out = tmp1.t() * tmp2;
  out /= norm_val;
  }



//! @}
