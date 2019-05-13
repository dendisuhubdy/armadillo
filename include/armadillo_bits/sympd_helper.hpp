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


//! \addtogroup sympd_helper
//! @{


namespace sympd_helper
{

// computationally inexpensive algorithm to guess whether a matrix is positive definite:
// (1) ensure the matrix is symmetric/hermitian (within a tolerance)
// (2) ensure the diagonal entries are greater than zero
// (3) ensure that the value with largest modulus is on the main diagonal
// (4) ensure rudimentary diagonal dominance: (real(A_ii) + real(A_jj)) > 2*abs(real(A_ij))
// the above conditions are necessary, but not sufficient;
// doing it properly would be too computationally expensive for our purposes
// more info:
// http://mathworld.wolfram.com/PositiveDefiniteMatrix.html
// http://mathworld.wolfram.com/DiagonallyDominantMatrix.html
  
template<typename eT>
inline
typename enable_if2<is_cx<eT>::no, bool>::result
guess_sympd(const Mat<eT>& A)
  {
  arma_extra_debug_sigprint();
  
  if((A.n_rows != A.n_cols) || (A.n_rows < 16))  { return false; }
  
  const eT tol = eT(100) * std::numeric_limits<eT>::epsilon();  // allow some leeway
  
  const uword N = A.n_rows;
  
  const eT* A_mem = A.memptr();
  const eT* A_col = A_mem;
  
  eT max_diag = eT(0);
  
  for(uword j=0; j < N; ++j)
    {
    const eT A_jj = A_col[j];
    
    if(A_jj <= eT(0))  { return false; }
    
    max_diag = (A_jj > max_diag) ? A_jj : max_diag;
    
    A_col += N;
    }
  
  A_col = A_mem;
  
  const uword Nm1 = N-1;
  
  for(uword j=0; j < Nm1; ++j)
    {
    const uword jp1   = j+1;
    const eT*   A_row = &(A_mem[j + jp1*N]);  // &(A.at(j,jp1));
    
    const eT A_jj = A_col[j];
    
    for(uword i=jp1; i < N; ++i)
      {
      const eT A_ij = A_col[i];
      const eT A_ji = (*A_row);
      
      const eT A_ij_abs = (std::abs)(A_ij);
      const eT A_ji_abs = (std::abs)(A_ji);
      
      if( (A_ij_abs >= max_diag) || (A_ji_abs >= max_diag) )  { return false; }
      
      const eT A_delta   = (std::abs)(A_ij - A_ji);
      const eT A_abs_max = (std::max)(A_ij_abs, A_ji_abs);
      
      if( (A_delta > tol) && (A_delta > (A_abs_max*tol)) )  { return false; }
      
      const eT A_ii = A_mem[i + i*N];
      
      if( (eT(2)*A_ij_abs) >= (A_ii + A_jj) )  { return false; }
      
      A_row += N;
      }
    
    A_col += N;
    }
  
  return true;
  }



template<typename eT>
inline
typename enable_if2<is_cx<eT>::yes, bool>::result
guess_sympd(const Mat<eT>& A)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  if((A.n_rows != A.n_cols) || (A.n_rows < 16))  { return false; }
  
  const T tol = T(100) * std::numeric_limits<T>::epsilon();  // allow some leeway
  
  const uword N = A.n_rows;
  
  const eT* A_mem = A.memptr();
  const eT* A_col = A_mem;
  
  T max_diag = T(0);
  
  for(uword j=0; j < N; ++j)
    {
    const eT& A_jj      = A_col[j];
    const  T  A_jj_real = std::real(A_jj);
    const  T  A_jj_imag = std::imag(A_jj);
    
    if( (A_jj_real <= T(0)) || (std::abs(A_jj_imag) > tol) )  { return false; }
    
    max_diag = (A_jj_real > max_diag) ? A_jj_real : max_diag;
    
    A_col += N;
    }
  
  A_col = A_mem;
  
  const uword Nm1 = N-1;
  
  for(uword j=0; j < Nm1; ++j)
    {
    const uword jp1   = j+1;
    const eT*   A_row = &(A_mem[j + jp1*N]);  // &(A.at(j,jp1));
    
    const T A_jj_real = std::real(A_col[j]);
    
    for(uword i=jp1; i < N; ++i)
      {
      const eT& A_ij      = A_col[i];
      const  T  A_ij_real = std::real(A_ij);
      const  T  A_ij_imag = std::imag(A_ij);
      const  T  A_ij_abs  =  std::abs(A_ij);
      
      if(A_ij_abs >= max_diag )  { return false; }
      
      const eT& A_ji      = (*A_row);
      const  T  A_ji_real = std::real(A_ji);
      const  T  A_ji_imag = std::imag(A_ji);
      //const  T  A_ji_abs  =  std::abs(A_ji);
      
      //if(A_ji_abs >= max_diag )  { return false; }
      
      
      const T A_ij_real_abs = (std::abs)(A_ij_real);
      const T A_ji_real_abs = (std::abs)(A_ji_real);
      
      const T A_real_delta   = (std::abs)(A_ij_real - A_ji_real);
      const T A_real_abs_max = (std::max)(A_ij_real_abs, A_ji_real_abs);
      
      if( (A_real_delta > tol) && (A_real_delta > (A_real_abs_max*tol)) )  { return false; }
      
      
      const T A_ij_imag_abs = (std::abs)(A_ij_imag);
      const T A_ji_imag_abs = (std::abs)(A_ji_imag);
      
      const T A_imag_delta   = (std::abs)(A_ij_imag + A_ji_imag);  // take into account complex conjugate
      const T A_imag_abs_max = (std::max)(A_ij_imag_abs, A_ji_imag_abs);
      
      if( (A_imag_delta > tol) && (A_imag_delta > (A_imag_abs_max*tol)) )  { return false; }
      
      
      const T A_ii_real = std::real(A_mem[i + i*N]);
      
      if( (T(2)*A_ij_real_abs) >= (A_ii_real + A_jj_real) )  { return false; }
      
      A_row += N;
      }
    
    A_col += N;
    }
  
  return true;
  }



}  // end of namespace sympd_helper


//! @}
