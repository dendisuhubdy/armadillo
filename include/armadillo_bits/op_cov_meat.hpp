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



//! \addtogroup op_cov
//! @{



template<typename eT>
inline
void
op_cov::direct_cov(Mat<eT>& out, const Mat<eT>& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();
  
  const Mat<eT> B( const_cast<eT*>(A.memptr()), A.n_cols, A.n_rows, false, true);
  
  const Mat<eT>& C = (A.n_rows == 1) ? B : A;
  
  const uword N        = C.n_rows;
  const eT    norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
  
  const Mat<eT> tmp = C.each_row() - mean(C,0);
  
  out = tmp.t() * tmp;
  out /= norm_val;
  }



template<typename eT>
inline
void
op_cov::direct_cov_htrans(Mat<eT>& out, const Mat<eT>& A, const uword norm_type)
  {
  arma_extra_debug_sigprint();
  
  const Mat<eT> B( const_cast<eT*>(A.memptr()), A.n_cols, A.n_rows, false, true);
  
  const Mat<eT>& C = (A.n_cols == 1) ? B : A;
  
  const uword N        = C.n_cols;
  const eT    norm_val = (norm_type == 0) ? ( (N > 1) ? eT(N-1) : eT(1) ) : eT(N);
  
  const Mat<eT> tmp = C.each_col() - mean(C,1);
  
  out = tmp * tmp.t();
  out /= norm_val;
  }



template<typename T1>
inline
void
op_cov::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_cov>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword norm_type = in.aux_uword_a;
  
  const quasi_unwrap<T1> U(in.m);
  const Mat<eT>& A     = U.M;
  
  if(U.is_alias(out))
    {
    Mat<eT> tmp;
    
    op_cov::direct_cov(tmp, A, norm_type);
    
    out.steal_mem(tmp);
    }
  else
    {
    op_cov::direct_cov(out, A, norm_type);
    }
  }



template<typename T1>
inline
void
op_cov::apply(Mat<typename T1::elem_type>& out, const Op< Op<T1,op_htrans>, op_cov>& in)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword norm_type = in.aux_uword_a;
  
  if(is_cx<eT>::no)
    {
    const quasi_unwrap<T1> U(in.m.m);
    const Mat<eT>& A     = U.M;
    
    if(U.is_alias(out))
      {
      Mat<eT> tmp;
      
      op_cov::direct_cov_htrans(tmp, A, norm_type);
      
      out.steal_mem(tmp);
      }
    else
      {
      op_cov::direct_cov_htrans(out, A, norm_type);
      }
    }
  else
    {
    const Mat<eT> tmp = in.m;  // force the evaluation of Op<T1,op_htrans>
    
    op_cov::direct_cov(out, tmp, norm_type);
    }
  }



//! @}
