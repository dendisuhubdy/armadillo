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


//! \addtogroup glue_solve
//! @{



//
// glue_solve_gen


template<typename T1, typename T2>
inline
void
glue_solve_gen::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_solve_gen>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = glue_solve_gen::apply( out, X.A, X.B, X.aux_uword );
  
  if(status == false)
    {
    arma_stop_runtime_error("solve(): solution not found");
    }
  }



template<typename eT, typename T1, typename T2>
inline
bool
glue_solve_gen::apply(Mat<eT>& out, const Base<eT,T1>& A_expr, const Base<eT,T2>& B_expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const bool fast        = bool(flags & solve_opts::flag_fast       );
  const bool equilibrate = bool(flags & solve_opts::flag_equilibrate);
  const bool no_approx   = bool(flags & solve_opts::flag_no_approx  );
  const bool no_band     = bool(flags & solve_opts::flag_no_band    );
  const bool no_sym      = bool(flags & solve_opts::flag_no_sym     );
  
  arma_extra_debug_print("glue_solve_gen::apply(): enabled flags:");
  
  if(fast       )  { arma_extra_debug_print("fast");        }
  if(equilibrate)  { arma_extra_debug_print("equilibrate"); }
  if(no_approx  )  { arma_extra_debug_print("no_approx");   }
  if(no_band    )  { arma_extra_debug_print("no_band");     }
  if(no_sym     )  { arma_extra_debug_print("no_sym");      }
  
  arma_debug_check( (fast && equilibrate), "solve(): options 'fast' and 'equilibrate' are mutually exclusive" );
  
  T    rcond  = T(0);
  bool status = false;
  
  Mat<eT> A = A_expr.get_ref();
  
  if(A.n_rows == A.n_cols)
    {
    arma_extra_debug_print("glue_solve_gen::apply(): detected square system");
    
    // TODO: detect symmetric matrix and use lapack::sysv() and lapack::sysvx() (real/complex matrices)
    // TODO: detect hermitian matrix and use lapack::hesv() and lapack::hesvx() (only complex matrices)
    
    uword KL = 0;
    uword KU = 0;
    
    const bool is_band = ((no_band == false) && (auxlib::crippled_lapack(A) == false)) ? band_helper::is_band(KL, KU, A, uword(32)) : false;
    // const bool is_sym  = ((no_sym == false) && (is_band == false)) ? A.is_symmetric() : false;
    
    if(fast)
      {
      if(is_band == false)
        {
        arma_extra_debug_print("glue_solve_gen::apply(): fast + dense");
        
        status = auxlib::solve_square_fast(out, A, B_expr.get_ref());  // A is overwritten
        }
      else
        {
        if( (KL == 1) && (KU == 1) )
          {
          arma_extra_debug_print("glue_solve_gen::apply(): fast + tridiagonal");
          
          status = auxlib::solve_tridiag_fast(out, A, B_expr.get_ref());
          }
        else
          {
          arma_extra_debug_print("glue_solve_gen::apply(): fast + band");
          
          status = auxlib::solve_band_fast(out, A, KL, KU, B_expr.get_ref());
          }
        }
      }
    else
      {
      if(is_band == false)
        {
        arma_extra_debug_print("glue_solve_gen::apply(): refine + dense");
        
        status = auxlib::solve_square_refine(out, rcond, A, B_expr, equilibrate);  // A is overwritten
        }
      else
        {
        if( (KL == 1) && (KU == 1) && (equilibrate == false) )
          {
          arma_extra_debug_print("glue_solve_gen::apply(): refine + tridiagonal");
          
          status = auxlib::solve_tridiag_refine(out, rcond, A, B_expr);
          }
        else
          {
          arma_extra_debug_print("glue_solve_gen::apply(): refine + band");
          
          status = auxlib::solve_band_refine(out, rcond, A, KL, KU, B_expr, equilibrate);
          }
        }
      }
    
    
      // if((status == true) && (rcond > T(0)) && (rcond <= (T(0.5)*std::numeric_limits<T>::epsilon())) )
      //   {
      //   // arma_debug_warn("solve(): system seems singular to working precision (rcond: ", rcond, ")");
      //   
      //   status = approx_equal( A_expr.get_ref() * out, B_expr.get_ref(), "reldiff", T(2)*std::numeric_limits<T>::epsilon() );
      //   
      //   if(status == true)
      //     {
      //     // solution seems okay, but warn the user about rcond
      //     arma_debug_warn("solve(): system seems singular to working precision (rcond: ", rcond, ")");
      //     }
      //   }
    
    
    if( (status == false) && (no_approx == false) )
      {
      arma_extra_debug_print("glue_solve_gen::apply(): solving rank deficient system");
      
      arma_debug_warn("solve(): system seems singular (rcond: ", rcond, "); attempting approx solution");
      
      // TODO: rather than using AA, conditionally recreate A and use it;
      // TODO: conditionally recreate A: have a separate state flag which indicates whether A was previously overwritten
      
      A.reset();
      
      Mat<eT> AA = A_expr.get_ref();
      status = auxlib::solve_approx_svd(out, AA, B_expr.get_ref());  // AA is overwritten
      }
    }
  else
    {
    arma_extra_debug_print("glue_solve_gen::apply(): detected non-square system");
    
    if(equilibrate)  { arma_debug_warn( "solve(): option 'equilibrate' ignored for non-square matrix" ); }
    
    if(fast)
      {
      status = auxlib::solve_approx_fast(out, A, B_expr.get_ref());  // A is overwritten
      
      if(status == false)
        {
        Mat<eT> AA = A_expr.get_ref();
        
        status = auxlib::solve_approx_svd(out, AA, B_expr.get_ref());  // AA is overwritten
        }
      }
    else
      {
      status = auxlib::solve_approx_svd(out, A, B_expr.get_ref());  // A is overwritten
      }
    }
  
  
  if(status == false)  { out.soft_reset(); }
  
  return status;
  }



//
// glue_solve_tri


template<typename T1, typename T2>
inline
void
glue_solve_tri::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_solve_tri>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = glue_solve_tri::apply( out, X.A, X.B, X.aux_uword );
  
  if(status == false)
    {
    arma_stop_runtime_error("solve(): solution not found");
    }
  }



template<typename eT, typename T1, typename T2>
inline
bool
glue_solve_tri::apply(Mat<eT>& out, const Base<eT,T1>& A_expr, const Base<eT,T2>& B_expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  const bool fast        = bool(flags & solve_opts::flag_fast       );
  const bool equilibrate = bool(flags & solve_opts::flag_equilibrate);
  const bool no_approx   = bool(flags & solve_opts::flag_no_approx  );
  const bool triu        = bool(flags & solve_opts::flag_triu       );
  const bool tril        = bool(flags & solve_opts::flag_tril       );
  
  arma_extra_debug_print("glue_solve_tri::apply(): enabled flags:");
  
  if(fast       )  { arma_extra_debug_print("fast");        }
  if(equilibrate)  { arma_extra_debug_print("equilibrate"); }
  if(no_approx  )  { arma_extra_debug_print("no_approx");   }
  if(triu       )  { arma_extra_debug_print("triu");        }
  if(tril       )  { arma_extra_debug_print("tril");        }
  
  if(equilibrate)  { arma_debug_warn("solve(): option 'equilibrate' ignored for triangular matrices"); }
  
  bool status = false;
  
  const uword layout = (triu) ? uword(0) : uword(1);
  
  const quasi_unwrap<T1> U(A_expr.get_ref());
  const Mat<eT>& A     = U.M;
  
  arma_debug_check( (A.is_square() == false), "solve(): matrix marked as triangular must be square sized" );
  
  if(U.is_alias(out))
    {
    Mat<eT> tmp;
    
    status = auxlib::solve_tri(tmp, A, B_expr.get_ref(), layout);  // A is not modified
    
    out.steal_mem(tmp);
    }
  else
    {
    status = auxlib::solve_tri(out, A, B_expr.get_ref(), layout);  // A is not modified
    }
  
  
  if( (status == false) && (no_approx == false) )
    {
    arma_extra_debug_print("glue_solve_tri::apply(): solving rank deficient system");
    
    arma_debug_warn("solve(): system seems singular; attempting approx solution");
    
    Mat<eT> triA = (triu) ? trimatu(A) : trimatl(A);  // trimatu() and trimatl() return the same type
    
    status = auxlib::solve_approx_svd(out, triA, B_expr.get_ref());  // triA is overwritten
    }
  
  
  if(status == false)  { out.soft_reset(); }
  
  return status;
  }



//! @}
